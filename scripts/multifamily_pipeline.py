#!/usr/bin/env python3
"""
scripts/multifamily_pipeline.py - Step 2: Multi-Family OOF, Clustering Test, Scorer v2, Hole Fill,
and Final Submissions A (Scorer v2) & B (Threshold Rule).

All evaluations run over ALL 707 OOF images (including 13 empty-GT observations)
and report 1,000-iteration paired bootstrap (seed=2026) at the observation level.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu
from scipy.ndimage import binary_fill_holes, label

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, find_dataset_paths
from scripts.instance_scorer import (
    FEATURE_NAMES,
    build_gt_lookup,
    compute_candidate_targets,
    encode_crop_to_rle,
)

FRAME_SIZE = 2048

SCORER_V2_FEATURES = [
    "cluster_score",
    "num_views",
    "member_score_mean",
    "member_score_std",
    "prob_mean",
    "prob_std",
    "prob_sharpness_core",
    "prob_sharpness_iqr",
    "log_area",
    "bbox_fill",
    "aspect_ratio",
    "rank_in_image",
    "rel_score_to_max",
    "n_cands_gt_05",
    # Multi-family agreement & overlap context features:
    "score_clahe",
    "score_a5",
    "score_dice",
    "family_score_std",
    "n_families_active",
    "unoccluded_area_ratio",
    "max_iou_higher_rank",
]

SCORER_V2_VIEW_INVARIANT_FEATURES = [
    "cluster_score",
    "member_score_mean",
    "prob_mean",
    "prob_std",
    "prob_sharpness_core",
    "prob_sharpness_iqr",
    "log_area",
    "bbox_fill",
    "aspect_ratio",
    "rank_in_image",
    "rel_score_to_max",
    "n_cands_gt_05",
    # Multi-family agreement & overlap context features:
    "score_clahe",
    "score_a5",
    "score_dice",
    "family_score_std",
    "n_families_active",
    "unoccluded_area_ratio",
    "max_iou_higher_rank",
]


def drop_os_page_cache(file_path: str):
    """Evicts file from Linux page cache so WSL2 RAM stays flat."""
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(file_path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def build_gt_rles_by_obs(gt_df: pd.DataFrame) -> Dict[str, List[List[str]]]:
    """Maps each observation stem to its list of per-annotator GT RLE lists."""
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_by_obs: Dict[str, List[List[str]]] = defaultdict(list)
    for ai, group in gt_df.groupby(gt_ai, sort=False):
        obs = ai.split("-", 1)[1]
        gt_by_obs[obs].append(group["segmentation_rle"].tolist())
    return gt_by_obs


def eval_single_obs_rles(annotator_gt_lists: List[List[str]], pred_rles: List[str]) -> dict:
    """Computes exact [iou_sum, tp, fp, fn] summed across all annotator records of one observation."""
    iou_s, tp, fp, fn = 0.0, 0, 0, 0
    for gt_rles in annotator_gt_lists:
        m = pq_eval.iou_matrix(gt_rles, pred_rles)
        n_gt, n_pred = m.shape
        if n_gt == 0:
            fp += n_pred
        elif n_pred == 0:
            fn += n_gt
        else:
            hit = m > pq_eval.IOU_THRESHOLD
            iou_s += float(m[hit].sum())
            tp += int(hit.sum())
            fp += int((hit.sum(axis=0) == 0).sum())
            fn += int((hit.sum(axis=1) == 0).sum())
    return {"iou_sum": iou_s, "tp": tp, "fp": fp, "fn": fn}


def paired_bootstrap_from_per_obs(
    per_obs_a: Dict[str, dict],
    per_obs_b: Dict[str, dict],
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict:
    """
    Runs 1,000-iteration paired bootstrap over all 707 observations using precomputed
    per-observation {'iou_sum', 'tp', 'fp', 'fn'} dicts.
    """
    all_obs = sorted(list(per_obs_a.keys()))
    arr_a = np.array([[per_obs_a[o]["iou_sum"], per_obs_a[o]["tp"], per_obs_a[o]["fp"], per_obs_a[o]["fn"]] for o in all_obs], dtype=np.float64)
    arr_b = np.array([[per_obs_b[o]["iou_sum"], per_obs_b[o]["tp"], per_obs_b[o]["fp"], per_obs_b[o]["fn"]] for o in all_obs], dtype=np.float64)

    def calc_metrics(arr_sum):
        iou_s, tp, fp, fn = arr_sum
        denom = tp + 0.5 * fp + 0.5 * fn
        pq = float(iou_s / denom) if denom > 0 else 0.0
        sq = float(iou_s / tp) if tp > 0 else 0.0
        rq = float(tp / denom) if denom > 0 else 0.0
        return {"pq": pq, "sq": sq, "rq": rq, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    full_a = calc_metrics(arr_a.sum(axis=0))
    full_b = calc_metrics(arr_b.sum(axis=0))
    point_delta = full_b["pq"] - full_a["pq"]

    rng = np.random.default_rng(seed)
    n_obs = len(all_obs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b_i in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        sa = arr_a[idx].sum(axis=0)
        sb = arr_b[idx].sum(axis=0)
        da = sa[1] + 0.5 * sa[2] + 0.5 * sa[3]
        db = sb[1] + 0.5 * sb[2] + 0.5 * sb[3]
        pq_a = sa[0] / da if da > 0 else 0.0
        pq_b = sb[0] / db if db > 0 else 0.0
        deltas[b_i] = pq_b - pq_a

    return {
        "full_a": full_a,
        "full_b": full_b,
        "point_delta": float(point_delta),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "p_b_better": float((deltas > 0).mean()),
    }


def fast_crop_mask_iou(c1: dict, c2: dict) -> float:
    """
    Exact mask IoU between two candidates at mask_thresh=0.5 using intersection box only.
    ~50x faster than union-box allocation because non-overlapping boxes exit in 1us.
    """
    b1 = c1["box"]
    b2 = c2["box"]
    ix1 = max(int(b1[0]), int(b2[0]))
    iy1 = max(int(b1[1]), int(b2[1]))
    ix2 = min(int(b1[2]), int(b2[2]))
    iy2 = min(int(b1[3]), int(b2[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    bin1 = c1.get("_bin05")
    if bin1 is None:
        bin1 = (c1["prob_crop"] >= 0.5)
        c1["_bin05"] = bin1
        c1["_area05"] = int(bin1.sum())
    bin2 = c2.get("_bin05")
    if bin2 is None:
        bin2 = (c2["prob_crop"] >= 0.5)
        c2["_bin05"] = bin2
        c2["_area05"] = int(bin2.sum())

    sub1 = bin1[iy1 - int(b1[1]) : iy2 - int(b1[1]), ix1 - int(b1[0]) : ix2 - int(b1[0])]
    sub2 = bin2[iy1 - int(b2[1]) : iy2 - int(b2[1]), ix1 - int(b2[0]) : ix2 - int(b2[0])]
    inter = int((sub1 & sub2).sum())
    if inter == 0:
        return 0.0
    union = c1["_area05"] + c2["_area05"] - inter
    return float(inter) / float(union) if union > 0 else 0.0


def weighted_mask_fusion_tagged(
    candidate_list: List[dict],
    num_views: int = 12,
    iou_thresh: float = 0.40,
    family_views: Optional[Dict[str, int]] = None,
    use_running_mean_iou: bool = False,
) -> List[dict]:
    """
    Weighted Mask Fusion (WMF) that records per-family score contributions
    and optionally uses running-mean cluster mask IoU instead of leader-only IoU.
    """
    if not candidate_list:
        return []

    if family_views is None:
        family_views = {"clahe": 4, "a5": 4, "dice": 4}

    sorted_candidates = sorted(candidate_list, key=lambda c: c["score"], reverse=True)
    clusters: List[List[dict]] = []
    running_reps: List[dict] = []

    for cand in sorted_candidates:
        best_cluster_idx = -1
        best_iou = -1.0
        for c_idx, cluster in enumerate(clusters):
            rep = running_reps[c_idx] if use_running_mean_iou else cluster[0]
            iou = fast_crop_mask_iou(cand, rep)
            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = c_idx

        if best_iou > iou_thresh:
            clusters[best_cluster_idx].append(cand)
            if use_running_mean_iou:
                cl = clusters[best_cluster_idx]
                s_sum = sum(x["score"] for x in cl)
                all_b = np.stack([x["box"] for x in cl], axis=0)
                ux1, uy1 = int(all_b[:, 0].min()), int(all_b[:, 1].min())
                ux2, uy2 = int(all_b[:, 2].max()), int(all_b[:, 3].max())
                f_crop = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.float32)
                for x in cl:
                    w = x["score"] / s_sum
                    b = x["box"]
                    f_crop[b[1] - uy1 : b[3] - uy1, b[0] - ux1 : b[2] - ux1] += w * x["prob_crop"]
                running_reps[best_cluster_idx] = {
                    "box": np.array([ux1, uy1, ux2, uy2], dtype=np.int32),
                    "prob_crop": f_crop,
                    "score": s_sum,
                }
        else:
            clusters.append([cand])
            if use_running_mean_iou:
                running_reps.append(cand)

    fused_clusters = []
    for cluster in clusters:
        scores = [float(c["score"]) for c in cluster]
        score_sum = sum(scores)
        cluster_score = score_sum / float(num_views)

        fam_sums = defaultdict(float)
        for c in cluster:
            fam = c.get("family", "clahe")
            fam_sums[fam] += float(c["score"])

        fam_scores = {}
        for fam, fv in family_views.items():
            fam_scores[fam] = fam_sums[fam] / float(max(1, fv))

        all_boxes = np.stack([c["box"] for c in cluster], axis=0)
        ux1 = int(np.floor(all_boxes[:, 0].min()))
        uy1 = int(np.floor(all_boxes[:, 1].min()))
        ux2 = int(np.ceil(all_boxes[:, 2].max()))
        uy2 = int(np.ceil(all_boxes[:, 3].max()))
        union_box = np.array([ux1, uy1, ux2, uy2], dtype=np.int32)

        uh = uy2 - uy1
        uw = ux2 - ux1
        fused_prob_crop = np.zeros((uh, uw), dtype=np.float32)

        for c in cluster:
            w = float(c["score"]) / score_sum
            b = c["box"]
            fused_prob_crop[b[1] - uy1 : b[3] - uy1, b[0] - ux1 : b[2] - ux1] += w * c["prob_crop"]

        fused_clusters.append({
            "score": float(cluster_score),
            "prob_crop": fused_prob_crop,
            "box": union_box,
            "cluster_size": len(cluster),
            "member_scores": scores,
            "family_scores": fam_scores,
        })

    fused_clusters.sort(key=lambda x: x["score"], reverse=True)
    return fused_clusters


def fill_holes_in_crop(binary_crop: np.ndarray, max_hole_area: Optional[int] = None) -> np.ndarray:
    """
    Fills internal holes inside a binary mask crop.
    If max_hole_area is None, fills all internal holes via binary_fill_holes.
    If max_hole_area is specified, only fills holes with pixel count <= max_hole_area.
    """
    filled = binary_fill_holes(binary_crop)
    if max_hole_area is None:
        return filled
    holes = filled & (~binary_crop)
    hole_pixels = int(holes.sum())
    if hole_pixels == 0:
        return binary_crop
    if hole_pixels <= max_hole_area:
        return filled
    labeled_holes, num_features = label(holes)
    if num_features == 0:
        return binary_crop
    out = binary_crop.copy()
    sizes = np.bincount(labeled_holes.ravel())
    keep_labels = np.where((sizes > 0) & (sizes <= max_hole_area))[0]
    keep_labels = keep_labels[keep_labels > 0]
    if len(keep_labels) > 0:
        out |= np.isin(labeled_holes, keep_labels)
    return out


def compute_binary_crop_iou(box1: np.ndarray, crop1: np.ndarray, box2: np.ndarray, crop2: np.ndarray) -> float:
    """Fast IoU between two binary crops given their bounding boxes."""
    ix1 = max(int(box1[0]), int(box2[0]))
    iy1 = max(int(box1[1]), int(box2[1]))
    ix2 = min(int(box1[2]), int(box2[2]))
    iy2 = min(int(box1[3]), int(box2[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    sub1 = crop1[iy1 - int(box1[1]) : iy2 - int(box1[1]), ix1 - int(box1[0]) : ix2 - int(box1[0])]
    sub2 = crop2[iy1 - int(box2[1]) : iy2 - int(box2[1]), ix1 - int(box2[0]) : ix2 - int(box2[0])]
    inter = int((sub1 & sub2).sum())
    if inter == 0:
        return 0.0
    union = int(crop1.sum()) + int(crop2.sum()) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def load_fused_oof_pool(
    folds_df: pd.DataFrame,
    family_specs: List[Tuple[str, str, int, float]],
    iou_thresh: float = 0.40,
    use_running_mean_iou: bool = False,
    min_cluster_score: float = 0.15,
) -> Dict[str, List[dict]]:
    """
    Streams through all 707 observations across 5 folds, fuses raw candidates from
    family_specs = [(fam_name, oof_dir, views, prob_shift), ...], keeps clusters with
    score >= min_cluster_score, and immediately frees raw candidate tensors + OS page cache.
    """
    num_views_total = sum(v for _, _, v, _ in family_specs)
    family_views = {fam: v for fam, _, v, _ in family_specs}
    fused_by_obs: Dict[str, List[dict]] = {}

    for f in range(5):
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        for obs in val_obs:
            combined_cands = []
            for fam_name, oof_dir, _, prob_shift in family_specs:
                pt_path = os.path.join(oof_dir, f"fold{f}", f"{obs}.pt")
                if not os.path.isfile(pt_path):
                    continue
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                raw_cands = data.get("candidates", [])
                for c in raw_cands:
                    pc = c["prob_crop"]
                    if prob_shift != 0.0:
                        pc = np.clip(pc.astype(np.float32) + prob_shift, 0.0, 1.0)
                    combined_cands.append({
                        "box": c["box"],
                        "score": float(c["score"]),
                        "prob_crop": pc,
                        "family": fam_name,
                    })
                del data, raw_cands
                drop_os_page_cache(pt_path)

            if not combined_cands:
                fused_by_obs[obs] = []
                continue

            fused = weighted_mask_fusion_tagged(
                combined_cands,
                num_views=num_views_total,
                iou_thresh=iou_thresh,
                family_views=family_views,
                use_running_mean_iou=use_running_mean_iou,
            )
            fused_by_obs[obs] = [c for c in fused if c["score"] >= min_cluster_score]
            del combined_cands, fused

    gc.collect()
    return fused_by_obs


def evaluate_threshold_grid_nested_cv(
    fused_by_obs: Dict[str, List[dict]],
    folds_df: pd.DataFrame,
    gt_by_obs: Dict[str, List[List[str]]],
    score_grid: List[float],
    mask_grid: List[float],
    min_area: int = 10,
    area_retention: float = 0.50,
) -> dict:
    """
    Uses the prefix-invariance of descending-score canvas painting:
    For each tau_mask in mask_grid, paints the disjoint canvas ONCE per observation for
    candidates >= min(score_grid) and computes pq_eval.iou_matrix(gt_rles, painted_rles) ONCE!
    Then evaluating any tau_score in score_grid slices `iou_mat[:, :k_keep]` in microseconds!
    """
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    min_grid_sc = min(score_grid)

    # cache_by_mk[mk][obs] = (painted_scores_array, list_of_iou_matrices)
    cache_by_mk: Dict[float, Dict[str, Tuple[np.ndarray, List[np.ndarray]]]] = {}
    grid_fold_totals: Dict[Tuple[float, float], Dict[int, List[float]]] = {}

    for mk in mask_grid:
        mk_cache = {}
        for f in range(5):
            for obs in obs_by_fold[f]:
                clusters = fused_by_obs.get(obs, [])
                occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                painted_rles = []
                painted_scores = []
                for c in clusters:
                    sc = float(c["score"])
                    if sc < min_grid_sc:
                        break
                    box = c["box"]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    bc = (c["prob_crop"] >= mk)
                    oa = int(bc.sum())
                    if oa < min_area:
                        continue
                    sub_occ = occupied[y1:y2, x1:x2]
                    clean_crop = bc & (~sub_occ)
                    ca = int(clean_crop.sum())
                    if ca < min_area or (ca / float(oa)) < area_retention:
                        continue
                    occupied[y1:y2, x1:x2] |= clean_crop
                    painted_rles.append(encode_crop_to_rle(box, clean_crop))
                    painted_scores.append(sc)

                annot_gts = gt_by_obs.get(obs, [])
                iou_mats = [pq_eval.iou_matrix(gt_rles, painted_rles) for gt_rles in annot_gts]
                mk_cache[obs] = (np.asarray(painted_scores, dtype=np.float64), iou_mats)
        cache_by_mk[mk] = mk_cache

        for sc in score_grid:
            f_tots = {}
            for f in range(5):
                iou_s, tp, fp, fn = 0.0, 0, 0, 0
                for obs in obs_by_fold[f]:
                    scores_arr, iou_mats = mk_cache[obs]
                    k_keep = int(np.sum(scores_arr >= sc))
                    for m_full in iou_mats:
                        m = m_full[:, :k_keep]
                        n_gt, n_pred = m.shape
                        if n_gt == 0:
                            fp += n_pred
                        elif n_pred == 0:
                            fn += n_gt
                        else:
                            hit = m > pq_eval.IOU_THRESHOLD
                            iou_s += float(m[hit].sum())
                            tp += int(hit.sum())
                            fp += int((hit.sum(axis=0) == 0).sum())
                            fn += int((hit.sum(axis=1) == 0).sum())
                f_tots[f] = [iou_s, tp, fp, fn]
            grid_fold_totals[(sc, mk)] = f_tots

    def agg_metrics(iou_s, tp, fp, fn):
        sq = iou_s / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_sum": float(iou_s)}

    def get_per_obs_from_cache(fold_sc_mk: Dict[int, Tuple[float, float]]) -> Dict[str, dict]:
        per_obs: Dict[str, dict] = {}
        for f in range(5):
            sc, mk = fold_sc_mk[f]
            mk_cache = cache_by_mk[mk]
            for obs in obs_by_fold[f]:
                scores_arr, iou_mats = mk_cache[obs]
                k_keep = int(np.sum(scores_arr >= sc))
                iou_s, tp, fp, fn = 0.0, 0, 0, 0
                for m_full in iou_mats:
                    m = m_full[:, :k_keep]
                    n_gt, n_pred = m.shape
                    if n_gt == 0:
                        fp += n_pred
                    elif n_pred == 0:
                        fn += n_gt
                    else:
                        hit = m > pq_eval.IOU_THRESHOLD
                        iou_s += float(m[hit].sum())
                        tp += int(hit.sum())
                        fp += int((hit.sum(axis=0) == 0).sum())
                        fn += int((hit.sum(axis=1) == 0).sum())
                per_obs[obs] = {"iou_sum": iou_s, "tp": tp, "fp": fp, "fn": fn}
        return per_obs

    default_key = (0.85, 0.50) if (0.85, 0.50) in grid_fold_totals else (score_grid[-1], 0.50)
    def_tots = [sum(grid_fold_totals[default_key][f][i] for f in range(5)) for i in range(4)]
    default_res = agg_metrics(*def_tots)

    best_global_key = None
    best_global_res = {"pq": -1.0}
    for key, f_tots in grid_fold_totals.items():
        tots = [sum(f_tots[f][i] for f in range(5)) for i in range(4)]
        res = agg_metrics(*tots)
        if res["pq"] > best_global_res["pq"]:
            best_global_res = res
            best_global_key = key

    lofo_fold_choices = {}
    lofo_fold_sc_mk = {}
    lofo_iou, lofo_tp, lofo_fp, lofo_fn = 0.0, 0, 0, 0

    for val_f in range(5):
        best_tr_key = None
        best_tr_pq = -1.0
        for key, f_tots in grid_fold_totals.items():
            tr_tots = [sum(f_tots[f][i] for f in range(5) if f != val_f) for i in range(4)]
            tr_res = agg_metrics(*tr_tots)
            if tr_res["pq"] > best_tr_pq:
                best_tr_pq = tr_res["pq"]
                best_tr_key = key

        val_res = agg_metrics(*grid_fold_totals[best_tr_key][val_f])
        lofo_fold_sc_mk[val_f] = best_tr_key
        lofo_fold_choices[val_f] = {
            "tau_score": best_tr_key[0],
            "tau_mask": best_tr_key[1],
            "train_pq": best_tr_pq,
            "val_pq": val_res["pq"],
            "val_sq": val_res["sq"],
            "val_rq": val_res["rq"],
        }
        lofo_iou += grid_fold_totals[best_tr_key][val_f][0]
        lofo_tp += grid_fold_totals[best_tr_key][val_f][1]
        lofo_fp += grid_fold_totals[best_tr_key][val_f][2]
        lofo_fn += grid_fold_totals[best_tr_key][val_f][3]

    lofo_pooled = agg_metrics(lofo_iou, lofo_tp, lofo_fp, lofo_fn)
    lofo_per_obs = get_per_obs_from_cache(lofo_fold_sc_mk)
    del cache_by_mk

    return {
        "default_085_050": default_res,
        "global_best_tau_score": best_global_key[0],
        "global_best_tau_mask": best_global_key[1],
        "global_best": best_global_res,
        "nested_cv": lofo_pooled,
        "nested_cv_folds": lofo_fold_choices,
        "nested_cv_per_obs": lofo_per_obs,
    }


def extract_scorer_v2_dataset(
    fused_by_obs: Dict[str, List[dict]],
    folds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    train_json_path: str,
    num_views_total: int = 12,
    min_cluster_score: float = 0.15,
    tau_mask: float = 0.50,
    min_area: int = 10,
    use_tight_bbox: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """
    Extracts the 21 Scorer v2 features (14 base + 5 multi-family agreement + 2 canvas overlap context)
    and target y for all candidates with S >= min_cluster_score across all 707 observations.
    Uses tight binary-mask bounding box when use_tight_bbox=True so bbox_fill and aspect_ratio
    are 100% invariant to WMF union-box growth across view counts.
    """
    k_by_obs, gt_records_by_obs = build_gt_lookup(gt_df, train_json_path)
    rows = []
    pool_by_obs: Dict[str, List[dict]] = {}

    for f in range(5):
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        for obs in val_obs:
            fused = fused_by_obs.get(obs, [])
            if not fused:
                pool_by_obs[obs] = []
                continue

            all_scores = [float(c["score"]) for c in fused]
            max_score_in_img = max(all_scores) if all_scores else 1e-6
            n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

            obs_candidates = []
            cand_rles_for_target = []
            higher_rank_boxes_and_crops = []
            sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

            for rank_0, c in enumerate(fused):
                sc = float(c["score"])
                if sc < min_cluster_score:
                    continue

                prob_crop = c["prob_crop"]
                box = c["box"]
                binary_crop = (prob_crop >= tau_mask)
                area = int(binary_crop.sum())
                if area < min_area:
                    continue

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if use_tight_bbox:
                    ys, xs = np.nonzero(binary_crop)
                    bw = max(1, int(xs.max() - xs.min() + 1))
                    bh = max(1, int(ys.max() - ys.min() + 1))
                else:
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                bbox_area = float(bw * bh)

                masked_probs = prob_crop[binary_crop].astype(np.float32)
                prob_mean = float(masked_probs.mean())
                prob_std = float(masked_probs.std()) if area > 1 else 0.0
                core_pixels = int((masked_probs >= 0.75).sum())
                prob_sharpness_core = float(core_pixels) / float(area)
                q90, q10 = np.percentile(masked_probs, [90, 10])
                prob_sharpness_iqr = float(q90 - q10)

                m_scores = c.get("member_scores", [sc])
                m_arr = np.asarray(m_scores, dtype=np.float32)
                member_mean = float(m_arr.mean()) if len(m_arr) > 0 else sc
                member_std = float(m_arr.std()) if len(m_arr) > 1 else 0.0
                norm_views = float(c["cluster_size"]) * (4.0 / float(num_views_total))

                log_area = float(np.log10(area + 1.0))
                bbox_fill = float(area) / bbox_area
                aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
                rank_in_image = rank_0 + 1
                rel_score_to_max = sc / max(1e-6, max_score_in_img)

                fam_sc = c.get("family_scores", {})
                sc_clahe = float(fam_sc.get("clahe", sc))
                sc_a5 = float(fam_sc.get("a5", sc))
                sc_dice = float(fam_sc.get("dice", sc))
                fam_arr = np.array([sc_clahe, sc_a5, sc_dice], dtype=np.float32)
                family_score_std = float(fam_arr.std())
                n_families_active = float(sum(1 for x in fam_arr if x > 0.0))

                sub_occ = sim_occupied[y1:y2, x1:x2]
                unocc_area = int((binary_crop & (~sub_occ)).sum())
                unoccluded_area_ratio = float(unocc_area) / float(area)

                max_iou_higher = 0.0
                for h_box, h_crop in higher_rank_boxes_and_crops:
                    iou_h = compute_binary_crop_iou(box, binary_crop, h_box, h_crop)
                    if iou_h > max_iou_higher:
                        max_iou_higher = iou_h

                if sc >= 0.50 and unoccluded_area_ratio >= 0.50:
                    sim_occupied[y1:y2, x1:x2] |= (binary_crop & (~sub_occ))
                higher_rank_boxes_and_crops.append((box, binary_crop))

                rle_str = encode_crop_to_rle(box, binary_crop)
                cand_rles_for_target.append(rle_str)

                rows.append({
                    "obs": obs,
                    "fold": f,
                    "cand_idx": len(obs_candidates),
                    "cluster_score": sc,
                    "num_views": norm_views,
                    "member_score_mean": member_mean,
                    "member_score_std": member_std,
                    "prob_mean": prob_mean,
                    "prob_std": prob_std,
                    "prob_sharpness_core": prob_sharpness_core,
                    "prob_sharpness_iqr": prob_sharpness_iqr,
                    "log_area": log_area,
                    "bbox_fill": bbox_fill,
                    "aspect_ratio": aspect_ratio,
                    "rank_in_image": float(rank_in_image),
                    "rel_score_to_max": rel_score_to_max,
                    "n_cands_gt_05": float(n_cands_gt_05),
                    "score_clahe": sc_clahe,
                    "score_a5": sc_a5,
                    "score_dice": sc_dice,
                    "family_score_std": family_score_std,
                    "n_families_active": n_families_active,
                    "unoccluded_area_ratio": unoccluded_area_ratio,
                    "max_iou_higher_rank": max_iou_higher,
                })

                obs_candidates.append({
                    "box": box,
                    "binary_crop": binary_crop,
                    "orig_area": area,
                    "score": sc,
                    "pred_y": 0.0,
                })

            targets = compute_candidate_targets(cand_rles_for_target, obs, k_by_obs, gt_records_by_obs)
            start_idx = len(rows) - len(obs_candidates)
            for idx_in_obs, y_val in enumerate(targets):
                rows[start_idx + idx_in_obs]["target_y"] = float(y_val)

            pool_by_obs[obs] = obs_candidates

    return pd.DataFrame(rows), pool_by_obs


def evaluate_Pool_at_tau(
    pool_by_obs: Dict[str, List[dict]],
    obs_list: List[str],
    gt_by_obs: Dict[str, List[List[str]]],
    tau: float,
    min_area: int = 10,
    area_retention: float = 0.50,
    hole_fill_mode: Optional[str] = None,  # None, "all", "small100"
) -> Dict[str, dict]:
    """
    Selects candidates with pred_y > tau, sorts descending by pred_y, optionally applies hole fill,
    and resolves overlaps via canvas painting. Returns per_obs {'iou_sum', 'tp', 'fp', 'fn'}.
    """
    per_obs: Dict[str, dict] = {}
    for obs in obs_list:
        cands = pool_by_obs.get(obs, [])
        kept = [c for c in cands if c["pred_y"] > tau]
        if not kept:
            per_obs[obs] = eval_single_obs_rles(gt_by_obs.get(obs, []), [])
            continue

        kept.sort(key=lambda x: x["pred_y"], reverse=True)
        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        kept_rles = []

        for c in kept:
            box = c["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            binary_crop = c["binary_crop"]
            if hole_fill_mode == "all":
                binary_crop = fill_holes_in_crop(binary_crop, max_hole_area=None)
            elif hole_fill_mode == "small100":
                binary_crop = fill_holes_in_crop(binary_crop, max_hole_area=100)

            orig_area = int(binary_crop.sum())
            if orig_area == 0:
                continue

            sub_occ = occupied[y1:y2, x1:x2]
            clean_crop = binary_crop & (~sub_occ)
            clean_area = int(clean_crop.sum())
            if clean_area < min_area or (clean_area / float(orig_area)) < area_retention:
                continue

            occupied[y1:y2, x1:x2] |= clean_crop
            kept_rles.append(encode_crop_to_rle(box, clean_crop))

        per_obs[obs] = eval_single_obs_rles(gt_by_obs.get(obs, []), kept_rles)
    return per_obs


def run_cross_fitted_scorer(
    feat_df: pd.DataFrame,
    pool_by_obs: Dict[str, List[dict]],
    folds_df: pd.DataFrame,
    gt_by_obs: Dict[str, List[List[str]]],
    feature_cols: List[str],
    seed: int = 2026,
) -> Tuple[dict, List[lgb.LGBMRegressor]]:
    """
    Runs 5-fold cross-fitted LightGBM regression using feature_cols, paints canvas ONCE per obs
    in descending order of pred_y, selects tau on the 4 training folds only (around PQ_train / 2 +- 0.04),
    and evaluates on the held-out fold.
    """
    oof_preds = np.zeros(len(feat_df), dtype=np.float32)
    if lgb is None:
        raise ImportError("lightgbm is required to run Scorer v2 cross-fitting (pip install lightgbm)")
    models: List[lgb.LGBMRegressor] = []
    gain_importances = np.zeros(len(feature_cols), dtype=np.float64)

    for val_f in range(5):
        tr_mask = (feat_df["fold"] != val_f).values
        va_mask = (feat_df["fold"] == val_f).values

        X_tr = feat_df.loc[tr_mask, feature_cols].values
        y_tr = feat_df.loc[tr_mask, "target_y"].values
        X_va = feat_df.loc[va_mask, feature_cols].values

        model = lgb.LGBMRegressor(
            objective="regression",
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            n_estimators=300,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed + val_f,
            verbose=-1,
            n_jobs=4,
        )
        model.fit(X_tr, y_tr)
        preds_va = np.clip(model.predict(X_va), 0.0, 1.0).astype(np.float32)
        oof_preds[va_mask] = preds_va
        models.append(model)
        gain_importances += model.booster_.feature_importance(importance_type="gain") / 5.0

    feat_df["pred_y"] = oof_preds
    for row in feat_df[["obs", "cand_idx", "pred_y"]].itertuples(index=False):
        pool_by_obs[row.obs][int(row.cand_idx)]["pred_y"] = float(row.pred_y)

    # Prefix-invariant canvas painting in descending order of pred_y (1 pass over 707 images)
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    obs_cache: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}

    for f in range(5):
        for obs in obs_by_fold[f]:
            cands = pool_by_obs.get(obs, [])
            sorted_cands = sorted(cands, key=lambda c: c["pred_y"], reverse=True)
            occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            painted_rles = []
            painted_scores = []
            for c in sorted_cands:
                b = c["box"]
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                bc = c["binary_crop"]
                oa = c["orig_area"]
                sub_occ = occupied[y1:y2, x1:x2]
                clean_crop = bc & (~sub_occ)
                ca = int(clean_crop.sum())
                if ca < 10 or (ca / float(oa)) < 0.50:
                    continue
                occupied[y1:y2, x1:x2] |= clean_crop
                painted_rles.append(encode_crop_to_rle(b, clean_crop))
                painted_scores.append(float(c["pred_y"]))
            annot_gts = gt_by_obs.get(obs, [])
            iou_mats = [pq_eval.iou_matrix(gt_rles, painted_rles) for gt_rles in annot_gts]
            obs_cache[obs] = (np.asarray(painted_scores, dtype=np.float64), iou_mats)

    tau_grid = [round(0.180 + 0.005 * k, 4) for k in range(19)]  # 0.180 .. 0.270
    fold_tau_metrics: Dict[int, Dict[float, dict]] = {f: {} for f in range(5)}
    for f in range(5):
        for tau in tau_grid:
            iou_s, tp, fp, fn = 0.0, 0, 0, 0
            for obs in obs_by_fold[f]:
                scores_arr, iou_mats = obs_cache[obs]
                k_keep = int(np.sum(scores_arr > tau))
                for m_full in iou_mats:
                    m = m_full[:, :k_keep]
                    n_gt, n_pred = m.shape
                    if n_gt == 0:
                        fp += n_pred
                    elif n_pred == 0:
                        fn += n_gt
                    else:
                        hit = m > pq_eval.IOU_THRESHOLD
                        iou_s += float(m[hit].sum())
                        tp += int(hit.sum())
                        fp += int((hit.sum(axis=0) == 0).sum())
                        fn += int((hit.sum(axis=1) == 0).sum())
            sq = iou_s / tp if tp > 0 else 0.0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
            fold_tau_metrics[f][tau] = {
                "pq": sq * rq,
                "sq": sq,
                "rq": rq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou_sum": iou_s,
            }

    scorer_per_obs: Dict[str, dict] = {}
    fold_results = {}
    tot_iou, tot_tp, tot_fp, tot_fn = 0.0, 0, 0, 0

    for val_f in range(5):
        best_tau = None
        best_tr_pq = -1.0
        for tau in tau_grid:
            tr_iou = sum(fold_tau_metrics[f][tau]["iou_sum"] for f in range(5) if f != val_f)
            tr_tp = sum(fold_tau_metrics[f][tau]["tp"] for f in range(5) if f != val_f)
            tr_fp = sum(fold_tau_metrics[f][tau]["fp"] for f in range(5) if f != val_f)
            tr_fn = sum(fold_tau_metrics[f][tau]["fn"] for f in range(5) if f != val_f)
            tr_sq = tr_iou / tr_tp if tr_tp > 0 else 0.0
            tr_rq = tr_tp / (tr_tp + 0.5 * tr_fp + 0.5 * tr_fn) if (tr_tp + tr_fp + tr_fn) > 0 else 0.0
            tr_pq = tr_sq * tr_rq
            if tr_pq > best_tr_pq:
                best_tr_pq = tr_pq
                best_tau = tau

        val_agg = fold_tau_metrics[val_f][best_tau]
        fold_results[val_f] = {
            "chosen_tau": best_tau,
            "train_pq": best_tr_pq,
            "val_pq": val_agg["pq"],
            "val_sq": val_agg["sq"],
            "val_rq": val_agg["rq"],
            "tp": val_agg["tp"],
            "fp": val_agg["fp"],
            "fn": val_agg["fn"],
        }
        tot_iou += val_agg["iou_sum"]
        tot_tp += val_agg["tp"]
        tot_fp += val_agg["fp"]
        tot_fn += val_agg["fn"]

        for obs in obs_by_fold[val_f]:
            scores_arr, iou_mats = obs_cache[obs]
            k_keep = int(np.sum(scores_arr > best_tau))
            iou_s, tp, fp, fn = 0.0, 0, 0, 0
            for m_full in iou_mats:
                m = m_full[:, :k_keep]
                n_gt, n_pred = m.shape
                if n_gt == 0:
                    fp += n_pred
                elif n_pred == 0:
                    fn += n_gt
                else:
                    hit = m > pq_eval.IOU_THRESHOLD
                    iou_s += float(m[hit].sum())
                    tp += int(hit.sum())
                    fp += int((hit.sum(axis=0) == 0).sum())
                    fn += int((hit.sum(axis=1) == 0).sum())
            scorer_per_obs[obs] = {"iou_sum": iou_s, "tp": tp, "fp": fp, "fn": fn}

    del obs_cache
    pooled_sq = tot_iou / tot_tp if tot_tp > 0 else 0.0
    pooled_rq = tot_tp / (tot_tp + 0.5 * tot_fp + 0.5 * tot_fn) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0
    pooled_pq = pooled_sq * pooled_rq

    total_gain = max(1e-9, gain_importances.sum())
    feat_imp = [
        {"feature": feature_cols[i], "gain_pct": float(gain_importances[i] / total_gain * 100.0)}
        for i in range(len(feature_cols))
    ]
    feat_imp.sort(key=lambda x: x["gain_pct"], reverse=True)
    mean_tau = float(np.mean([fold_results[f]["chosen_tau"] for f in range(5)]))

    return {
        "pooled": {
            "pq": float(pooled_pq),
            "sq": float(pooled_sq),
            "rq": float(pooled_rq),
            "tp": int(tot_tp),
            "fp": int(tot_fp),
            "fn": int(tot_fn),
            "iou_sum": float(tot_iou),
        },
        "mean_chosen_tau": mean_tau,
        "folds": fold_results,
        "per_obs": scorer_per_obs,
        "feature_importances": feat_imp,
    }, models


def print_boot_line(label: str, res_b: dict, boot: dict):
    print(
        f"  {label:<44} | PQ: {res_b['pq']:.4f} (SQ:{res_b['sq']:.4f}, RQ:{res_b['rq']:.4f}) | "
        f"dPQ: {boot['point_delta']:+.4f} [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}] "
        f"(P={boot['p_b_better']*100:.1f}%)"
    )


def generate_final_test_submissions_a_and_b(
    scorer_models: List[lgb.LGBMRegressor],
    scorer_feature_cols: List[str],
    scorer_tau: float,
    scorer_tau_mask: float,
    thresh_tau_score: float,
    thresh_tau_mask: float,
    dice_prob_shift: float = 0.0,
    hole_fill_mode: Optional[str] = None,
    cache_dir: str = "work/test_cache",
):
    """
    Generates Final Submissions A (Scorer v2) and B (Threshold Rule) on the test set
    for the exact same multi-family ensemble:
      - 8-Model Multi-Family Test Ensemble (5-fold 1536_clahe + f0_anchors5 + f0_dice + f0_2048)
      - AND 3-Model Matched 12-View Test Ensemble (f0_1536 + f0_anchors5 + f0_dice)
    Validates all generated CSVs with scripts/validate_submission.py and copies to Windows Downloads.
    """
    print("\n" + "=" * 92)
    print("  STEP 2.5: GENERATING FINAL SUBMISSIONS A (SCORER V2) & B (THRESHOLD RULE)")
    print("=" * 92)

    first_dir = os.path.join(cache_dir, "f0_1536")
    test_stems = sorted([Path(p).stem for p in os.listdir(first_dir) if p.endswith(".pt")])
    print(f"Found {len(test_stems)} cached test stems in {cache_dir}.")

    ensemble_configs = [
        (
            "multifam_8model",
            [
                ("f0_1536", "clahe", 0.0),
                ("f1_1536", "clahe", 0.0),
                ("f2_1536", "clahe", 0.0),
                ("f3_1536", "clahe", 0.0),
                ("f4_1536", "clahe", 0.0),
                ("f0_2048", "clahe", 0.0),
                ("f0_anchors5", "a5", 0.0),
                ("f0_dice", "dice", dice_prob_shift),
            ],
            32,
            {"clahe": 24, "a5": 4, "dice": 4},
        ),
        (
            "multifam_12view_matched",
            [
                ("f0_1536", "clahe", 0.0),
                ("f0_anchors5", "a5", 0.0),
                ("f0_dice", "dice", dice_prob_shift),
            ],
            12,
            {"clahe": 4, "a5": 4, "dice": 4},
        ),
    ]

    for ens_tag, model_list, num_views_total, fam_views in ensemble_configs:
        rows_A = []  # Submission A: Scorer v2
        rows_B = []  # Submission B: Threshold Rule

        for stem in test_stems:
            combined_cands = []
            for m_name, fam_name, p_shift in model_list:
                pt_path = os.path.join(cache_dir, m_name, f"{stem}.pt")
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                for c in data.get("candidates", []):
                    pc = c["prob_crop"]
                    if p_shift != 0.0:
                        pc = np.clip(pc.astype(np.float32) + p_shift, 0.0, 1.0)
                    combined_cands.append({
                        "box": c["box"],
                        "score": float(c["score"]),
                        "prob_crop": pc,
                        "family": fam_name,
                    })
                del data
                drop_os_page_cache(pt_path)

            fused = weighted_mask_fusion_tagged(
                combined_cands,
                num_views=num_views_total,
                iou_thresh=0.40,
                family_views=fam_views,
            )
            del combined_cands

            # --- B: Threshold Rule on Fused Clusters ---
            occupied_B = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            rles_B = []
            for c in fused:
                if c["score"] < thresh_tau_score:
                    break
                box = c["box"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bc = (c["prob_crop"] >= thresh_tau_mask)
                if hole_fill_mode == "all":
                    bc = fill_holes_in_crop(bc, max_hole_area=None)
                elif hole_fill_mode == "small100":
                    bc = fill_holes_in_crop(bc, max_hole_area=100)
                oa = int(bc.sum())
                if oa == 0:
                    continue
                sub_occ = occupied_B[y1:y2, x1:x2]
                clean_crop = bc & (~sub_occ)
                ca = int(clean_crop.sum())
                if ca < 10 or (ca / float(oa)) < 0.50:
                    continue
                occupied_B[y1:y2, x1:x2] |= clean_crop
                rles_B.append(encode_crop_to_rle(box, clean_crop))

            if not rles_B:
                rows_B.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
            else:
                for k, enc in enumerate(rles_B, start=1):
                    rows_B.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

            # --- A: Scorer v2 on Fused Clusters (S >= 0.15) ---
            all_scores = [float(c["score"]) for c in fused]
            max_score_in_img = max(all_scores) if all_scores else 1e-6
            n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

            feat_dicts = []
            cands_A = []
            higher_rank_boxes_and_crops = []
            sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

            for rank_0, c in enumerate(fused):
                sc = float(c["score"])
                if sc < 0.15:
                    continue
                prob_crop = c["prob_crop"]
                box = c["box"]
                binary_crop = (prob_crop >= scorer_tau_mask)
                area = int(binary_crop.sum())
                if area < 10:
                    continue

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                ys, xs = np.nonzero(binary_crop)
                bw = max(1, int(xs.max() - xs.min() + 1))
                bh = max(1, int(ys.max() - ys.min() + 1))
                bbox_area = float(bw * bh)

                masked_probs = prob_crop[binary_crop].astype(np.float32)
                prob_mean = float(masked_probs.mean())
                prob_std = float(masked_probs.std()) if area > 1 else 0.0
                core_pixels = int((masked_probs >= 0.75).sum())
                prob_sharpness_core = float(core_pixels) / float(area)
                q90, q10 = np.percentile(masked_probs, [90, 10])
                prob_sharpness_iqr = float(q90 - q10)

                m_scores = c.get("member_scores", [sc])
                m_arr = np.asarray(m_scores, dtype=np.float32)
                member_mean = float(m_arr.mean()) if len(m_arr) > 0 else sc
                member_std = float(m_arr.std()) if len(m_arr) > 1 else 0.0
                norm_views = float(c["cluster_size"]) * (4.0 / float(num_views_total))

                log_area = float(np.log10(area + 1.0))
                bbox_fill = float(area) / bbox_area
                aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
                rank_in_image = rank_0 + 1
                rel_score_to_max = sc / max(1e-6, max_score_in_img)

                fam_sc = c.get("family_scores", {})
                sc_clahe = float(fam_sc.get("clahe", sc))
                sc_a5 = float(fam_sc.get("a5", sc))
                sc_dice = float(fam_sc.get("dice", sc))
                fam_arr = np.array([sc_clahe, sc_a5, sc_dice], dtype=np.float32)
                family_score_std = float(fam_arr.std())
                n_families_active = float(sum(1 for x in fam_arr if x > 0.0))

                sub_occ = sim_occupied[y1:y2, x1:x2]
                unocc_area = int((binary_crop & (~sub_occ)).sum())
                unoccluded_area_ratio = float(unocc_area) / float(area)

                max_iou_higher = 0.0
                for h_box, h_crop in higher_rank_boxes_and_crops:
                    iou_h = compute_binary_crop_iou(box, binary_crop, h_box, h_crop)
                    if iou_h > max_iou_higher:
                        max_iou_higher = iou_h

                if sc >= 0.50 and unoccluded_area_ratio >= 0.50:
                    sim_occupied[y1:y2, x1:x2] |= (binary_crop & (~sub_occ))
                higher_rank_boxes_and_crops.append((box, binary_crop))

                feat_dicts.append({
                    "cluster_score": sc,
                    "num_views": norm_views,
                    "member_score_mean": member_mean,
                    "member_score_std": member_std,
                    "prob_mean": prob_mean,
                    "prob_std": prob_std,
                    "prob_sharpness_core": prob_sharpness_core,
                    "prob_sharpness_iqr": prob_sharpness_iqr,
                    "log_area": log_area,
                    "bbox_fill": bbox_fill,
                    "aspect_ratio": aspect_ratio,
                    "rank_in_image": float(rank_in_image),
                    "rel_score_to_max": rel_score_to_max,
                    "n_cands_gt_05": float(n_cands_gt_05),
                    "score_clahe": sc_clahe,
                    "score_a5": sc_a5,
                    "score_dice": sc_dice,
                    "family_score_std": family_score_std,
                    "n_families_active": n_families_active,
                    "unoccluded_area_ratio": unoccluded_area_ratio,
                    "max_iou_higher_rank": max_iou_higher,
                })
                cands_A.append({
                    "box": box,
                    "binary_crop": binary_crop,
                    "orig_area": area,
                    "pred_y": 0.0,
                })

            rles_A = []
            if cands_A:
                X_stem = pd.DataFrame(feat_dicts)[scorer_feature_cols].values
                pred_y_stem = np.mean([m.predict(X_stem) for m in scorer_models], axis=0)
                for c_item, py in zip(cands_A, pred_y_stem):
                    c_item["pred_y"] = float(py)

                kept_A = [c_item for c_item in cands_A if c_item["pred_y"] > scorer_tau]
                kept_A.sort(key=lambda x: x["pred_y"], reverse=True)
                occupied_A = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                for c_item in kept_A:
                    box = c_item["box"]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    bc = c_item["binary_crop"]
                    if hole_fill_mode == "all":
                        bc = fill_holes_in_crop(bc, max_hole_area=None)
                    elif hole_fill_mode == "small100":
                        bc = fill_holes_in_crop(bc, max_hole_area=100)
                    oa = int(bc.sum())
                    if oa == 0:
                        continue
                    sub_occ = occupied_A[y1:y2, x1:x2]
                    clean_crop = bc & (~sub_occ)
                    ca = int(clean_crop.sum())
                    if ca < 10 or (ca / float(oa)) < 0.50:
                        continue
                    occupied_A[y1:y2, x1:x2] |= clean_crop
                    rles_A.append(encode_crop_to_rle(box, clean_crop))

            if not rles_A:
                rows_A.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
            else:
                for k, enc in enumerate(rles_A, start=1):
                    rows_A.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

            del fused

        sub_A_path = f"work/oof/submission_A_scorer_v2_{ens_tag}.csv"
        sub_B_path = f"work/oof/submission_B_threshold_{ens_tag}.csv"
        df_A = pd.DataFrame(rows_A)
        df_B = pd.DataFrame(rows_B)
        df_A.to_csv(sub_A_path, index=False)
        df_B.to_csv(sub_B_path, index=False)

        non_empty_A = int((df_A["segmentation_rle"] != EMPTY_RLE).sum())
        non_empty_B = int((df_B["segmentation_rle"] != EMPTY_RLE).sum())
        print(f"  [{ens_tag}] Submission A (Scorer v2, tau={scorer_tau:.4f}): {non_empty_A} filaments -> {sub_A_path}")
        print(f"  [{ens_tag}] Submission B (Threshold, sc={thresh_tau_score:.3f}, mk={thresh_tau_mask:.3f}): {non_empty_B} filaments -> {sub_B_path}")

        dl_dir = os.environ.get("SOLAR_EXPORT_DIR") or os.environ.get("DOWNLOADS_DIR")
        for p in (sub_A_path, sub_B_path):
            subprocess.run([sys.executable, "scripts/validate_submission.py", "--sub", p], check=True)
            if dl_dir and os.path.isdir(dl_dir):
                shutil.copy2(p, os.path.join(dl_dir, os.path.basename(p)))


def main():
    argparse.ArgumentParser(
        description="Multi-family OOF evaluation, nested-CV threshold grid, Scorer v2, and submission checks."
    ).parse_args()
    t0 = time.time()
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    gt_df = pd.read_csv(paths["gt_all_csv"])
    gt_by_obs = build_gt_rles_by_obs(gt_df)

    score_grid = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875]
    mask_grid = [0.475, 0.500, 0.525, 0.550, 0.575]

    print("\n" + "#" * 92)
    print("  STEP 2.1: MULTI-FAMILY OOF WITH TAU_MASK GRID (707 IMAGES + PAIRED BOOTSTRAP)")
    print("#" * 92)

    print("\n[1/6] Loading 4-view 1536_clahe OOF...")
    pool_clahe = load_fused_oof_pool(folds_df, [("clahe", "work/oof", 4, 0.0)])
    eval_clahe = evaluate_threshold_grid_nested_cv(pool_clahe, folds_df, gt_by_obs, score_grid, mask_grid)
    base_per_obs = eval_clahe["nested_cv_per_obs"]
    print(
        f"  1536_clahe (4v)   | Default(0.85,0.50): {eval_clahe['default_085_050']['pq']:.4f} | "
        f"GlobalBest({eval_clahe['global_best_tau_score']:.3f},{eval_clahe['global_best_tau_mask']:.3f}): {eval_clahe['global_best']['pq']:.4f} | "
        f"Nested-CV: {eval_clahe['nested_cv']['pq']:.4f} (SQ:{eval_clahe['nested_cv']['sq']:.4f}, RQ:{eval_clahe['nested_cv']['rq']:.4f})"
    )

    feat_4v, obs_pool_4v = extract_scorer_v2_dataset(
        pool_clahe, folds_df, gt_df, paths["train_json"], num_views_total=4, tau_mask=0.50
    )
    scorer_4v_res, _ = run_cross_fitted_scorer(feat_4v, obs_pool_4v, folds_df, gt_by_obs, FEATURE_NAMES)
    scorer_v1_4v_per_obs = scorer_4v_res["per_obs"]
    del pool_clahe, feat_4v, obs_pool_4v
    gc.collect()

    print("[2/6] Loading 4-view 1536_anchors5 OOF...")
    pool_a5 = load_fused_oof_pool(folds_df, [("a5", "work/oof_a5", 4, 0.0)])
    eval_a5 = evaluate_threshold_grid_nested_cv(pool_a5, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_a5 = paired_bootstrap_from_per_obs(base_per_obs, eval_a5["nested_cv_per_obs"])
    print_boot_line("1536_anchors5 (4v) Nested-CV", eval_a5["nested_cv"], boot_a5)
    del pool_a5
    gc.collect()

    print("[3/6] Loading 4-view 1536_dice OOF...")
    pool_dice = load_fused_oof_pool(folds_df, [("dice", "work/oof_dice", 4, 0.0)])
    eval_dice = evaluate_threshold_grid_nested_cv(pool_dice, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_dice = paired_bootstrap_from_per_obs(base_per_obs, eval_dice["nested_cv_per_obs"])
    print_boot_line(
        f"1536_dice (4v, best mk={eval_dice['global_best_tau_mask']:.3f}) Nested-CV",
        eval_dice["nested_cv"],
        boot_dice,
    )
    del pool_dice
    gc.collect()

    print("[4/6] Loading 8-view 2-Family OOF (clahe + anchors5)...")
    pool_8v = load_fused_oof_pool(folds_df, [("clahe", "work/oof", 4, 0.0), ("a5", "work/oof_a5", 4, 0.0)])
    eval_8v = evaluate_threshold_grid_nested_cv(pool_8v, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_8v = paired_bootstrap_from_per_obs(base_per_obs, eval_8v["nested_cv_per_obs"])
    print_boot_line(
        f"2-Family 8v (clahe+a5, Global {eval_8v['global_best']['pq']:.4f})",
        eval_8v["nested_cv"],
        boot_8v,
    )

    feat_8v, obs_pool_8v = extract_scorer_v2_dataset(
        pool_8v, folds_df, gt_df, paths["train_json"], num_views_total=8, tau_mask=0.50
    )
    scorer_8v_res, _ = run_cross_fitted_scorer(feat_8v, obs_pool_8v, folds_df, gt_by_obs, FEATURE_NAMES)
    del pool_8v, feat_8v, obs_pool_8v
    gc.collect()

    print("[5/6] Loading 12-view 3-Family OOF (clahe + anchors5 + dice, raw probs)...")
    specs_12v_raw = [("clahe", "work/oof", 4, 0.0), ("a5", "work/oof_a5", 4, 0.0), ("dice", "work/oof_dice", 4, 0.0)]
    pool_12v_raw = load_fused_oof_pool(folds_df, specs_12v_raw)
    eval_12v_raw = evaluate_threshold_grid_nested_cv(pool_12v_raw, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_12v_raw = paired_bootstrap_from_per_obs(base_per_obs, eval_12v_raw["nested_cv_per_obs"])
    print_boot_line(
        f"3-Family 12v Raw (Global({eval_12v_raw['global_best_tau_score']:.3f},{eval_12v_raw['global_best_tau_mask']:.3f})={eval_12v_raw['global_best']['pq']:.4f})",
        eval_12v_raw["nested_cv"],
        boot_12v_raw,
    )

    print("[6/6] Loading 12-view 3-Family OOF (clahe + anchors5 + dice, dice prob shift -0.05)...")
    specs_12v_cal = [("clahe", "work/oof", 4, 0.0), ("a5", "work/oof_a5", 4, 0.0), ("dice", "work/oof_dice", 4, -0.05)]
    pool_12v_cal = load_fused_oof_pool(folds_df, specs_12v_cal)
    eval_12v_cal = evaluate_threshold_grid_nested_cv(pool_12v_cal, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_12v_cal = paired_bootstrap_from_per_obs(base_per_obs, eval_12v_cal["nested_cv_per_obs"])
    print_boot_line(
        f"3-Family 12v Calibrated (Global({eval_12v_cal['global_best_tau_score']:.3f},{eval_12v_cal['global_best_tau_mask']:.3f})={eval_12v_cal['global_best']['pq']:.4f})",
        eval_12v_cal["nested_cv"],
        boot_12v_cal,
    )

    if eval_12v_cal["nested_cv"]["pq"] > eval_12v_raw["nested_cv"]["pq"]:
        best_12v_pool = pool_12v_cal
        best_12v_eval = eval_12v_cal
        best_12v_specs = specs_12v_cal
        best_dice_shift = -0.05
        del pool_12v_raw
    else:
        best_12v_pool = pool_12v_raw
        best_12v_eval = eval_12v_raw
        best_12v_specs = specs_12v_raw
        best_dice_shift = 0.0
        del pool_12v_cal
    gc.collect()

    best_12v_tau_mask = float(best_12v_eval["global_best_tau_mask"])
    best_12v_tau_score = float(best_12v_eval["global_best_tau_score"])
    best_12v_thresh_per_obs = best_12v_eval["nested_cv_per_obs"]

    # -------------------------------------------------------------------------
    # 2. Step 2b: WMF Clustering Test (707 OOF Images + Paired Bootstrap)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  STEP 2.2: WMF CLUSTERING TEST ON 12-VIEW MULTI-FAMILY OOF (707 IMAGES)")
    print("#" * 92)
    print(
        f"  Baseline Leader-IoU 0.40                  | PQ: {best_12v_eval['nested_cv']['pq']:.4f} "
        f"(SQ:{best_12v_eval['nested_cv']['sq']:.4f}, RQ:{best_12v_eval['nested_cv']['rq']:.4f})"
    )

    pool_iou035 = load_fused_oof_pool(folds_df, best_12v_specs, iou_thresh=0.35)
    eval_iou035 = evaluate_threshold_grid_nested_cv(pool_iou035, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_iou035 = paired_bootstrap_from_per_obs(best_12v_thresh_per_obs, eval_iou035["nested_cv_per_obs"])
    print_boot_line("Leader-IoU 0.35 vs Baseline 0.40", eval_iou035["nested_cv"], boot_iou035)
    del pool_iou035
    gc.collect()

    pool_iou045 = load_fused_oof_pool(folds_df, best_12v_specs, iou_thresh=0.45)
    eval_iou045 = evaluate_threshold_grid_nested_cv(pool_iou045, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_iou045 = paired_bootstrap_from_per_obs(best_12v_thresh_per_obs, eval_iou045["nested_cv_per_obs"])
    print_boot_line("Leader-IoU 0.45 vs Baseline 0.40", eval_iou045["nested_cv"], boot_iou045)
    del pool_iou045
    gc.collect()

    pool_runmean = load_fused_oof_pool(folds_df, best_12v_specs, iou_thresh=0.40, use_running_mean_iou=True)
    eval_runmean = evaluate_threshold_grid_nested_cv(pool_runmean, folds_df, gt_by_obs, score_grid, mask_grid)
    boot_runmean = paired_bootstrap_from_per_obs(best_12v_thresh_per_obs, eval_runmean["nested_cv_per_obs"])
    print_boot_line("Running-Mean Mask IoU 0.40 vs Baseline 0.40", eval_runmean["nested_cv"], boot_runmean)
    del pool_runmean
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Step 2c: Scorer v1 (4v, 8v, 12v) & Scorer v2 (Multi-Family + View-Invariant)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  STEP 2.3: SCORER V1 (4v, 8v, 12v) & SCORER V2 ON 707 OOF IMAGES")
    print("#" * 92)

    feat_12v_050, obs_pool_12v_050 = extract_scorer_v2_dataset(
        best_12v_pool, folds_df, gt_df, paths["train_json"], num_views_total=12, tau_mask=0.50
    )
    scorer_v1_12v_050, _ = run_cross_fitted_scorer(
        feat_12v_050, obs_pool_12v_050, folds_df, gt_by_obs, FEATURE_NAMES
    )
    scorer_v2_full_050, models_v2_full_050 = run_cross_fitted_scorer(
        feat_12v_050, obs_pool_12v_050, folds_df, gt_by_obs, SCORER_V2_FEATURES
    )
    scorer_v2_inv_050, models_v2_inv_050 = run_cross_fitted_scorer(
        feat_12v_050, obs_pool_12v_050, folds_df, gt_by_obs, SCORER_V2_VIEW_INVARIANT_FEATURES
    )

    feat_12v_0525, obs_pool_12v_0525 = extract_scorer_v2_dataset(
        best_12v_pool, folds_df, gt_df, paths["train_json"], num_views_total=12, tau_mask=0.525
    )
    scorer_v2_full_0525, models_v2_full_0525 = run_cross_fitted_scorer(
        feat_12v_0525, obs_pool_12v_0525, folds_df, gt_by_obs, SCORER_V2_FEATURES
    )
    scorer_v2_inv_0525, models_v2_inv_0525 = run_cross_fitted_scorer(
        feat_12v_0525, obs_pool_12v_0525, folds_df, gt_by_obs, SCORER_V2_VIEW_INVARIANT_FEATURES
    )

    boot_sc_4v = paired_bootstrap_from_per_obs(base_per_obs, scorer_v1_4v_per_obs)
    boot_sc_8v = paired_bootstrap_from_per_obs(base_per_obs, scorer_8v_res["per_obs"])
    boot_sc_12v_v1 = paired_bootstrap_from_per_obs(base_per_obs, scorer_v1_12v_050["per_obs"])
    boot_sc_12v_v2_full_050 = paired_bootstrap_from_per_obs(base_per_obs, scorer_v2_full_050["per_obs"])
    boot_sc_12v_v2_inv_050 = paired_bootstrap_from_per_obs(base_per_obs, scorer_v2_inv_050["per_obs"])
    boot_sc_12v_v2_full_0525 = paired_bootstrap_from_per_obs(base_per_obs, scorer_v2_full_0525["per_obs"])
    boot_sc_12v_v2_inv_0525 = paired_bootstrap_from_per_obs(base_per_obs, scorer_v2_inv_0525["per_obs"])

    print("  Comparison vs 1536_clahe Nested-CV Threshold Baseline (0.4343):")
    print_boot_line("Scorer v1 (4-view clahe, mk=0.50)", scorer_4v_res["pooled"], boot_sc_4v)
    print_boot_line("Scorer v1 (8-view clahe+a5, mk=0.50)", scorer_8v_res["pooled"], boot_sc_8v)
    print_boot_line("Scorer v1 (12-view 3-fam, mk=0.50)", scorer_v1_12v_050["pooled"], boot_sc_12v_v1)
    print_boot_line("Scorer v2 Full (12-view 3-fam, mk=0.50)", scorer_v2_full_050["pooled"], boot_sc_12v_v2_full_050)
    print_boot_line("Scorer v2 View-Invariant (12v, mk=0.50)", scorer_v2_inv_050["pooled"], boot_sc_12v_v2_inv_050)
    print_boot_line("Scorer v2 Full (12-view 3-fam, mk=0.525)", scorer_v2_full_0525["pooled"], boot_sc_12v_v2_full_0525)
    print_boot_line("Scorer v2 View-Invariant (12v, mk=0.525)", scorer_v2_inv_0525["pooled"], boot_sc_12v_v2_inv_0525)

    candidates_v2 = [
        ("v2_inv_0525", scorer_v2_inv_0525, models_v2_inv_0525, SCORER_V2_VIEW_INVARIANT_FEATURES, 0.525, obs_pool_12v_0525, feat_12v_0525),
        ("v2_full_0525", scorer_v2_full_0525, models_v2_full_0525, SCORER_V2_FEATURES, 0.525, obs_pool_12v_0525, feat_12v_0525),
        ("v2_inv_050", scorer_v2_inv_050, models_v2_inv_050, SCORER_V2_VIEW_INVARIANT_FEATURES, 0.500, obs_pool_12v_050, feat_12v_050),
        ("v2_full_050", scorer_v2_full_050, models_v2_full_050, SCORER_V2_FEATURES, 0.500, obs_pool_12v_050, feat_12v_050),
    ]
    candidates_v2.sort(key=lambda x: x[1]["pooled"]["pq"], reverse=True)
    best_v2_name, best_v2_res, best_v2_models, best_v2_cols, best_v2_mk, best_v2_pool, best_v2_df = candidates_v2[0]

    oof_p = np.zeros(len(best_v2_df), dtype=np.float32)
    for val_f in range(5):
        va_m = (best_v2_df["fold"] == val_f).values
        oof_p[va_m] = np.clip(best_v2_models[val_f].predict(best_v2_df.loc[va_m, best_v2_cols].values), 0.0, 1.0)
    for r_idx, row in enumerate(best_v2_df[["obs", "cand_idx"]].itertuples(index=False)):
        best_v2_pool[row.obs][int(row.cand_idx)]["pred_y"] = float(oof_p[r_idx])

    boot_v2_vs_12v_thresh = paired_bootstrap_from_per_obs(best_12v_thresh_per_obs, best_v2_res["per_obs"])
    boot_v2_vs_sc_v1 = paired_bootstrap_from_per_obs(scorer_v1_4v_per_obs, best_v2_res["per_obs"])
    print("\n  Direct Paired Bootstrap Comparisons for Best Scorer v2 (" + best_v2_name + "):")
    print_boot_line("Best Scorer v2 vs 12v Nested-CV Threshold", best_v2_res["pooled"], boot_v2_vs_12v_thresh)
    print_boot_line("Best Scorer v2 vs 4v Scorer v1 (0.4387)", best_v2_res["pooled"], boot_v2_vs_sc_v1)

    print("\n  Top 12 Features in Best Scorer v2 (by Gain %):")
    for item in best_v2_res["feature_importances"][:12]:
        print(f"    - {item['feature']:<24}: {item['gain_pct']:5.1f}%")

    # -------------------------------------------------------------------------
    # 4. Step 2d: Morphological Hole Fill Test (707 OOF Images + Paired Bootstrap)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  STEP 2.4: MORPHOLOGICAL HOLE FILL TEST ON 707 OOF IMAGES")
    print("#" * 92)

    hf_all_per_obs: Dict[str, dict] = {}
    hf_small_per_obs: Dict[str, dict] = {}
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}

    for val_f in range(5):
        tau_f = best_v2_res["folds"][val_f]["chosen_tau"]
        per_all = evaluate_Pool_at_tau(best_v2_pool, obs_by_fold[val_f], gt_by_obs, tau=tau_f, hole_fill_mode="all")
        per_sm = evaluate_Pool_at_tau(best_v2_pool, obs_by_fold[val_f], gt_by_obs, tau=tau_f, hole_fill_mode="small100")
        hf_all_per_obs.update(per_all)
        hf_small_per_obs.update(per_sm)

    boot_hf_all = paired_bootstrap_from_per_obs(best_v2_res["per_obs"], hf_all_per_obs)
    boot_hf_small = paired_bootstrap_from_per_obs(best_v2_res["per_obs"], hf_small_per_obs)
    print_boot_line("Scorer v2 + Fill All Holes vs No Hole Fill", boot_hf_all["full_b"], boot_hf_all)
    print_boot_line("Scorer v2 + Fill Small Holes (<=100px)", boot_hf_small["full_b"], boot_hf_small)

    use_hole_fill = None
    if boot_hf_small["point_delta"] > 0.0002 and boot_hf_small["p_b_better"] >= 0.80:
        use_hole_fill = "small100"
    elif boot_hf_all["point_delta"] > 0.0002 and boot_hf_all["p_b_better"] >= 0.80:
        use_hole_fill = "all"

    # -------------------------------------------------------------------------
    # 5. Step 2.5: Generate Final Submissions A (Scorer v2) and B (Threshold Rule)
    # -------------------------------------------------------------------------
    sub_models = models_v2_inv_0525 if scorer_v2_inv_0525["pooled"]["pq"] >= best_v2_res["pooled"]["pq"] - 0.0005 else best_v2_models
    sub_cols = SCORER_V2_VIEW_INVARIANT_FEATURES if sub_models is models_v2_inv_0525 else best_v2_cols
    sub_tau = scorer_v2_inv_0525["mean_chosen_tau"] if sub_models is models_v2_inv_0525 else best_v2_res["mean_chosen_tau"]
    sub_mk = 0.525 if sub_models is models_v2_inv_0525 else best_v2_mk

    generate_final_test_submissions_a_and_b(
        scorer_models=sub_models,
        scorer_feature_cols=sub_cols,
        scorer_tau=sub_tau,
        scorer_tau_mask=sub_mk,
        thresh_tau_score=best_12v_tau_score,
        thresh_tau_mask=best_12v_tau_mask,
        dice_prob_shift=best_dice_shift,
        hole_fill_mode=use_hole_fill,
    )

    summary_out = {
        "step_2_1_multifamily_oof": {
            "clahe_4v_nested_cv": eval_clahe["nested_cv"],
            "anchors5_4v_nested_cv": eval_a5["nested_cv"],
            "dice_4v_nested_cv": eval_dice["nested_cv"],
            "2fam_8v_nested_cv": eval_8v["nested_cv"],
            "2fam_8v_global_best": eval_8v["global_best"],
            "3fam_12v_raw_nested_cv": eval_12v_raw["nested_cv"],
            "3fam_12v_raw_global_best": eval_12v_raw["global_best"],
            "3fam_12v_raw_best_tau": [eval_12v_raw["global_best_tau_score"], eval_12v_raw["global_best_tau_mask"]],
            "3fam_12v_cal_nested_cv": eval_12v_cal["nested_cv"],
            "3fam_12v_cal_global_best": eval_12v_cal["global_best"],
            "boot_8v_vs_clahe": boot_8v,
            "boot_12v_raw_vs_clahe": boot_12v_raw,
            "boot_12v_cal_vs_clahe": boot_12v_cal,
        },
        "step_2_2_clustering": {
            "iou_035_nested_cv": eval_iou035["nested_cv"],
            "boot_iou_035_vs_040": boot_iou035,
            "iou_045_nested_cv": eval_iou045["nested_cv"],
            "boot_iou_045_vs_040": boot_iou045,
            "running_mean_iou_040_nested_cv": eval_runmean["nested_cv"],
            "boot_running_mean_vs_040": boot_runmean,
        },
        "step_2_3_scorer": {
            "scorer_v1_4v": scorer_4v_res["pooled"],
            "scorer_v1_8v": scorer_8v_res["pooled"],
            "scorer_v1_12v": scorer_v1_12v_050["pooled"],
            "scorer_v2_full_050": scorer_v2_full_050["pooled"],
            "scorer_v2_inv_050": scorer_v2_inv_050["pooled"],
            "scorer_v2_full_0525": scorer_v2_full_0525["pooled"],
            "scorer_v2_inv_0525": scorer_v2_inv_0525["pooled"],
            "best_v2_name": best_v2_name,
            "best_v2_folds": best_v2_res["folds"],
            "best_v2_feature_importances": best_v2_res["feature_importances"],
            "boot_best_v2_vs_clahe_thresh": paired_bootstrap_from_per_obs(base_per_obs, best_v2_res["per_obs"]),
            "boot_best_v2_vs_12v_thresh": boot_v2_vs_12v_thresh,
            "boot_best_v2_vs_scorer_v1": boot_v2_vs_sc_v1,
        },
        "step_2_4_hole_fill": {
            "fill_all_holes": boot_hf_all["full_b"],
            "boot_fill_all": boot_hf_all,
            "fill_small100_holes": boot_hf_small["full_b"],
            "boot_fill_small100": boot_hf_small,
        },
    }
    out_json = "work/oof/multifamily_pipeline_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved complete Step 2 summary to {out_json} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
