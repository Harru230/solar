#!/usr/bin/env python3
"""
audit/run_tasks_1_2_3.py - Ultra-low-memory (<500 MB RAM) streaming CPU evaluation for Tasks 1, 2, 3:
  1) Exact +/- 1 px erosion and dilation on 5-family OOF (under both annToMask and cv2.fillPoly GT,
     for both Threshold Rule and Scorer v2).
  2) Candidate M1: WMF family weights (w_highres/w_1536 in {1, 1.25, 1.5, 2}, gamma in {1, 1.5, 2})
     plus Scorer v2 retrained with score_2048, score_1792, score_a5dice, score_highres - score_1536.
     Strictly nested 5-fold cross-fitted evaluation over all 707 observations (at most 4 free params),
     with 10,000-iteration paired bootstrap by image.
  3) Side-by-side OOF PQ under annToMask (gt_all.csv / pq_eval.py) vs cv2.fillPoly (SolarDataset training target).
"""
import argparse
import os
import sys
import gc
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import torch
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLAR_ROOT = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))).resolve()
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import find_dataset_paths
from scripts.instance_scorer import (
    encode_crop_to_rle,
    build_gt_lookup,
    compute_candidate_targets,
)
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    build_gt_rles_by_obs,
    compute_binary_crop_iou,
    drop_os_page_cache,
    fast_crop_mask_iou,
    paired_bootstrap_from_per_obs,
)
from audit.run_audit_metrics import rasterize_coco_fillpoly

FAMILY_SPECS_5 = [
    ("clahe", str(SOLAR_ROOT / "work/oof"), 4),
    ("a5", str(SOLAR_ROOT / "work/oof_a5"), 4),
    ("dice", str(SOLAR_ROOT / "work/oof_dice"), 4),
    ("a5dice", str(SOLAR_ROOT / "work/oof_a5dice"), 4),
    ("2048", str(SOLAR_ROOT / "work/oof_2048"), 4),
]
OOF_1792_DIR = str(SOLAR_ROOT / "work/oof_1792")

HIGHRES_FAMS = {"2048", "1792"}
R_GRID = [1.0, 1.25, 1.5, 2.0]
GAMMA_GRID = [1.0, 1.5, 2.0]
TAU_MASK_GRID = [0.50, 0.55]
SCORE_GRID = [round(x, 3) for x in np.arange(0.65, 0.926, 0.025)]
TAU_Y_GRID = [round(0.180 + 0.005 * k, 4) for k in range(25)]  # 0.180 .. 0.300


def check_available_ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return 16.0


def cluster_raw_candidates(candidate_list: List[dict], iou_thresh: float = 0.40) -> List[List[dict]]:
    if not candidate_list:
        return []
    sorted_candidates = sorted(candidate_list, key=lambda c: c["score"], reverse=True)
    clusters: List[List[dict]] = []
    for cand in sorted_candidates:
        best_cluster_idx = -1
        best_iou = -1.0
        for c_idx, cluster in enumerate(clusters):
            iou = fast_crop_mask_iou(cand, cluster[0])
            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = c_idx
        if best_iou > iou_thresh:
            clusters[best_cluster_idx].append(cand)
        else:
            clusters.append([cand])
    return clusters


def agg_pq(tots_4: List[float]) -> dict:
    iou_s, tp, fp, fn = tots_4
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = float(iou_s / denom) if denom > 0 else 0.0
    sq = float(iou_s / tp) if tp > 0 else 0.0
    rq = float(tp / denom) if denom > 0 else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_sum": float(iou_s)}


def score_iou_mats_at_k(iou_mats: List[np.ndarray], k_keep: int) -> List[float]:
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
    return [iou_s, tp, fp, fn]


def paint_and_morph_single_obs(
    clusters: List[dict],
    tau_mask: float,
    min_score: float,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Paints candidates in descending score order for one observation and produces RLE lists for:
      - 'none': standard disjoint canvas
      - 'erode': 1px 3x3 eroded after canvas painting
      - 'dilate_raw': 1px 3x3 dilated after canvas painting (matching run_audit_metrics.py)
      - 'dilate_disjoint': 1px 3x3 dilated clipped against occupied canvas
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    occupied_dil = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

    rles_none, sc_none = [], []
    rles_erode, sc_erode = [], []
    rles_dil_raw, sc_dil_raw = [], []
    rles_dil_disj, sc_dil_disj = [], []

    for c in clusters:
        sc = float(c["score"])
        if sc < min_score:
            break
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bc = (c["prob_crop"] >= tau_mask)
        oa = int(bc.sum())
        if oa < 10:
            continue
        sub_occ = occupied[y1:y2, x1:x2]
        clean_crop = bc & (~sub_occ)
        ca = int(clean_crop.sum())
        if ca < 10 or (ca / float(oa)) < 0.50:
            continue
        occupied[y1:y2, x1:x2] |= clean_crop

        # 1. None
        rles_none.append(encode_crop_to_rle(box, clean_crop))
        sc_none.append(sc)

        # 2. Erode 1px
        eroded = cv2.erode(clean_crop.astype(np.uint8), kernel)
        if eroded.sum() >= 5:
            rles_erode.append(encode_crop_to_rle(box, eroded.astype(bool)))
            sc_erode.append(sc)

        # 3. Dilate 1px raw & disjoint
        x1_p, y1_p = max(0, x1 - 1), max(0, y1 - 1)
        x2_p, y2_p = min(FRAME_SIZE, x2 + 1), min(FRAME_SIZE, y2 + 1)
        local_box = np.array([x1_p, y1_p, x2_p, y2_p], dtype=np.int32)
        local_canvas = np.zeros((y2_p - y1_p, x2_p - x1_p), dtype=np.uint8)
        local_canvas[(y1 - y1_p):(y2 - y1_p), (x1 - x1_p):(x2 - x1_p)] = clean_crop.astype(np.uint8)
        dilated = cv2.dilate(local_canvas, kernel).astype(bool)
        rles_dil_raw.append(encode_crop_to_rle(local_box, dilated))
        sc_dil_raw.append(sc)

        dil_disj = dilated & (~occupied_dil[y1_p:y2_p, x1_p:x2_p])
        if dil_disj.sum() >= 10:
            occupied_dil[y1_p:y2_p, x1_p:x2_p] |= dil_disj
            rles_dil_disj.append(encode_crop_to_rle(local_box, dil_disj))
            sc_dil_disj.append(sc)

    return {
        "none": (np.asarray(sc_none, dtype=np.float64), rles_none),
        "erode": (np.asarray(sc_erode, dtype=np.float64), rles_erode),
        "dilate_raw": (np.asarray(sc_dil_raw, dtype=np.float64), rles_dil_raw),
        "dilate_disjoint": (np.asarray(sc_dil_disj, dtype=np.float64), rles_dil_disj),
    }


def extract_features_for_obs_rg(
    obs: str,
    fold: int,
    clusters: List[dict],
    k_by_obs: Dict[str, int],
    gt_records_by_obs: Dict[str, Dict[str, List[str]]],
    tau_mask: float = 0.525,
    min_cluster_score: float = 0.18,
) -> List[dict]:
    """
    Extracts Scorer v2 feature dicts + compact RLE string for one observation and one (r, gamma).
    Never retains 2D binary_crop or prob_crop arrays in returned rows!
    """
    if not clusters:
        return []
    all_scores = [float(c["score"]) for c in clusters]
    max_score_in_img = max(all_scores) if all_scores else 1e-6
    n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

    rows = []
    cand_rles_for_target = []
    higher_rank_boxes_and_crops = []
    sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

    for rank_0, c in enumerate(clusters):
        sc = float(c["score"])
        if sc < min_cluster_score:
            continue
        prob_crop = c["prob_crop"]
        box = c["box"]
        binary_crop = (prob_crop >= tau_mask)
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
        norm_views = float(c["cluster_size"]) * (4.0 / 20.0)

        log_area = float(np.log10(area + 1.0))
        bbox_fill = float(area) / bbox_area
        aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        rank_in_image = rank_0 + 1
        rel_score_to_max = sc / max(1e-6, max_score_in_img)

        fam_sc = c.get("family_scores", {})
        sc_clahe = float(fam_sc.get("clahe", 0.0))
        sc_a5 = float(fam_sc.get("a5", 0.0))
        sc_dice = float(fam_sc.get("dice", 0.0))
        sc_a5dice = float(fam_sc.get("a5dice", 0.0))
        sc_2048 = float(fam_sc.get("2048", 0.0))
        sc_1792 = float(fam_sc.get("1792", 0.0))

        mean_1536 = 0.25 * (sc_clahe + sc_a5 + sc_dice + sc_a5dice)
        highres_sc = sc_2048 if sc_1792 == 0.0 else 0.5 * (sc_2048 + sc_1792)
        sc_highres_minus_1536 = float(highres_sc - mean_1536)

        fam_arr_3 = np.array([sc_clahe, sc_a5, sc_dice], dtype=np.float32)

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
            "fold": fold,
            "cand_idx": len(rows),
            "rle": rle_str,
            "orig_area": area,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
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
            "score_a5dice": sc_a5dice,
            "score_2048": sc_2048,
            "score_1792": sc_1792,
            "score_highres_minus_1536": sc_highres_minus_1536,
            "family_score_std": float(fam_arr_3.std()),
            "n_families_active": float(sum(1 for x in fam_arr_3 if x > 0.0)),
            "unoccluded_area_ratio": unoccluded_area_ratio,
            "max_iou_higher_rank": max_iou_higher,
        })

    if rows:
        targets = compute_candidate_targets(cand_rles_for_target, obs, k_by_obs, gt_records_by_obs)
        for idx_in_obs, y_val in enumerate(targets):
            rows[idx_in_obs]["target_y"] = float(y_val)
    return rows


def main():
    argparse.ArgumentParser(
        description="Evaluate morphological boundary sensitivity, candidate WMF weights, and rasterization PQ."
    ).parse_args()
    t_start = time.time()
    avail_gb = check_available_ram_gb()
    print(f"Initial available RAM: {avail_gb:.2f} GB", flush=True)
    if avail_gb < 6.0:
        print("Aborting: less than 6 GB RAM available.")
        sys.exit(1)

    paths = find_dataset_paths(str(SOLAR_ROOT))
    folds_df = pd.read_csv(paths["folds_csv"])
    gt_df = pd.read_csv(paths["gt_all_csv"])
    train_json = paths["train_json"]
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}

    print("1. Building annToMask and cv2.fillPoly GT lookups...", flush=True)
    gt_ann = build_gt_rles_by_obs(gt_df)
    gt_cv2 = rasterize_coco_fillpoly(train_json)
    k_by_obs, gt_records_by_obs = build_gt_lookup(gt_df, train_json)

    # Accumulators across streaming pass:
    # thresh_obs_ann[(r, g, sc, mk)][obs] = [iou_s, tp, fp, fn]
    thresh_obs_ann = defaultdict(dict)
    thresh_obs_cv2 = defaultdict(dict)
    thresh_fold_ann = defaultdict(lambda: {f: [0.0, 0, 0, 0] for f in range(5)})
    thresh_fold_cv2 = defaultdict(lambda: {f: [0.0, 0, 0, 0] for f in range(5)})

    # Task 1 accumulators at (r=1, g=1, mk=0.55, sc=0.85 and sc grid)
    task1_thresh_ann = {m: [0.0, 0, 0, 0] for m in ("none", "erode", "dilate_raw", "dilate_disjoint")}
    task1_thresh_cv2 = {m: [0.0, 0, 0, 0] for m in ("none", "erode", "dilate_raw", "dilate_disjoint")}

    # Compact feature rows for each (r, gamma): only scalars + 40-byte RLE string (~2 MB per (r, gamma))
    scorer_rows_by_rg = defaultdict(list)

    family_views = {fam: v for fam, _, v in FAMILY_SPECS_5}
    min_sc_grid = min(SCORE_GRID)

    print("2. Streaming through 707 observations (one image in RAM at a time)...", flush=True)
    t_stream = time.time()

    for f in range(5):
        for idx_obs, obs in enumerate(obs_by_fold[f]):
            if idx_obs % 50 == 0 and check_available_ram_gb() < 5.0:
                print("Low RAM detected (<5 GB), stopping immediately!")
                sys.exit(1)

            combined_cands = []
            for fam_name, oof_dir, _ in FAMILY_SPECS_5:
                pt_path = os.path.join(oof_dir, f"fold{f}", f"{obs}.pt")
                if not os.path.isfile(pt_path):
                    continue
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                for c in data.get("candidates", []):
                    combined_cands.append({
                        "box": c["box"],
                        "score": float(c["score"]),
                        "prob_crop": c["prob_crop"],
                        "family": fam_name,
                    })
                del data
                drop_os_page_cache(pt_path)

            # Check if 1792 fold file exists for this image (for score_1792 feature)
            cands_1792 = []
            pt_1792 = os.path.join(OOF_1792_DIR, f"fold{f}", f"{obs}.pt")
            if os.path.isfile(pt_1792):
                d1792 = torch.load(pt_1792, map_location="cpu", weights_only=False)
                cands_1792 = d1792.get("candidates", [])
                del d1792
                drop_os_page_cache(pt_1792)

            raw_clusters = cluster_raw_candidates(combined_cands, iou_thresh=0.40)

            # Precompute per-cluster low/high crops for this single image
            cl_items = []
            for cl in raw_clusters:
                scores = [float(c["score"]) for c in cl]
                if (sum(scores) / 20.0) < 0.15:
                    continue
                fam_sums = defaultdict(float)
                for c in cl:
                    fam_sums[c["family"]] += float(c["score"])
                fam_scores = {fam: fam_sums[fam] / float(fv) for fam, fv in family_views.items()}

                # Match against cands_1792 if available, else 0.0
                sc_1792 = 0.0
                if cands_1792:
                    s1792 = 0.0
                    for c17 in cands_1792:
                        if fast_crop_mask_iou(c17, cl[0]) > 0.40:
                            s1792 += float(c17["score"])
                    sc_1792 = s1792 / 4.0
                fam_scores["1792"] = sc_1792

                all_boxes = np.stack([c["box"] for c in cl], axis=0)
                ux1, uy1 = int(all_boxes[:, 0].min()), int(all_boxes[:, 1].min())
                ux2, uy2 = int(all_boxes[:, 2].max()), int(all_boxes[:, 3].max())
                union_box = np.array([ux1, uy1, ux2, uy2], dtype=np.int32)
                uh, uw = uy2 - uy1, ux2 - ux1

                g_map = {}
                for gamma in GAMMA_GRID:
                    w_low, w_high = 0.0, 0.0
                    crop_low = np.zeros((uh, uw), dtype=np.float32)
                    crop_high = np.zeros((uh, uw), dtype=np.float32)
                    for c in cl:
                        s = float(c["score"])
                        wg = s ** gamma
                        b = c["box"]
                        sl_y = slice(b[1] - uy1, b[3] - uy1)
                        sl_x = slice(b[0] - ux1, b[2] - ux1)
                        if c["family"] in HIGHRES_FAMS:
                            w_high += wg
                            crop_high[sl_y, sl_x] += wg * c["prob_crop"].astype(np.float32)
                        else:
                            w_low += wg
                            crop_low[sl_y, sl_x] += wg * c["prob_crop"].astype(np.float32)
                    g_map[gamma] = (w_low, w_high, crop_low, crop_high)

                cl_items.append({
                    "box": union_box,
                    "cluster_size": len(cl),
                    "member_scores": scores,
                    "family_scores": fam_scores,
                    "g_map": g_map,
                })

            annot_gts_ann = gt_ann.get(obs, [])
            annot_gts_cv2 = gt_cv2.get(obs, [])

            for r_val in R_GRID:
                denom_v = 16.0 + 4.0 * r_val
                for g_val in GAMMA_GRID:
                    obs_clusters = []
                    for item in cl_items:
                        w_low, w_high, crop_low, crop_high = item["g_map"][g_val]
                        tot_w = w_low + r_val * w_high
                        if tot_w <= 1e-9:
                            continue
                        sc = float(tot_w / denom_v)
                        fused_crop = (crop_low + r_val * crop_high) / float(tot_w)
                        obs_clusters.append({
                            "score": sc,
                            "prob_crop": fused_crop,
                            "box": item["box"],
                            "cluster_size": item["cluster_size"],
                            "member_scores": item["member_scores"],
                            "family_scores": item["family_scores"],
                        })
                    obs_clusters.sort(key=lambda x: x["score"], reverse=True)

                    # Evaluate Threshold Rule on TAU_MASK_GRID x SCORE_GRID
                    for mk in TAU_MASK_GRID:
                        if r_val == 1.0 and g_val == 1.0 and mk == 0.55:
                            morph_res = paint_and_morph_single_obs(obs_clusters, mk, min_sc_grid)
                            sc_arr, painted_rles = morph_res["none"]
                            for m_name, (m_sc_arr, m_rles) in morph_res.items():
                                k_85 = int(np.sum(m_sc_arr >= 0.85))
                                mats_a = [pq_eval.iou_matrix(gt_r, m_rles) for gt_r in annot_gts_ann]
                                mats_c = [pq_eval.iou_matrix(gt_r, m_rles) for gt_r in annot_gts_cv2]
                                t_a = score_iou_mats_at_k(mats_a, k_85)
                                t_c = score_iou_mats_at_k(mats_c, k_85)
                                for i in range(4):
                                    task1_thresh_ann[m_name][i] += t_a[i]
                                    task1_thresh_cv2[m_name][i] += t_c[i]
                        else:
                            occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                            painted_rles = []
                            painted_scores = []
                            for c in obs_clusters:
                                sc = float(c["score"])
                                if sc < min_sc_grid:
                                    break
                                box = c["box"]
                                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                                bc = (c["prob_crop"] >= mk)
                                oa = int(bc.sum())
                                if oa < 10:
                                    continue
                                sub_occ = occupied[y1:y2, x1:x2]
                                clean_crop = bc & (~sub_occ)
                                ca = int(clean_crop.sum())
                                if ca < 10 or (ca / float(oa)) < 0.50:
                                    continue
                                occupied[y1:y2, x1:x2] |= clean_crop
                                painted_rles.append(encode_crop_to_rle(box, clean_crop))
                                painted_scores.append(sc)
                            sc_arr = np.asarray(painted_scores, dtype=np.float64)

                        iou_mats_ann = [pq_eval.iou_matrix(gt_r, painted_rles) for gt_r in annot_gts_ann]
                        iou_mats_cv2 = [pq_eval.iou_matrix(gt_r, painted_rles) for gt_r in annot_gts_cv2]

                        for sc in SCORE_GRID:
                            k_keep = int(np.sum(sc_arr >= sc))
                            t_a = score_iou_mats_at_k(iou_mats_ann, k_keep)
                            t_c = score_iou_mats_at_k(iou_mats_cv2, k_keep)
                            key = (r_val, g_val, sc, mk)
                            thresh_obs_ann[key][obs] = t_a
                            thresh_obs_cv2[key][obs] = t_c
                            for i in range(4):
                                thresh_fold_ann[key][f][i] += t_a[i]
                                thresh_fold_cv2[key][f][i] += t_c[i]

                    # Extract compact Scorer v2 feature rows (scalars + RLE string only)
                    f_rows = extract_features_for_obs_rg(
                        obs, f, obs_clusters, k_by_obs, gt_records_by_obs, tau_mask=0.525, min_cluster_score=0.18
                    )
                    scorer_rows_by_rg[(r_val, g_val)].extend(f_rows)
                    del obs_clusters

            del combined_cands, raw_clusters, cl_items

        gc.collect()
        print(
            f"  Fold {f} streamed ({time.time() - t_stream:.1f}s elapsed, free RAM: {check_available_ram_gb():.2f} GB)",
            flush=True,
        )

    # 3. Fit and evaluate Cross-Fitted Scorer v2 across all 12 (r, gamma) pairs
    print("3. Fitting and evaluating cross-fitted LightGBM Scorer v2 models...", flush=True)
    features_m0 = list(SCORER_V2_VIEW_INVARIANT_FEATURES)
    features_m1 = list(SCORER_V2_VIEW_INVARIANT_FEATURES) + [
        "score_2048",
        "score_1792",
        "score_a5dice",
        "score_highres_minus_1536",
    ]

    scorer_m1_fold_ann = {}
    scorer_m1_obs_ann = {}
    scorer_m1_fold_cv2 = {}
    scorer_m1_obs_cv2 = {}

    m0_sc_fold_ann = {}
    m0_sc_obs_ann = {}
    m0_sc_fold_cv2 = {}
    m0_sc_obs_cv2 = {}

    task1_scorer_ann = {m: [0.0, 0, 0, 0] for m in ("none", "erode", "dilate_raw", "dilate_disjoint")}
    task1_scorer_cv2 = {m: [0.0, 0, 0, 0] for m in ("none", "erode", "dilate_raw", "dilate_disjoint")}

    def run_lgb_and_eval(df_rg: pd.DataFrame, feat_cols: List[str], run_task1_morph: bool = False):
        oof_preds = np.zeros(len(df_rg), dtype=np.float32)
        for val_f in range(5):
            tr_mask = (df_rg["fold"] != val_f).values
            va_mask = (df_rg["fold"] == val_f).values
            X_tr = df_rg.loc[tr_mask, feat_cols].values
            y_tr = df_rg.loc[tr_mask, "target_y"].values
            X_va = df_rg.loc[va_mask, feat_cols].values

            m = lgb.LGBMRegressor(
                objective="regression",
                max_depth=3,
                num_leaves=7,
                min_child_samples=50,
                n_estimators=300,
                learning_rate=0.03,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=2026 + val_f,
                verbose=-1,
                n_jobs=4,
            )
            m.fit(X_tr, y_tr)
            oof_preds[va_mask] = np.clip(m.predict(X_va), 0.0, 1.0).astype(np.float32)

        df_rg = df_rg.copy()
        df_rg["pred_y"] = oof_preds
        grouped = {
            obs: grp.sort_values("pred_y", ascending=False)
            for obs, grp in df_rg.groupby("obs", sort=False)
        }

        f_tots_ann = {ty: {f: [0.0, 0, 0, 0] for f in range(5)} for ty in TAU_Y_GRID}
        p_obs_ann = {ty: {} for ty in TAU_Y_GRID}
        f_tots_cv2 = {ty: {f: [0.0, 0, 0, 0] for f in range(5)} for ty in TAU_Y_GRID}
        p_obs_cv2 = {ty: {} for ty in TAU_Y_GRID}

        kernel = np.ones((3, 3), dtype=np.uint8)
        min_ty = min(TAU_Y_GRID)

        for f in range(5):
            for obs in obs_by_fold[f]:
                grp = grouped.get(obs)
                occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                painted_rles = []
                painted_scores = []

                if run_task1_morph:
                    occupied_dil = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                    r_erode, s_erode = [], []
                    r_dil_raw, s_dil_raw = [], []
                    r_dil_disj, s_dil_disj = [], []

                if grp is not None:
                    for row in grp.itertuples(index=False):
                        py = float(row.pred_y)
                        if py < min_ty:
                            break
                        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                        full_m = mu.decode({"size": [FRAME_SIZE, FRAME_SIZE], "counts": row.rle.encode("ascii")})
                        bc = full_m[y1:y2, x1:x2].astype(bool)
                        oa = int(row.orig_area)
                        sub_occ = occupied[y1:y2, x1:x2]
                        clean_crop = bc & (~sub_occ)
                        ca = int(clean_crop.sum())
                        if ca < 10 or (ca / float(oa)) < 0.50:
                            continue
                        occupied[y1:y2, x1:x2] |= clean_crop
                        box = np.array([x1, y1, x2, y2], dtype=np.int32)
                        painted_rles.append(encode_crop_to_rle(box, clean_crop))
                        painted_scores.append(py)

                        if run_task1_morph:
                            eroded = cv2.erode(clean_crop.astype(np.uint8), kernel)
                            if eroded.sum() >= 5:
                                r_erode.append(encode_crop_to_rle(box, eroded.astype(bool)))
                                s_erode.append(py)
                            x1_p, y1_p = max(0, x1 - 1), max(0, y1 - 1)
                            x2_p, y2_p = min(FRAME_SIZE, x2 + 1), min(FRAME_SIZE, y2 + 1)
                            lbox = np.array([x1_p, y1_p, x2_p, y2_p], dtype=np.int32)
                            lcanv = np.zeros((y2_p - y1_p, x2_p - x1_p), dtype=np.uint8)
                            lcanv[(y1 - y1_p):(y2 - y1_p), (x1 - x1_p):(x2 - x1_p)] = clean_crop.astype(np.uint8)
                            dil = cv2.dilate(lcanv, kernel).astype(bool)
                            r_dil_raw.append(encode_crop_to_rle(lbox, dil))
                            s_dil_raw.append(py)
                            dil_d = dil & (~occupied_dil[y1_p:y2_p, x1_p:x2_p])
                            if dil_d.sum() >= 10:
                                occupied_dil[y1_p:y2_p, x1_p:x2_p] |= dil_d
                                r_dil_disj.append(encode_crop_to_rle(lbox, dil_d))
                                s_dil_disj.append(py)

                sc_arr = np.asarray(painted_scores, dtype=np.float64)
                mats_ann = [pq_eval.iou_matrix(gt_r, painted_rles) for gt_r in gt_ann.get(obs, [])]
                mats_cv2 = [pq_eval.iou_matrix(gt_r, painted_rles) for gt_r in gt_cv2.get(obs, [])]

                if run_task1_morph:
                    for m_name, (m_s, m_r) in [
                        ("none", (sc_arr, painted_rles)),
                        ("erode", (np.asarray(s_erode, dtype=np.float64), r_erode)),
                        ("dilate_raw", (np.asarray(s_dil_raw, dtype=np.float64), r_dil_raw)),
                        ("dilate_disjoint", (np.asarray(s_dil_disj, dtype=np.float64), r_dil_disj)),
                    ]:
                        k_25 = int(np.sum(m_s > 0.250))
                        ta = score_iou_mats_at_k([pq_eval.iou_matrix(gt_r, m_r) for gt_r in gt_ann.get(obs, [])], k_25)
                        tc = score_iou_mats_at_k([pq_eval.iou_matrix(gt_r, m_r) for gt_r in gt_cv2.get(obs, [])], k_25)
                        for i in range(4):
                            task1_scorer_ann[m_name][i] += ta[i]
                            task1_scorer_cv2[m_name][i] += tc[i]

                for ty in TAU_Y_GRID:
                    k_keep = int(np.sum(sc_arr > ty))
                    ta = score_iou_mats_at_k(mats_ann, k_keep)
                    tc = score_iou_mats_at_k(mats_cv2, k_keep)
                    p_obs_ann[ty][obs] = ta
                    p_obs_cv2[ty][obs] = tc
                    for i in range(4):
                        f_tots_ann[ty][f][i] += ta[i]
                        f_tots_cv2[ty][f][i] += tc[i]

        return f_tots_ann, p_obs_ann, f_tots_cv2, p_obs_cv2

    for (r_val, g_val), rows_list in scorer_rows_by_rg.items():
        df_rg = pd.DataFrame(rows_list)
        if r_val == 1.0 and g_val == 1.0:
            fa, pa, fc, pc = run_lgb_and_eval(df_rg, features_m0, run_task1_morph=True)
            for ty in TAU_Y_GRID:
                m0_sc_fold_ann[(1.0, 1.0, ty)] = fa[ty]
                m0_sc_obs_ann[(1.0, 1.0, ty)] = pa[ty]
                m0_sc_fold_cv2[(1.0, 1.0, ty)] = fc[ty]
                m0_sc_obs_cv2[(1.0, 1.0, ty)] = pc[ty]

        fa1, pa1, fc1, pc1 = run_lgb_and_eval(df_rg, features_m1, run_task1_morph=False)
        for ty in TAU_Y_GRID:
            scorer_m1_fold_ann[(r_val, g_val, ty)] = fa1[ty]
            scorer_m1_obs_ann[(r_val, g_val, ty)] = pa1[ty]
            scorer_m1_fold_cv2[(r_val, g_val, ty)] = fc1[ty]
            scorer_m1_obs_cv2[(r_val, g_val, ty)] = pc1[ty]
        print(f"  Scored LightGBM for (r={r_val}, gamma={g_val})", flush=True)

    # Helper for 5-fold Nested CV + Global Best
    def evaluate_nested_and_global(grid_dict, obs_dict, allowed_keys):
        best_glob_key = None
        best_glob_pq = -1.0
        for k in allowed_keys:
            tots = [sum(grid_dict[k][f][i] for f in range(5)) for i in range(4)]
            res = agg_pq(tots)
            if res["pq"] > best_glob_pq:
                best_glob_pq = res["pq"]
                best_glob_key = k

        glob_tots = [sum(grid_dict[best_glob_key][f][i] for f in range(5)) for i in range(4)]
        glob_res = agg_pq(glob_tots)

        lofo_tots = [0.0, 0, 0, 0]
        lofo_choices = {}
        lofo_per_obs = {}
        for val_f in range(5):
            best_tr_key = None
            best_tr_pq = -1.0
            for k in allowed_keys:
                tr_tots = [sum(grid_dict[k][f][i] for f in range(5) if f != val_f) for i in range(4)]
                tr_res = agg_pq(tr_tots)
                if tr_res["pq"] > best_tr_pq:
                    best_tr_pq = tr_res["pq"]
                    best_tr_key = k
            val_f_tots = grid_dict[best_tr_key][val_f]
            for i in range(4):
                lofo_tots[i] += val_f_tots[i]
            lofo_choices[val_f] = {
                "selected_params": [float(x) for x in best_tr_key],
                "train_pq": best_tr_pq,
                "val_pq": agg_pq(val_f_tots)["pq"],
            }
            for obs in obs_by_fold[val_f]:
                o_iou, o_tp, o_fp, o_fn = obs_dict[best_tr_key][obs]
                lofo_per_obs[obs] = {"iou_sum": o_iou, "tp": o_tp, "fp": o_fp, "fn": o_fn}

        return {
            "global_best_params": [float(x) for x in best_glob_key],
            "global_best": glob_res,
            "nested_cv": agg_pq(lofo_tots),
            "nested_cv_folds": lofo_choices,
            "nested_cv_per_obs": lofo_per_obs,
        }

    keys_thresh_m0 = [k for k in thresh_fold_ann.keys() if k[0] == 1.0 and k[1] == 1.0]
    keys_thresh_m1 = list(thresh_fold_ann.keys())
    keys_scorer_r1_g1 = [(1.0, 1.0, ty) for ty in TAU_Y_GRID]
    keys_scorer_m1 = list(scorer_m1_fold_ann.keys())

    res_ann = {
        "M0_threshold_r1_g1": evaluate_nested_and_global(thresh_fold_ann, thresh_obs_ann, keys_thresh_m0),
        "M1_threshold_weighted_wmf": evaluate_nested_and_global(thresh_fold_ann, thresh_obs_ann, keys_thresh_m1),
        "M0_scorer_v2_19feats_r1_g1": evaluate_nested_and_global(m0_sc_fold_ann, m0_sc_obs_ann, keys_scorer_r1_g1),
        "M1a_scorer_v2_23feats_r1_g1": evaluate_nested_and_global(scorer_m1_fold_ann, scorer_m1_obs_ann, keys_scorer_r1_g1),
        "M1_full_weighted_wmf_plus_scorer_v2_23feats": evaluate_nested_and_global(scorer_m1_fold_ann, scorer_m1_obs_ann, keys_scorer_m1),
    }

    res_cv2 = {
        "M0_threshold_r1_g1": evaluate_nested_and_global(thresh_fold_cv2, thresh_obs_cv2, keys_thresh_m0),
        "M1_threshold_weighted_wmf": evaluate_nested_and_global(thresh_fold_cv2, thresh_obs_cv2, keys_thresh_m1),
        "M0_scorer_v2_19feats_r1_g1": evaluate_nested_and_global(m0_sc_fold_cv2, m0_sc_obs_cv2, keys_scorer_r1_g1),
        "M1a_scorer_v2_23feats_r1_g1": evaluate_nested_and_global(scorer_m1_fold_cv2, scorer_m1_obs_cv2, keys_scorer_r1_g1),
        "M1_full_weighted_wmf_plus_scorer_v2_23feats": evaluate_nested_and_global(scorer_m1_fold_cv2, scorer_m1_obs_cv2, keys_scorer_m1),
    }

    print("4. Running 10,000-iteration paired bootstraps across all 707 images...", flush=True)
    bootstraps_ann = {
        "M1_thresh_vs_M0_thresh": paired_bootstrap_from_per_obs(
            res_ann["M0_threshold_r1_g1"]["nested_cv_per_obs"],
            res_ann["M1_threshold_weighted_wmf"]["nested_cv_per_obs"],
            n_boot=10000,
            seed=2026,
        ),
        "M1_full_vs_M0_thresh": paired_bootstrap_from_per_obs(
            res_ann["M0_threshold_r1_g1"]["nested_cv_per_obs"],
            res_ann["M1_full_weighted_wmf_plus_scorer_v2_23feats"]["nested_cv_per_obs"],
            n_boot=10000,
            seed=2026,
        ),
        "M1_full_vs_M0_scorer": paired_bootstrap_from_per_obs(
            res_ann["M0_scorer_v2_19feats_r1_g1"]["nested_cv_per_obs"],
            res_ann["M1_full_weighted_wmf_plus_scorer_v2_23feats"]["nested_cv_per_obs"],
            n_boot=10000,
            seed=2026,
        ),
        "M1a_23feats_vs_M0_scorer": paired_bootstrap_from_per_obs(
            res_ann["M0_scorer_v2_19feats_r1_g1"]["nested_cv_per_obs"],
            res_ann["M1a_scorer_v2_23feats_r1_g1"]["nested_cv_per_obs"],
            n_boot=10000,
            seed=2026,
        ),
        "M1a_23feats_vs_M0_thresh": paired_bootstrap_from_per_obs(
            res_ann["M0_threshold_r1_g1"]["nested_cv_per_obs"],
            res_ann["M1a_scorer_v2_23feats_r1_g1"]["nested_cv_per_obs"],
            n_boot=10000,
            seed=2026,
        ),
    }

    for d in (res_ann, res_cv2):
        for k in d:
            d[k].pop("nested_cv_per_obs", None)

    task1_summary = {
        "threshold_rule_085_055": {
            "annToMask": {k: agg_pq(v) for k, v in task1_thresh_ann.items()},
            "cv2.fillPoly": {k: agg_pq(v) for k, v in task1_thresh_cv2.items()},
        },
        "scorer_v2_0250_0525": {
            "annToMask": {k: agg_pq(v) for k, v in task1_scorer_ann.items()},
            "cv2.fillPoly": {k: agg_pq(v) for k, v in task1_scorer_cv2.items()},
        },
    }

    final_out = {
        "elapsed_sec": round(time.time() - t_start, 1),
        "task1_erosion_dilation_5fam": task1_summary,
        "task2_and_3_annToMask": res_ann,
        "task2_and_3_cv2_fillPoly": res_cv2,
        "task2_paired_bootstraps_10k": bootstraps_ann,
    }

    out_json = REPO_ROOT / "audit" / "tasks_1_2_3_results.json"
    with open(out_json, "w") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nSaved complete results to {out_json} in {time.time() - t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
