#!/usr/bin/env python3
"""
scripts/morphology_analysis.py - Quantitative Filament Morphology & Topology Analysis

Computes geometric and topological descriptors across:
  1. Ground Truth annotations (`work/gt_all.csv`, 8,199 instances across 707 images),
     plus multi-annotator consensus vs. single-annotator disagreement morphology on 296 images.
  2. 30-Model OOF Ensemble predictions (`Rule A` & `Rule B` streamed from 6-family OOF caches),
     including morphology-stratified PQ/SQ/RQ and TP/FP/FN morphological signatures.
  3. 30-Model Test submissions (`submission_A_uniform_30models.csv` and
     `submission_B_weighted_30models.csv` across 180 test images), plus OOF-vs-Test
     two-sample Kolmogorov-Smirnov (KS) distributional alignment tests.

Writes a JSON report (`work/morphology_analysis.json` and `audit/morphology_analysis.json`)
containing exclusively directly measured statistics.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, find_dataset_paths
from scripts.instance_scorer import encode_crop_to_rle
from scripts.multifamily_pipeline import FRAME_SIZE, build_gt_rles_by_obs
try:
    from scripts.run_priority_2_cpu import fuse_candidates, load_raw_candidates_per_obs
except ImportError:
    from audit.run_final_analyses import fuse_with_rule, load_oof_raw_candidates

    def load_raw_candidates_per_obs(folds_df: pd.DataFrame) -> Dict[str, List[dict]]:
        work_dir = REPO_ROOT / "work"
        if not (work_dir / "oof").is_dir() and (REPO_ROOT.parent / "solar" / "work" / "oof").is_dir():
            work_dir = REPO_ROOT.parent / "solar" / "work"
        return load_oof_raw_candidates(work_dir, folds_df)

    def fuse_candidates(
        cands: List[dict],
        family_weights: Dict[str, float],
        gamma: float = 1.0,
        total_views: float = 24.0,
    ) -> List[dict]:
        _ = total_views
        return fuse_with_rule(cands, family_weights, gamma=gamma)

CROSS_3X3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

AREA_BINS = [(0, 300, "<300"), (300, 1000, "300-1000"), (1000, 3000, "1000-3000"), (3000, 10**9, ">=3000")]
WIDTH_BINS = [(0.0, 5.0, "<5"), (5.0, 10.0, "5-10"), (10.0, 18.0, "10-18"), (18.0, 1e9, ">=18")]
ELONG_BINS = [(0.0, 3.0, "<3"), (3.0, 7.0, "3-7"), (7.0, 15.0, "7-15"), (15.0, 1e9, ">=15")]

FEATURES_TO_SUMMARIZE = [
    "area_px",
    "skeleton_length_px",
    "perimeter_px",
    "width_mean_px",
    "width_median_px",
    "width_max_px",
    "length_to_width",
    "elongation_pca",
    "major_axis_px",
    "minor_axis_px",
    "solidity",
    "circularity",
    "num_components",
    "orientation_deg",
    "center_dist_px",
]


def assign_bin(val: float, bins: List[Tuple[float, float, str]]) -> str:
    for lo, hi, label in bins:
        if lo <= val < hi:
            return label
    return bins[-1][2]


def decode_rle(rle_str: str) -> np.ndarray:
    return mu.decode({"size": [FRAME_SIZE, FRAME_SIZE], "counts": rle_str.encode("ascii")})


def measure_binary_mask_morphology(full_mask: np.ndarray) -> dict:
    """Measure geometric and topological descriptors of one 2048x2048 binary filament mask."""
    ys, xs = np.nonzero(full_mask)
    area = int(len(ys))
    if area == 0:
        return {}

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop = full_mask[y1:y2, x1:x2].astype(np.uint8)
    # 1px zero border avoids border clipping in distanceTransform & contours
    padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    n_labels, _ = cv2.connectedComponents(crop, connectivity=8)
    num_components = int(n_labels - 1)

    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 0.0
    if contours:
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        hull_canvas = np.zeros_like(padded)
        cv2.fillPoly(hull_canvas, [hull], 1)
        hull_area_px = int((hull_canvas | padded).sum())
        solidity = float(area) / float(max(1, hull_area_px))
    else:
        solidity = 1.0

    circularity = float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
    circularity = min(1.0, circularity)

    # Distance transform -> inscribed diameter = 2 * radius
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    width_max = 2.0 * float(dt.max())

    # Lantuejoul morphological skeletonization on tight padded crop
    skel = np.zeros_like(padded)
    work = padded.copy()
    while True:
        eroded = cv2.erode(work, CROSS_3X3)
        opened = cv2.dilate(eroded, CROSS_3X3)
        skel |= cv2.subtract(work, opened)
        work = eroded
        if not np.any(work):
            break

    skel_mask = (skel > 0) & (padded > 0)
    skel_len = int(skel_mask.sum())
    if skel_len == 0:
        # Fallback if a 1px mask was fully eroded at first step
        skel_mask = padded > 0
        skel_len = int(skel_mask.sum())

    widths_skel = 2.0 * dt[skel_mask]
    width_mean = float(np.mean(widths_skel))
    width_median = float(np.median(widths_skel))
    length_to_width = float(skel_len) / max(1.0, width_mean)

    # Rotation-invariant PCA elongation and solar axis orientation
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean_xy = coords.mean(axis=0)
    centered = coords - mean_xy
    if area > 1:
        cov = (centered.T @ centered) / float(area) + np.eye(2) * (1.0 / 12.0)
    else:
        cov = np.eye(2) * (1.0 / 12.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_desc = np.argsort(eigvals)[::-1]
    l1 = max(1e-6, float(eigvals[idx_desc[0]]))
    l2 = max(1e-6, float(eigvals[idx_desc[1]]))
    v1 = eigvecs[:, idx_desc[0]]
    elongation_pca = math.sqrt(l1 / l2)
    major_axis = 4.0 * math.sqrt(l1)
    minor_axis = 4.0 * math.sqrt(l2)
    angle_deg = math.degrees(math.atan2(v1[1], v1[0])) % 180.0
    # Fold orientation to [0, 90] distance from horizontal solar equator (E-W) or keep [0, 180)
    ew_dev_deg = min(angle_deg, 180.0 - angle_deg)

    cx, cy = float(mean_xy[0]), float(mean_xy[1])
    center_dist = math.hypot(cx - 1024.0, cy - 1024.0)

    return {
        "area_px": area,
        "bbox_w_px": bbox_w,
        "bbox_h_px": bbox_h,
        "skeleton_length_px": skel_len,
        "perimeter_px": perimeter,
        "width_mean_px": width_mean,
        "width_median_px": width_median,
        "width_max_px": width_max,
        "length_to_width": length_to_width,
        "elongation_pca": elongation_pca,
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "solidity": solidity,
        "circularity": circularity,
        "num_components": num_components,
        "orientation_deg": angle_deg,
        "ew_deviation_deg": ew_dev_deg,
        "center_dist_px": center_dist,
        "area_bin": assign_bin(area, AREA_BINS),
        "width_bin": assign_bin(width_mean, WIDTH_BINS),
        "elong_bin": assign_bin(elongation_pca, ELONG_BINS),
        "is_fragmented": int(num_components >= 2),
    }


def summarize_dataframe(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0}
    out = {
        "count": int(len(df)),
        "fragmented_count": int(df["is_fragmented"].sum()),
        "fragmented_pct": float(100.0 * df["is_fragmented"].mean()),
        "ew_aligned_pct_lt_30deg": float(100.0 * (df["ew_deviation_deg"] <= 30.0).mean()),
        "area_bin_counts": df["area_bin"].value_counts().to_dict(),
        "width_bin_counts": df["width_bin"].value_counts().to_dict(),
        "elong_bin_counts": df["elong_bin"].value_counts().to_dict(),
        "features": {},
    }
    for col in FEATURES_TO_SUMMARIZE:
        vals = df[col].dropna().to_numpy(dtype=np.float64)
        out["features"][col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def apply_disjoint_canvas_rles(
    fused_clusters: List[dict],
    tau_score: float,
    tau_mask: float,
    min_area: int = 10,
    area_ret: float = 0.50,
) -> List[str]:
    occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    pred_rles = []
    for c in sorted(fused_clusters, key=lambda x: float(x["score"]), reverse=True):
        if float(c["score"]) < tau_score:
            break
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bc = c["prob_crop"] >= tau_mask
        oa = int(bc.sum())
        if oa < min_area:
            continue
        sub_occ = occupied[y1:y2, x1:x2]
        clean_crop = bc & (~sub_occ)
        ca = int(clean_crop.sum())
        if ca < min_area or (ca / float(oa)) < area_ret:
            continue
        occupied[y1:y2, x1:x2] |= clean_crop
        pred_rles.append(encode_crop_to_rle(box, clean_crop))
    return pred_rles


def measure_submission_csv(csv_path: Path, tag: str) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    rows = []
    empty_images = 0
    obs_counts = defaultdict(int)
    for _, r in df.iterrows():
        fid = str(r["filament_id"])
        img_id = fid.split("_", 1)[0]
        rle = str(r["segmentation_rle"]).strip()
        if not rle or rle == EMPTY_RLE:
            empty_images += 1
            obs_counts.setdefault(img_id, 0)
            continue
        obs_counts[img_id] += 1
        m = decode_rle(rle)
        morph = measure_binary_mask_morphology(m)
        if morph:
            morph["tag"] = tag
            morph["image_id"] = img_id
            morph["filament_id"] = fid
            morph["rle"] = rle
            rows.append(morph)
    morph_df = pd.DataFrame(rows)
    counts_arr = np.array(list(obs_counts.values()), dtype=np.float64)
    img_stats = {
        "n_images": len(obs_counts),
        "empty_prediction_images": int(empty_images),
        "instances_per_image_mean": float(counts_arr.mean()) if len(counts_arr) else 0.0,
        "instances_per_image_median": float(np.median(counts_arr)) if len(counts_arr) else 0.0,
    }
    return morph_df, img_stats


def compute_ks_alignment(df_oof: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    res = {}
    for col in FEATURES_TO_SUMMARIZE:
        a = df_oof[col].dropna().to_numpy(dtype=np.float64)
        b = df_test[col].dropna().to_numpy(dtype=np.float64)
        ks = stats.ks_2samp(a, b)
        med_a = float(np.median(a))
        med_b = float(np.median(b))
        res[col] = {
            "oof_median": med_a,
            "test_median": med_b,
            "median_rel_shift_pct": float(100.0 * (med_b - med_a) / med_a) if med_a != 0 else 0.0,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
    return res


def analyze_gt(gt_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Measure all 8,199 GT instances + multi-annotator consensus vs single-annotator split."""
    rows = []
    # Parse record key "<annotator>-<obs>"
    ann_imgs = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_df = gt_df.copy()
    gt_df["ann_img"] = ann_imgs
    gt_df["annotator"] = ann_imgs.str.split("-", n=1).str[0]
    gt_df["obs"] = ann_imgs.str.split("-", n=1).str[1]

    rle_morph_cache = {}
    for _, r in gt_df.iterrows():
        rle = str(r["segmentation_rle"]).strip()
        if not rle or rle == EMPTY_RLE:
            continue
        if rle not in rle_morph_cache:
            rle_morph_cache[rle] = measure_binary_mask_morphology(decode_rle(rle))
        morph = dict(rle_morph_cache[rle])
        morph["filament_id"] = str(r["filament_id"])
        morph["ann_img"] = str(r["ann_img"])
        morph["annotator"] = str(r["annotator"])
        morph["obs"] = str(r["obs"])
        morph["rle"] = rle
        rows.append(morph)

    df_gt_morph = pd.DataFrame(rows)

    # Identify multi-annotator consensus vs single-annotator GT instances on 296 multi-annotator images
    obs_to_ann_imgs = df_gt_morph.groupby("obs")["ann_img"].unique().to_dict()
    multi_obs = {obs for obs, ais in obs_to_ann_imgs.items() if len(ais) >= 2}

    idx_by_ai = df_gt_morph.groupby("ann_img").groups
    multi_consensus_flags = {}
    for obs in sorted(multi_obs):
        ais = obs_to_ann_imgs[obs]
        for ai in ais:
            my_idxs = list(idx_by_ai[ai])
            my_rles = df_gt_morph.loc[my_idxs, "rle"].tolist()
            other_rles = []
            for o_ai in ais:
                if o_ai != ai:
                    other_rles.extend(df_gt_morph.loc[list(idx_by_ai[o_ai]), "rle"].tolist())
            m = pq_eval.iou_matrix(my_rles, other_rles)
            has_peer = (m > pq_eval.IOU_THRESHOLD).any(axis=1) if m.size > 0 else np.zeros(len(my_idxs), dtype=bool)
            for row_idx, agreed in zip(my_idxs, has_peer):
                multi_consensus_flags[row_idx] = bool(agreed)

    agreed_idxs = [idx for idx, flag in multi_consensus_flags.items() if flag]
    single_idxs = [idx for idx, flag in multi_consensus_flags.items() if not flag]

    gt_summary = {
        "all_gt": summarize_dataframe(df_gt_morph),
        "n_unique_observations": int(df_gt_morph["obs"].nunique()),
        "n_annotator_records": int(df_gt_morph["ann_img"].nunique()),
        "n_multi_annotator_observations": len(multi_obs),
        "multi_annotator_agreed_instances": summarize_dataframe(df_gt_morph.loc[agreed_idxs]),
        "multi_annotator_single_only_instances": summarize_dataframe(df_gt_morph.loc[single_idxs]),
    }
    return df_gt_morph, gt_summary


def evaluate_oof_morphology_and_stratified_pq(
    gt_df: pd.DataFrame,
    df_gt_morph: pd.DataFrame,
    clusters_by_obs: Dict[str, List[dict]],
    tau_score: float,
    tau_mask: float,
    tag: str,
) -> Tuple[pd.DataFrame, dict]:
    """Generate OOF RLEs, measure prediction morphology, and compute stratified PQ & TP/FP/FN morphology."""
    gt_by_obs = build_gt_rles_by_obs(gt_df)
    gt_morph_lookup = {}
    for ann_img, sub in df_gt_morph.groupby("ann_img"):
        gt_morph_lookup[ann_img] = sub.reset_index(drop=True)

    # Also keep mapping obs -> [(ann_img, rle_list)]
    gt_records_by_obs = defaultdict(list)
    for ann_img, sub in gt_morph_lookup.items():
        obs = sub.loc[0, "obs"]
        gt_records_by_obs[obs].append((ann_img, sub))

    pred_rows = []
    # Stratification accumulators: stratum_col -> stratum_val -> dict(iou_sum, tp, fp, fn)
    strata_cols = ["area_bin", "width_bin", "elong_bin", "is_fragmented"]
    strata_acc = {col: defaultdict(lambda: {"iou_sum": 0.0, "tp": 0, "fp": 0, "fn": 0}) for col in strata_cols}

    global_acc = {"iou_sum": 0.0, "tp": 0, "fp": 0, "fn": 0}
    tp_pred_rows, fp_pred_rows, fn_gt_rows = [], [], []

    all_obs = sorted(gt_records_by_obs.keys())
    for obs in all_obs:
        pred_rles = apply_disjoint_canvas_rles(clusters_by_obs.get(obs, []), tau_score, tau_mask)
        pred_morphs = []
        for k, rle in enumerate(pred_rles):
            m = measure_binary_mask_morphology(decode_rle(rle))
            m["obs"] = obs
            m["filament_id"] = f"{obs}_{k}"
            m["rle"] = rle
            pred_morphs.append(m)
            pred_rows.append(m)

        for ann_img, gt_sub in gt_records_by_obs[obs]:
            gt_rles = gt_sub["rle"].tolist()
            mat = pq_eval.iou_matrix(gt_rles, pred_rles)
            n_g, n_p = mat.shape
            hit = mat > pq_eval.IOU_THRESHOLD

            # Record TPs and FNs per GT instance
            for gi in range(n_g):
                gm_row = gt_sub.iloc[gi]
                if hit.shape[1] > 0 and hit[gi].any():
                    best_pj = int(np.argmax(mat[gi]))
                    iou_val = float(mat[gi, best_pj])
                    # Sum all > 0.5 hits just as pq_eval does
                    gi_hits = np.where(hit[gi])[0]
                    for pj in gi_hits:
                        v = float(mat[gi, pj])
                        global_acc["iou_sum"] += v
                        global_acc["tp"] += 1
                        tp_pred_rows.append(pred_morphs[pj])
                        for col in strata_cols:
                            s_val = gm_row[col]
                            strata_acc[col][s_val]["iou_sum"] += v
                            strata_acc[col][s_val]["tp"] += 1
                else:
                    global_acc["fn"] += 1
                    fn_gt_rows.append(gm_row.to_dict())
                    for col in strata_cols:
                        s_val = gm_row[col]
                        strata_acc[col][s_val]["fn"] += 1

            # Record FPs per predicted instance
            for pj in range(n_p):
                if hit.shape[0] == 0 or not hit[:, pj].any():
                    global_acc["fp"] += 1
                    pm_row = pred_morphs[pj]
                    fp_pred_rows.append(pm_row)
                    for col in strata_cols:
                        s_val = pm_row[col]
                        strata_acc[col][s_val]["fp"] += 1

    def finalize_acc(acc):
        tp, fp, fn, iou_sum = acc["tp"], acc["fp"], acc["fn"], acc["iou_sum"]
        denom = tp + 0.5 * fp + 0.5 * fn
        return {
            "pq": float(iou_sum / denom) if denom > 0 else 0.0,
            "sq": float(iou_sum / tp) if tp > 0 else 0.0,
            "rq": float(tp / denom) if denom > 0 else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    stratified_results = {}
    for col in strata_cols:
        stratified_results[col] = {str(k): finalize_acc(v) for k, v in strata_acc[col].items()}

    df_pred = pd.DataFrame(pred_rows)
    obs_pred_counts = df_pred.groupby("obs").size().reindex(all_obs, fill_value=0).to_numpy(dtype=np.float64)
    summary = {
        "tag": tag,
        "global_pq": finalize_acc(global_acc),
        "n_images": len(all_obs),
        "empty_prediction_images": int((obs_pred_counts == 0).sum()),
        "instances_per_image_mean": float(obs_pred_counts.mean()),
        "instances_per_image_median": float(np.median(obs_pred_counts)),
        "morphology": summarize_dataframe(df_pred),
        "stratified_pq": stratified_results,
        "tp_pred_morphology": summarize_dataframe(pd.DataFrame(tp_pred_rows)),
        "fp_pred_morphology": summarize_dataframe(pd.DataFrame(fp_pred_rows)),
        "fn_gt_morphology": summarize_dataframe(pd.DataFrame(fn_gt_rows)),
    }
    return df_pred, summary


def main():
    parser = argparse.ArgumentParser(description="Quantitative filament morphology analysis across GT, OOF, and Test")
    parser.add_argument("--config", type=str, default="configs/final_ensemble.json", help="Ensemble config path")
    parser.add_argument(
        "--out",
        type=str,
        default="work/morphology_analysis.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        default="",
        help="Optional secondary output JSON path (e.g. ../solar_release/audit/morphology_analysis.json)",
    )
    args = parser.parse_args()

    t0 = time.time()
    paths = find_dataset_paths(str(REPO_ROOT))
    gt_df = pd.read_csv(paths["gt_all_csv"])
    folds_df = pd.read_csv(paths["folds_csv"])

    with open(REPO_ROOT / args.config, "r") as f:
        cfg = json.load(f)

    print("[1/4] Analyzing Ground Truth morphology (8,199 instances across 707 images)...", flush=True)
    df_gt_morph, gt_summary = analyze_gt(gt_df)
    print(f"  GT done: {len(df_gt_morph)} instances in {time.time() - t0:.1f}s", flush=True)

    print("[2/4] Loading 30-model OOF candidates and fusing Rule A & Rule B...", flush=True)
    obs_raw_cands = load_raw_candidates_per_obs(folds_df)
    rule_a = cfg["decision_rules"]["rule_A"]
    rule_b = cfg["decision_rules"]["rule_B"]

    eff_views_a = sum(float(rule_a["family_weights"].get(fam, 1.0)) * 4.0 for fam in rule_a["family_weights"])
    eff_views_b = sum(float(rule_b["family_weights"].get(fam, 1.0)) * 4.0 for fam in rule_b["family_weights"])

    clusters_a = {
        obs: fuse_candidates(cands, rule_a["family_weights"], gamma=float(rule_a["gamma"]), total_views=eff_views_a)
        for obs, cands in obs_raw_cands.items()
    }
    clusters_b = {
        obs: fuse_candidates(cands, rule_b["family_weights"], gamma=float(rule_b["gamma"]), total_views=eff_views_b)
        for obs, cands in obs_raw_cands.items()
    }

    print("[3/4] Evaluating 30-model OOF Rule A & Rule B morphology and stratified PQ...", flush=True)
    df_oof_a, oof_a_summary = evaluate_oof_morphology_and_stratified_pq(
        gt_df, df_gt_morph, clusters_a, float(rule_a["tau_score"]), float(rule_a["tau_mask"]), "oof_rule_A"
    )
    df_oof_b, oof_b_summary = evaluate_oof_morphology_and_stratified_pq(
        gt_df, df_gt_morph, clusters_b, float(rule_b["tau_score"]), float(rule_b["tau_mask"]), "oof_rule_B"
    )
    print(
        f"  OOF Rule A PQ={oof_a_summary['global_pq']['pq']:.4f} (n={len(df_oof_a)}) | "
        f"Rule B PQ={oof_b_summary['global_pq']['pq']:.4f} (n={len(df_oof_b)})",
        flush=True,
    )

    print("[4/4] Measuring 30-model Test submission morphology (Rule A & Rule B, 180 images)...", flush=True)
    sub_a_path = REPO_ROOT / "work/oof/submission_A_uniform_30models.csv"
    sub_b_path = REPO_ROOT / "work/oof/submission_B_weighted_30models.csv"
    df_test_a, test_a_img_stats = measure_submission_csv(sub_a_path, "test_rule_A")
    df_test_b, test_b_img_stats = measure_submission_csv(sub_b_path, "test_rule_B")

    report = {
        "elapsed_sec": float(time.time() - t0),
        "ground_truth": gt_summary,
        "oof_rule_A": oof_a_summary,
        "oof_rule_B": oof_b_summary,
        "test_rule_A": {
            "csv_path": str(sub_a_path),
            "image_stats": test_a_img_stats,
            "morphology": summarize_dataframe(df_test_a),
        },
        "test_rule_B": {
            "csv_path": str(sub_b_path),
            "image_stats": test_b_img_stats,
            "morphology": summarize_dataframe(df_test_b),
        },
        "ks_alignment_oof_vs_test_rule_A": compute_ks_alignment(df_oof_a, df_test_a),
        "ks_alignment_oof_vs_test_rule_B": compute_ks_alignment(df_oof_b, df_test_b),
        "ks_alignment_gt_vs_oof_rule_B": compute_ks_alignment(df_gt_morph, df_oof_b),
    }

    out_path = REPO_ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved morphology analysis to {out_path} ({report['elapsed_sec']:.1f}s)")

    if args.copy_to:
        copy_path = Path(args.copy_to) if os.path.isabs(args.copy_to) else REPO_ROOT / args.copy_to
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(copy_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Copied morphology analysis to {copy_path}")


if __name__ == "__main__":
    main()
