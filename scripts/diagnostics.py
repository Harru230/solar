#!/usr/bin/env python3
"""
scripts/diagnostics.py - Diagnostics on fold 0 predictions.

Evaluates:
1. Predictions with best-GT IoU in [0.35, 0.50] (near-miss false positives).
2. PQ breakdown on observations with 1, 2, and 3 annotators.
3. PQ under official counting vs strict 1:1 matching.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import find_dataset_paths


def _rles(strings):
    return [{"size": [2048, 2048], "counts": str(s).encode("ascii")} for s in strings]


def compute_iou_matrix(gt_strings, pred_strings):
    if not gt_strings or not pred_strings:
        return np.zeros((len(gt_strings), len(pred_strings)))
    iou = mu.iou(_rles(pred_strings), _rles(gt_strings), [0] * len(gt_strings))
    return np.asarray(iou, dtype=np.float64).T


def match_1to1(m: np.ndarray, threshold: float = 0.5):
    """
    Greedy 1:1 matching by descending IoU.
    Returns (matched_pairs, fp_count, fn_count).
    """
    n_gt, n_pred = m.shape
    if n_gt == 0:
        return [], n_pred, 0
    if n_pred == 0:
        return [], 0, n_gt

    # Find candidates > threshold
    pairs = []
    for r in range(n_gt):
        for c in range(n_pred):
            if m[r, c] > threshold:
                pairs.append((m[r, c], r, c))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou_val, r, c in pairs:
        if r not in matched_gt and c not in matched_pred:
            matched_gt.add(r)
            matched_pred.add(c)
            matches.append(iou_val)

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    return matches, fp, fn


def run_diagnostics(pred_path: str, fold: int = 0):
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    val_obs = folds_df.loc[folds_df["fold"] == fold, "observation"].tolist()
    val_obs_set = set(val_obs)

    gt_df = pd.read_csv(paths["gt_all_csv"])
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]
    val_gt = gt_df[gt_img.isin(val_obs_set)].copy()

    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df[pred_df["segmentation_rle"] != "PPPP4"].copy()

    pred_img = pred_df["filament_id"].str.split("_", n=1).str[0]
    pred_by_img = pred_df.groupby(pred_img)["segmentation_rle"].apply(list).to_dict()

    print("==================================================================")
    print(f"DIAGNOSTICS REPORT: {pred_path} (Fold {fold})")
    print(f"Total validation observations: {len(val_obs)}")
    print(f"Total predictions: {len(pred_df)}")
    print("==================================================================")

    # 1. Best-GT IoU per prediction
    # Group GT by observation (pool all annotators)
    gt_by_obs = defaultdict(list)
    for _, row in val_gt.iterrows():
        fid = row["filament_id"]
        obs = fid.split("_")[0].split("-", 1)[1]
        gt_by_obs[obs].append(row["segmentation_rle"])

    # Also per annotator-image
    gt_by_ai = defaultdict(list)
    for _, row in val_gt.iterrows():
        ai = row["filament_id"].split("_")[0]
        gt_by_ai[ai].append(row["segmentation_rle"])

    # For each prediction, find max IoU against pooled GT of that observation
    near_miss_pooled = 0
    total_preds = 0
    ious_pooled = []

    for obs, p_rles in pred_by_img.items():
        g_rles = gt_by_obs.get(obs, [])
        if not g_rles:
            for _ in p_rles:
                ious_pooled.append(0.0)
                total_preds += 1
            continue

        iou_mat = compute_iou_matrix(g_rles, p_rles)  # shape (n_gt, n_pred)
        # Max IoU for each prediction (column) across all GT masks of this obs
        best_ious = iou_mat.max(axis=0)
        for val in best_ious:
            ious_pooled.append(val)
            total_preds += 1
            if 0.35 <= val <= 0.50:
                near_miss_pooled += 1

    ious_pooled = np.array(ious_pooled)
    print("\n1. PREDICTIONS WITH BEST-GT IoU IN [0.35, 0.50] (Pooled across annotators):")
    print(f"   Near-miss count [0.35, 0.50]: {near_miss_pooled} / {total_preds} ({near_miss_pooled / total_preds * 100:.2f}%)")
    print(f"   IoU < 0.35:                   {np.sum(ious_pooled < 0.35)} ({np.sum(ious_pooled < 0.35) / total_preds * 100:.2f}%)")
    print(f"   IoU >= 0.50 (Matches):        {np.sum(ious_pooled >= 0.50)} ({np.sum(ious_pooled >= 0.50) / total_preds * 100:.2f}%)")

    # Near misses per-annotator evaluation
    near_miss_per_ai = 0
    total_pred_ai_evals = 0
    for ai, g_rles in gt_by_ai.items():
        obs = ai.split("-", 1)[1]
        p_rles = pred_by_img.get(obs, [])
        if not p_rles:
            continue
        iou_mat = compute_iou_matrix(g_rles, p_rles)
        best_ious = iou_mat.max(axis=0) if iou_mat.shape[0] > 0 else np.zeros(len(p_rles))
        for val in best_ious:
            total_pred_ai_evals += 1
            if 0.35 <= val <= 0.50:
                near_miss_per_ai += 1

    print(f"   Per-annotator evaluation slice: {near_miss_per_ai} / {total_pred_ai_evals} ({near_miss_per_ai / total_pred_ai_evals * 100:.2f}%)")

    # 2. PQ by number of annotators
    # Count annotators per observation
    obs_annotators = defaultdict(set)
    for ai in gt_by_ai:
        ann, obs = ai.split("-", 1)
        obs_annotators[obs].add(ann)

    counts_dist = defaultdict(list)
    for obs in val_obs:
        n_ann = len(obs_annotators.get(obs, set()))
        counts_dist[n_ann].append(obs)

    print("\n2. PQ BREAKDOWN BY NUMBER OF ANNOTATORS:")
    print(f"   Observation distribution: 1 ann: {len(counts_dist[1])}, 2 ann: {len(counts_dist[2])}, 3 ann: {len(counts_dist[3])}")

    for k in sorted(counts_dist.keys()):
        obs_subset = set(counts_dist[k])
        # Filter GT
        sub_gt = val_gt[val_gt["filament_id"].apply(lambda fid: fid.split("_")[0].split("-", 1)[1] in obs_subset)].copy()
        # Filter Pred
        sub_pred = pred_df[pred_df["filament_id"].apply(lambda fid: fid.split("_")[0] in obs_subset)].copy()

        score_res = pq_eval.score(sub_gt, sub_pred)
        print(f"   --- {k} Annotator(s) ({len(counts_dist[k])} obs, {score_res['annotator_images']} annotator-images) ---")
        print(f"       PQ: {score_res['pq']:.4f} | SQ: {score_res['sq']:.4f} | RQ: {score_res['rq']:.4f}")
        print(f"       TP: {score_res['tp']} | FP: {score_res['fp']} | FN: {score_res['fn']} | pred_multi_hit: {score_res['pred_multi_hit']}")

    # Overall Official Score
    overall_res = pq_eval.score(val_gt, pred_df)
    print("\n   --- Overall (All Observations) ---")
    print(f"       PQ: {overall_res['pq']:.4f} | SQ: {overall_res['sq']:.4f} | RQ: {overall_res['rq']:.4f}")
    print(f"       TP: {overall_res['tp']} | FP: {overall_res['fp']} | FN: {overall_res['fn']}")

    # 3. Official Counting vs 1:1 Matching
    print("\n3. OFFICIAL COUNTING VS STRICT 1:1 MATCHING:")
    matrices = []
    for ai, rows in val_gt.groupby(val_gt["filament_id"].str.split("_", n=1).str[0], sort=False):
        obs = ai.split("-", 1)[1]
        p_rles = pred_by_img.get(obs, [])
        matrices.append(compute_iou_matrix(rows["segmentation_rle"].tolist(), p_rles))

    # Official counting
    off_tp_iou, off_fp, off_fn = [], 0, 0
    strict_tp_iou, strict_fp, strict_fn = [], 0, 0

    for m in matrices:
        n_gt, n_pred = m.shape
        if n_gt == 0:
            off_fp += n_pred
            strict_fp += n_pred
            continue
        if n_pred == 0:
            off_fn += n_gt
            strict_fn += n_gt
            continue

        # Official
        hit = m > 0.5
        off_tp_iou.extend(m[hit].tolist())
        off_fp += int((hit.sum(axis=0) == 0).sum())
        off_fn += int((hit.sum(axis=1) == 0).sum())

        # Strict 1:1
        matches_1to1, fp_1to1, fn_1to1 = match_1to1(m, threshold=0.5)
        strict_tp_iou.extend(matches_1to1)
        strict_fp += fp_1to1
        strict_fn += fn_1to1

    def calc_pq(tp_ious, fp, fn):
        tp = len(tp_ious)
        denom = tp + 0.5 * fp + 0.5 * fn
        tot = sum(tp_ious)
        return (tot / denom if denom > 0 else 0.0), (tot / tp if tp > 0 else 0.0), (tp / denom if denom > 0 else 0.0), tp, fp, fn

    off_pq, off_sq, off_rq, off_tp, off_fp, off_fn = calc_pq(off_tp_iou, off_fp, off_fn)
    str_pq, str_sq, str_rq, str_tp, str_fp, str_fn = calc_pq(strict_tp_iou, strict_fp, strict_fn)

    print(f"   Metric          Official Counting   Strict 1:1 Matching   Difference")
    print(f"   --------------------------------------------------------------------")
    print(f"   PQ              {off_pq:.6f}            {str_pq:.6f}          {str_pq - off_pq:+.6f}")
    print(f"   SQ              {off_sq:.6f}            {str_sq:.6f}          {str_sq - off_sq:+.6f}")
    print(f"   RQ              {off_rq:.6f}            {str_rq:.6f}          {str_rq - off_rq:+.6f}")
    print(f"   TP              {off_tp:<19d} {str_tp:<21d} {str_tp - off_tp:+d}")
    print(f"   FP              {off_fp:<19d} {str_fp:<21d} {str_fp - off_fp:+d}")
    print(f"   FN              {off_fn:<19d} {str_fn:<21d} {str_fn - off_fn:+d}")
    print("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="work/pred_fold0.csv")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    run_diagnostics(args.pred, args.fold)
