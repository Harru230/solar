#!/usr/bin/env python3
"""
scripts/loss_decomposition.py - Step 0: Panoptic Quality (PQ) Loss Decomposition
on 5-Fold OOF predictions (clahe 5-fold, threshold rule) over all 707 images.

Decomposes all recognition & segmentation errors across all 1,154 annotator-image records into:
1. Complete Misses: Unmatched GT (FN) with best IoU < 0.25 (and not part of a split/merge).
2. Near-Misses (IoU 0.25 - 0.50): Unmatched GT (FN) paired with an unmatched prediction (FP)
   where best IoU is in [0.25, 0.50] (boundary/threshold miss just below the 0.50 cutoff).
3. Splits: A single GT filament broken into >= 2 predicted instances:
   - S1 (FN + >=2 FP fragments): Neither fragment reaches IoU > 0.50 alone, but >=2 fragments overlap the GT.
   - S2 (TP + >=1 FP fragment): Main fragment is TP (IoU > 0.50), extra fragment(s) on the same GT are counted as FP.
4. Merges: A single predicted instance covering >= 2 distinct GT filaments:
   - M1 (1 FP covering >=2 FN GTs): Merged mask has IoU <= 0.50 with each individual GT.
   - M2 (1 TP swallowing >=1 neighbor FN GT): Merged mask matches largest GT (IoU > 0.50) but leaves neighbor GT(s) as FN.
5. Pure False Positives (Hallucinations): Unmatched predictions (FP) with best IoU < 0.25 against all GTs
   (and not fragments of splits/near-misses).
6. TP Segmentation Inexactness (1 - SQ): Matched TP instances having mean IoU ~0.669 instead of 1.0.

Reports exact counts (FN, FP, affected instances) and exact PQ cost (Delta PQ if resolved) for each category.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, find_dataset_paths

SIZE = [2048, 2048]


def rle_areas(rle_strings):
    if not rle_strings:
        return np.zeros(0, dtype=np.float64)
    objs = [{"size": SIZE, "counts": str(s).encode("ascii")} for s in rle_strings]
    return np.asarray(mu.area(objs), dtype=np.float64)


def run_loss_decomposition(pred_csv: str, label: str):
    paths = find_dataset_paths()
    gt_df = pd.read_csv(paths["gt_all_csv"])
    pred_df_raw = pd.read_csv(pred_csv)
    pred_df = pred_df_raw[pred_df_raw["segmentation_rle"] != EMPTY_RLE].copy()

    base_res = pq_eval.score(gt_df, pred_df)
    base_pq = base_res["pq"]
    base_sq = base_res["sq"]
    base_rq = base_res["rq"]
    base_tp = base_res["tp"]
    base_fp = base_res["fp"]
    base_fn = base_res["fn"]
    base_iou_sum = base_sq * base_tp
    base_denom = base_tp + 0.5 * base_fp + 0.5 * base_fn

    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_img = pred_df["filament_id"].str.split("_", n=1).str[0]
    pred_by_img = pred_df.groupby(pred_img)["segmentation_rle"].apply(list).to_dict()

    # Counters and counterfactual deltas: [delta_iou_sum, delta_tp, delta_fp, delta_fn]
    categories = {
        "complete_misses": {"events": 0, "fn": 0, "fp": 0, "delta": np.zeros(4, dtype=np.float64)},
        "near_misses_025_050": {"events": 0, "fn": 0, "fp": 0, "delta": np.zeros(4, dtype=np.float64)},
        "splits": {"events": 0, "s1_fn_events": 0, "s2_tp_events": 0, "fn": 0, "fp": 0, "delta": np.zeros(4, dtype=np.float64)},
        "merges": {"events": 0, "m1_fp_events": 0, "m2_tp_events": 0, "fn": 0, "fp": 0, "delta": np.zeros(4, dtype=np.float64)},
        "pure_false_positives": {"events": 0, "fn": 0, "fp": 0, "delta": np.zeros(4, dtype=np.float64)},
    }

    mean_tp_iou = base_sq  # Expected IoU when a missed/merged/split GT is properly recovered

    for ai, rows in gt_df.groupby(gt_ai, sort=False):
        img = ai.split("-", 1)[1]
        gt_rles = rows["segmentation_rle"].tolist()
        pred_rles = pred_by_img.get(img, [])

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0 and n_pred == 0:
            continue
        if n_pred == 0:
            categories["complete_misses"]["events"] += n_gt
            categories["complete_misses"]["fn"] += n_gt
            categories["complete_misses"]["delta"] += np.array([n_gt * mean_tp_iou, n_gt, 0, -n_gt])
            continue
        if n_gt == 0:
            categories["pure_false_positives"]["events"] += n_pred
            categories["pure_false_positives"]["fp"] += n_pred
            categories["pure_false_positives"]["delta"] += np.array([0.0, 0, -n_pred, 0])
            continue

        iou_mat = pq_eval.iou_matrix(gt_rles, pred_rles)  # (n_gt, n_pred)
        gt_a = rle_areas(gt_rles)[:, None]               # (n_gt, 1)
        pred_a = rle_areas(pred_rles)[None, :]           # (1, n_pred)

        # Exact intersection pixels |g_j cap p_k| = IoU / (1 + IoU) * (|g_j| + |p_k|)
        inter_mat = (iou_mat / (1.0 + iou_mat)) * (gt_a + pred_a)
        # Fraction of prediction p_k that lies inside GT g_j: |g_j cap p_k| / |p_k|
        pred_in_gt = inter_mat / np.maximum(1.0, pred_a)
        # Fraction of GT g_j covered by prediction p_k: |g_j cap p_k| / |g_j|
        gt_in_pred = inter_mat / np.maximum(1.0, gt_a)

        hit = iou_mat > 0.50
        gt_is_tp = hit.sum(axis=1) > 0
        pred_is_tp = hit.sum(axis=0) > 0

        fn_used = np.zeros(n_gt, dtype=bool)
        fp_used = np.zeros(n_pred, dtype=bool)

        # 1. Check SPLITS on each GT j:
        # A GT j is split if >= 2 predictions have pred_in_gt[j, k] >= 0.30 (or iou >= 0.10)
        for j in range(n_gt):
            # Candidate fragments that predominantly lie inside GT j (best GT for k is j)
            frag_ks = [
                k for k in range(n_pred)
                if (pred_in_gt[j, k] >= 0.30 or iou_mat[j, k] >= 0.12)
                and np.argmax(pred_in_gt[:, k]) == j
                and not fp_used[k]
            ]
            if len(frag_ks) >= 2:
                # Compute union IoU of these disjoint fragments with GT j
                sum_inter = float(inter_mat[j, frag_ks].sum())
                sum_pred_a = float(pred_a[0, frag_ks].sum())
                union_denom = float(gt_a[j, 0]) + sum_pred_a - sum_inter
                union_iou = sum_inter / union_denom if union_denom > 0 else 0.0

                fp_frags = [k for k in frag_ks if not pred_is_tp[k]]
                if not gt_is_tp[j] and len(fp_frags) >= 2 and union_iou >= 0.25:
                    # Case S1: FN GT split into >= 2 FP fragments
                    fn_used[j] = True
                    for k in fp_frags:
                        fp_used[k] = True
                    categories["splits"]["events"] += 1
                    categories["splits"]["s1_fn_events"] += 1
                    categories["splits"]["fn"] += 1
                    categories["splits"]["fp"] += len(fp_frags)
                    rec_iou = max(union_iou, mean_tp_iou) if union_iou > 0.50 else mean_tp_iou
                    categories["splits"]["delta"] += np.array([rec_iou, 1, -len(fp_frags), -1])
                elif gt_is_tp[j] and len(fp_frags) >= 1:
                    # Case S2: TP GT accompanied by >= 1 extra split FP fragment(s)
                    tp_k = int(np.argmax(iou_mat[j, :]))
                    old_tp_iou = float(iou_mat[j, tp_k])
                    new_tp_iou = max(old_tp_iou, union_iou)
                    for k in fp_frags:
                        fp_used[k] = True
                    categories["splits"]["events"] += 1
                    categories["splits"]["s2_tp_events"] += 1
                    categories["splits"]["fp"] += len(fp_frags)
                    categories["splits"]["delta"] += np.array([new_tp_iou - old_tp_iou, 0, -len(fp_frags), 0])

        # 2. Check MERGES on each prediction k:
        # A prediction k merges >= 2 GTs if >= 2 GTs have gt_in_pred[j, k] >= 0.30 (or iou >= 0.12)
        for k in range(n_pred):
            if fp_used[k]:
                continue
            merged_js = [
                j for j in range(n_gt)
                if (gt_in_pred[j, k] >= 0.30 or iou_mat[j, k] >= 0.12)
                and np.argmax(gt_in_pred[j, :]) == k
                and not fn_used[j]
            ]
            if len(merged_js) >= 2:
                fn_js = [j for j in merged_js if not gt_is_tp[j]]
                if not pred_is_tp[k] and len(fn_js) >= 2:
                    # Case M1: 1 FP prediction merged >= 2 FN GTs
                    fp_used[k] = True
                    for j in fn_js:
                        fn_used[j] = True
                    m_cnt = len(fn_js)
                    categories["merges"]["events"] += 1
                    categories["merges"]["m1_fp_events"] += 1
                    categories["merges"]["fn"] += m_cnt
                    categories["merges"]["fp"] += 1
                    categories["merges"]["delta"] += np.array([m_cnt * mean_tp_iou, m_cnt, -1, -m_cnt])
                elif pred_is_tp[k] and len(fn_js) >= 1:
                    # Case M2: 1 TP prediction also swallowed >= 1 neighbor FN GT
                    for j in fn_js:
                        fn_used[j] = True
                    m_cnt = len(fn_js)
                    categories["merges"]["events"] += 1
                    categories["merges"]["m2_tp_events"] += 1
                    categories["merges"]["fn"] += m_cnt
                    categories["merges"]["delta"] += np.array([m_cnt * mean_tp_iou, m_cnt, 0, -m_cnt])

        # 3. Check NEAR-MISSES (best IoU in [0.25, 0.50]) on remaining FN GTs and FP predictions:
        for j in range(n_gt):
            if gt_is_tp[j] or fn_used[j]:
                continue
            best_k = int(np.argmax(iou_mat[j, :]))
            best_iou = float(iou_mat[j, best_k])
            if 0.25 <= best_iou <= 0.50:
                fn_used[j] = True
                paired_fp = 0
                if not pred_is_tp[best_k] and not fp_used[best_k]:
                    fp_used[best_k] = True
                    paired_fp = 1
                categories["near_misses_025_050"]["events"] += 1
                categories["near_misses_025_050"]["fn"] += 1
                categories["near_misses_025_050"]["fp"] += paired_fp
                categories["near_misses_025_050"]["delta"] += np.array([mean_tp_iou, 1, -paired_fp, -1])
            else:
                # 4. COMPLETE MISS (best IoU < 0.25)
                fn_used[j] = True
                categories["complete_misses"]["events"] += 1
                categories["complete_misses"]["fn"] += 1
                categories["complete_misses"]["delta"] += np.array([mean_tp_iou, 1, 0, -1])

        # Also check if any remaining unused FP has best IoU in [0.25, 0.50] with an already-handled GT
        for k in range(n_pred):
            if pred_is_tp[k] or fp_used[k]:
                continue
            best_j = int(np.argmax(iou_mat[:, k]))
            best_iou = float(iou_mat[best_j, k])
            if 0.25 <= best_iou <= 0.50:
                fp_used[k] = True
                categories["near_misses_025_050"]["fp"] += 1
                categories["near_misses_025_050"]["delta"] += np.array([0.0, 0, -1, 0])
            else:
                # 5. PURE FALSE POSITIVE (best IoU < 0.25)
                fp_used[k] = True
                categories["pure_false_positives"]["events"] += 1
                categories["pure_false_positives"]["fp"] += 1
                categories["pure_false_positives"]["delta"] += np.array([0.0, 0, -1, 0])

    # Compute exact PQ cost for each category (new_pq - base_pq if that error category is eliminated)
    print("========================================================================================")
    print(f"   STEP 0: OOF PQ LOSS DECOMPOSITION — {label}")
    print("========================================================================================")
    print(f"Baseline OOF PQ: {base_pq:.4f} | SQ: {base_sq:.4f} | RQ: {base_rq:.4f}")
    print(f"Total GT: {base_tp + base_fn} | TP: {base_tp} | FP: {base_fp} | FN: {base_fn}")
    print(f"Total Recognition Loss (1 - RQ): {1.0 - base_rq:.4f} | Segmentation Loss (1 - SQ): {1.0 - base_sq:.4f}")
    print("----------------------------------------------------------------------------------------")
    print(f"{'Error Category':<32} | {'Events':>7} | {'FN':>6} | {'FP':>6} | {'PQ If Fixed':>11} | {'PQ Cost (Delta)':>15}")
    print("----------------------------------------------------------------------------------------")

    summary_rows = []
    total_fn_acc = 0
    total_fp_acc = 0

    for key, label_name in [
        ("complete_misses", "1. Complete Misses (IoU < 0.25)"),
        ("near_misses_025_050", "2. Near-Misses (IoU 0.25-0.50)"),
        ("splits", "3. Splits (1 GT -> >=2 Preds)"),
        ("merges", "4. Merges (>=2 GTs -> 1 Pred)"),
        ("pure_false_positives", "5. Pure False Positives (<0.25)"),
    ]:
        info = categories[key]
        d_iou, d_tp, d_fp, d_fn = info["delta"]
        new_iou = base_iou_sum + d_iou
        new_tp = base_tp + d_tp
        new_fp = base_fp + d_fp
        new_fn = base_fn + d_fn
        new_denom = new_tp + 0.5 * new_fp + 0.5 * new_fn
        new_pq = new_iou / new_denom if new_denom > 0 else 0.0
        pq_cost = new_pq - base_pq

        total_fn_acc += info["fn"]
        total_fp_acc += info["fp"]

        print(f"{label_name:<32} | {info['events']:>7d} | {info['fn']:>6d} | {info['fp']:>6d} | {new_pq:>11.4f} | {pq_cost:>+15.4f}")
        row_dict = {
            "category": key,
            "label": label_name,
            "events": info["events"],
            "fn": info["fn"],
            "fp": info["fp"],
            "pq_if_fixed": float(new_pq),
            "pq_cost": float(pq_cost),
        }
        if key == "splits":
            row_dict["s1_fn_events"] = info["s1_fn_events"]
            row_dict["s2_tp_events"] = info["s2_tp_events"]
        if key == "merges":
            row_dict["m1_fp_events"] = info["m1_fp_events"]
            row_dict["m2_tp_events"] = info["m2_tp_events"]
        summary_rows.append(row_dict)

    # Also show TP boundary segmentation loss (if all current TPs had IoU = 1.0)
    pq_perfect_sq = float(base_tp) / base_denom
    sq_cost = pq_perfect_sq - base_pq
    print("----------------------------------------------------------------------------------------")
    print(f"{'Total FN / FP Accounted':<32} | {'-':>7} | {total_fn_acc:>6d} | {total_fp_acc:>6d} | {'-':>11} | {'-':>15}")
    print(f"{'6. TP Boundary Inexactness (1-SQ)':<32} | {base_tp:>7d} | {0:>6d} | {0:>6d} | {pq_perfect_sq:>11.4f} | {sq_cost:>+15.4f}")
    print("----------------------------------------------------------------------------------------")
    print(
        f"Splits breakdown: S1 (FN split into >=2 FPs) = {categories['splits']['s1_fn_events']} events "
        f"| S2 (TP + extra FP fragment) = {categories['splits']['s2_tp_events']} events"
    )
    print(
        f"Merges breakdown: M1 (1 FP covering >=2 FN GTs) = {categories['merges']['m1_fp_events']} events "
        f"| M2 (1 TP swallowing >=1 FN GT) = {categories['merges']['m2_tp_events']} events"
    )
    print("========================================================================================\n")

    return {
        "baseline": base_res,
        "categories": summary_rows,
        "tp_boundary_inexactness": {
            "tp_count": base_tp,
            "mean_sq": base_sq,
            "pq_if_perfect_sq": pq_perfect_sq,
            "pq_cost": sq_cost,
        },
    }


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Decompose OOF PQ errors (misses, near-misses, splits, merges, false positives)."
    ).parse_args()
    out_nested = run_loss_decomposition("work/oof/oof_predictions_nested_cv.csv", "1536_clahe 5-Fold Nested-CV Threshold Rule (707 imgs)")
    out_default = run_loss_decomposition("work/oof/oof_predictions.csv", "1536_clahe 5-Fold Default (0.85/0.50) Rule (707 imgs)")
    with open("work/oof/loss_decomposition_summary.json", "w") as f:
        json.dump({"nested_cv": out_nested, "default": out_default}, f, indent=2)
