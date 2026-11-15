#!/usr/bin/env python3
"""
scripts/run_grid_extension.py
Evaluates extended threshold grid ({0.90, 0.925, 0.95}) for 3-family threshold rule
and extended scorer tau_y grid ({0.275 .. 0.320}) on CPU using 5-fold Nested-CV and paired bootstrap.
Uses prefix-invariant canvas painting and IOU matrix caching.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLAR_ROOT = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))).resolve()
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import find_dataset_paths, EMPTY_RLE
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    build_gt_rles_by_obs,
    load_fused_oof_pool,
    evaluate_threshold_grid_nested_cv,
    extract_scorer_v2_dataset,
    paired_bootstrap_from_per_obs,
    encode_crop_to_rle,
    eval_single_obs_rles
)


def evaluate_scorer_prefix_cache(
    obs_cache: Dict[str, Tuple[np.ndarray, List[np.ndarray]]],
    folds_df: pd.DataFrame,
    tau_grid: List[float],
) -> dict:
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    fold_tau_metrics: Dict[int, Dict[float, dict]] = {f: {} for f in range(5)}
    fold_obs_metrics: Dict[int, Dict[float, Dict[str, dict]]] = {f: {} for f in range(5)}

    for f in range(5):
        for tau in tau_grid:
            iou_s, tp, fp, fn = 0.0, 0, 0, 0
            per_obs = {}
            for obs in obs_by_fold[f]:
                scores_arr, iou_mats = obs_cache[obs]
                k_keep = int(np.sum(scores_arr > tau))
                obs_iou, obs_tp, obs_fp, obs_fn = 0.0, 0, 0, 0
                for m_full in iou_mats:
                    m = m_full[:, :k_keep]
                    n_gt, n_pred = m.shape
                    if n_gt == 0:
                        obs_fp += n_pred
                    elif n_pred == 0:
                        obs_fn += n_gt
                    else:
                        hit = m > pq_eval.IOU_THRESHOLD
                        obs_iou += float(m[hit].sum())
                        n_tp = int(hit.sum())
                        obs_tp += n_tp
                        obs_fp += n_pred - n_tp
                        obs_fn += n_gt - n_tp
                per_obs[obs] = {
                    "iou_sum": obs_iou,
                    "tp": obs_tp,
                    "fp": obs_fp,
                    "fn": obs_fn,
                    "sq": obs_iou / max(1, obs_tp),
                    "rq": obs_tp / max(1, obs_tp + 0.5 * obs_fp + 0.5 * obs_fn),
                    "pq": (obs_iou / max(1, obs_tp)) * (obs_tp / max(1, obs_tp + 0.5 * obs_fp + 0.5 * obs_fn)),
                }
                iou_s += obs_iou
                tp += obs_tp
                fp += obs_fp
                fn += obs_fn

            sq = iou_s / max(1, tp)
            rq = tp / max(1, tp + 0.5 * fp + 0.5 * fn)
            fold_tau_metrics[f][tau] = {
                "iou_sum": iou_s,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sq": sq,
                "rq": rq,
                "pq": sq * rq,
            }
            fold_obs_metrics[f][tau] = per_obs

    # Nested Cross-Validation (select tau on folds != val_f, evaluate on val_f)
    nested_per_obs = {}
    chosen_taus = []
    lofo_iou, lofo_tp, lofo_fp, lofo_fn = 0.0, 0, 0, 0

    for val_f in range(5):
        best_tr_tau = None
        best_tr_pq = -1.0
        for tau in tau_grid:
            tr_iou = sum(fold_tau_metrics[f][tau]["iou_sum"] for f in range(5) if f != val_f)
            tr_tp = sum(fold_tau_metrics[f][tau]["tp"] for f in range(5) if f != val_f)
            tr_fp = sum(fold_tau_metrics[f][tau]["fp"] for f in range(5) if f != val_f)
            tr_fn = sum(fold_tau_metrics[f][tau]["fn"] for f in range(5) if f != val_f)
            sq = tr_iou / max(1, tr_tp)
            rq = tr_tp / max(1, tr_tp + 0.5 * tr_fp + 0.5 * tr_fn)
            pq = sq * rq
            if pq > best_tr_pq:
                best_tr_pq = pq
                best_tr_tau = tau

        chosen_taus.append(best_tr_tau)
        nested_per_obs.update(fold_obs_metrics[val_f][best_tr_tau])
        lofo_iou += fold_tau_metrics[val_f][best_tr_tau]["iou_sum"]
        lofo_tp += fold_tau_metrics[val_f][best_tr_tau]["tp"]
        lofo_fp += fold_tau_metrics[val_f][best_tr_tau]["fp"]
        lofo_fn += fold_tau_metrics[val_f][best_tr_tau]["fn"]

    nested_sq = lofo_iou / max(1, lofo_tp)
    nested_rq = lofo_tp / max(1, lofo_tp + 0.5 * lofo_fp + 0.5 * lofo_fn)
    nested_pq = nested_sq * nested_rq

    # Retrospective Global Best
    best_glob_tau = None
    best_glob_pq = -1.0
    for tau in tau_grid:
        g_iou = sum(fold_tau_metrics[f][tau]["iou_sum"] for f in range(5))
        g_tp = sum(fold_tau_metrics[f][tau]["tp"] for f in range(5))
        g_fp = sum(fold_tau_metrics[f][tau]["fp"] for f in range(5))
        g_fn = sum(fold_tau_metrics[f][tau]["fn"] for f in range(5))
        pq = (g_iou / max(1, g_tp)) * (g_tp / max(1, g_tp + 0.5 * g_fp + 0.5 * g_fn))
        if pq > best_glob_pq:
            best_glob_pq = pq
            best_glob_tau = tau

    return {
        "chosen_taus": chosen_taus,
        "mean_chosen_tau": float(np.mean(chosen_taus)),
        "nested_cv": {
            "pq": float(nested_pq),
            "sq": float(nested_sq),
            "rq": float(nested_rq),
            "tp": int(lofo_tp),
            "fp": int(lofo_fp),
            "fn": int(lofo_fn),
        },
        "global_best": {
            "tau": float(best_glob_tau),
            "pq": float(best_glob_pq),
        },
        "nested_per_obs": nested_per_obs,
    }


def main():
    argparse.ArgumentParser(
        description="Evaluate extended threshold and scorer grids using 5-fold Nested CV."
    ).parse_args()
    os.chdir(str(SOLAR_ROOT))
    t0 = time.time()
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    gt_df = pd.read_csv(paths["gt_all_csv"])
    gt_by_obs = build_gt_rles_by_obs(gt_df)

    print("Loading 12-view 3-Family OOF pool (clahe + anchors5 + dice)...")
    specs_12v_raw = [("clahe", "work/oof", 4, 0.0), ("a5", "work/oof_a5", 4, 0.0), ("dice", "work/oof_dice", 4, 0.0)]
    pool_12v = load_fused_oof_pool(folds_df, specs_12v_raw)
    print(f"Loaded pool in {time.time() - t0:.1f}s.")

    # 1. 3-Family Threshold Rule Grid Extension
    orig_score_grid = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875]
    ext_score_grid = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]
    mask_grid = [0.475, 0.500, 0.525, 0.550, 0.575]

    print("\n--- Evaluating Original Score Grid on 3-Family 12v ---")
    t1 = time.time()
    eval_orig = evaluate_threshold_grid_nested_cv(pool_12v, folds_df, gt_by_obs, orig_score_grid, mask_grid)
    print(f"Done in {time.time() - t1:.1f}s.")
    print(f"  Orig Best Tau: score={eval_orig['global_best_tau_score']}, mask={eval_orig['global_best_tau_mask']}")
    print(f"  Orig Global Best PQ: {eval_orig['global_best']['pq']:.6f}")
    print(f"  Orig Nested CV PQ:   {eval_orig['nested_cv']['pq']:.6f}")

    print("\n--- Evaluating Extended Score Grid ({0.90, 0.925, 0.95}) on 3-Family 12v ---")
    t2 = time.time()
    eval_ext = evaluate_threshold_grid_nested_cv(pool_12v, folds_df, gt_by_obs, ext_score_grid, mask_grid)
    print(f"Done in {time.time() - t2:.1f}s.")
    print(f"  Ext Best Tau: score={eval_ext['global_best_tau_score']}, mask={eval_ext['global_best_tau_mask']}")
    print(f"  Ext Global Best PQ: {eval_ext['global_best']['pq']:.6f}")
    print(f"  Ext Nested CV PQ:   {eval_ext['nested_cv']['pq']:.6f}")

    boot_thresh = paired_bootstrap_from_per_obs(eval_orig["nested_cv_per_obs"], eval_ext["nested_cv_per_obs"])
    print(f"Threshold Rule Bootstrap (Ext vs Orig):")
    print(f"  Point Delta: {boot_thresh['point_delta']:+.6f}")
    print(f"  95% CI:      [{boot_thresh['ci_lower']:+.6f}, {boot_thresh['ci_upper']:+.6f}]")
    print(f"  P(Ext > Orig): {boot_thresh['p_b_better']*100:.1f}%")

    # 2. Scorer v2 Tau_y Grid Extension
    print("\n--- Extracting Scorer v2 Dataset & Fitting Cross-Fitted Models ---")
    t3 = time.time()
    feat_12v, obs_pool_12v = extract_scorer_v2_dataset(
        pool_12v, folds_df, gt_df, paths["train_json"], num_views_total=12, tau_mask=0.525
    )
    feature_cols = SCORER_V2_VIEW_INVARIANT_FEATURES

    # Predict out-of-fold pred_y for all candidates
    for val_f in range(5):
        tr_mask = (feat_12v["fold"] != val_f).values
        va_mask = (feat_12v["fold"] == val_f).values
        X_tr = feat_12v.loc[tr_mask, feature_cols].values
        y_tr = feat_12v.loc[tr_mask, "target_y"].values
        X_va = feat_12v.loc[va_mask, feature_cols].values

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
            n_jobs=4,
            verbose=-1,
        )
        m.fit(X_tr, y_tr)
        pred_y_va = m.predict(X_va)
        feat_12v.loc[va_mask, "pred_y"] = pred_y_va

    # Assign pred_y to candidate objects in obs_pool_12v
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    obs_pred_map = feat_12v.groupby("obs")["pred_y"].apply(list).to_dict()
    for obs, pred_list in obs_pred_map.items():
        cands = obs_pool_12v.get(obs, [])
        for c, py in zip(cands, pred_list):
            c["pred_y"] = float(py)

    print(f"Models fitted & pred_y assigned in {time.time() - t3:.1f}s.")

    # Build prefix-invariant observation cache (1 pass over 707 images)
    print("Building prefix-invariant scorer observation cache...")
    t4 = time.time()
    obs_cache: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}
    for f in range(5):
        for obs in obs_by_fold[f]:
            cands = obs_pool_12v.get(obs, [])
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
    print(f"Prefix cache built in {time.time() - t4:.1f}s.")

    orig_tau_y_grid = [round(0.180 + 0.005 * k, 4) for k in range(19)]  # 0.180 .. 0.270
    ext_tau_y_grid = [round(0.180 + 0.005 * k, 4) for k in range(29)]   # 0.180 .. 0.320

    print("\n--- Evaluating Original Scorer tau_y Grid (0.180 .. 0.270) ---")
    t5 = time.time()
    res_sc_orig = evaluate_scorer_prefix_cache(obs_cache, folds_df, orig_tau_y_grid)
    print(f"Done in {time.time() - t5:.2f}s.")
    print(f"  Orig Scorer Nested CV PQ: {res_sc_orig['nested_cv']['pq']:.6f} (Chosen: {res_sc_orig['chosen_taus']}, Mean: {res_sc_orig['mean_chosen_tau']:.4f})")
    print(f"  Orig Scorer Global Best PQ: {res_sc_orig['global_best']['pq']:.6f} at tau={res_sc_orig['global_best']['tau']:.4f}")

    print("\n--- Evaluating Extended Scorer tau_y Grid (0.180 .. 0.320) ---")
    t6 = time.time()
    res_sc_ext = evaluate_scorer_prefix_cache(obs_cache, folds_df, ext_tau_y_grid)
    print(f"Done in {time.time() - t6:.2f}s.")
    print(f"  Ext Scorer Nested CV PQ:  {res_sc_ext['nested_cv']['pq']:.6f} (Chosen: {res_sc_ext['chosen_taus']}, Mean: {res_sc_ext['mean_chosen_tau']:.4f})")
    print(f"  Ext Scorer Global Best PQ: {res_sc_ext['global_best']['pq']:.6f} at tau={res_sc_ext['global_best']['tau']:.4f}")

    boot_scorer = paired_bootstrap_from_per_obs(res_sc_orig["nested_per_obs"], res_sc_ext["nested_per_obs"])
    print(f"\nScorer Bootstrap (Ext vs Orig):")
    print(f"  Point Delta: {boot_scorer['point_delta']:+.6f}")
    print(f"  95% CI:      [{boot_scorer['ci_lower']:+.6f}, {boot_scorer['ci_upper']:+.6f}]")
    print(f"  P(Ext > Orig): {boot_scorer['p_b_better']*100:.1f}%")

    out_json = str(REPO_ROOT / "audit" / "grid_extension_results.json")
    results_summary = {
        "threshold_rule": {
            "orig_grid": orig_score_grid,
            "ext_grid": ext_score_grid,
            "mask_grid": mask_grid,
            "orig_nested_cv": eval_orig["nested_cv"],
            "orig_global_best": eval_orig["global_best"],
            "orig_best_tau_score": eval_orig["global_best_tau_score"],
            "orig_best_tau_mask": eval_orig["global_best_tau_mask"],
            "ext_nested_cv": eval_ext["nested_cv"],
            "ext_global_best": eval_ext["global_best"],
            "ext_best_tau_score": eval_ext["global_best_tau_score"],
            "ext_best_tau_mask": eval_ext["global_best_tau_mask"],
            "bootstrap_ext_vs_orig": {
                "point_delta": boot_thresh["point_delta"],
                "ci_lower": boot_thresh["ci_lower"],
                "ci_upper": boot_thresh["ci_upper"],
                "p_b_better": boot_thresh["p_b_better"],
            }
        },
        "scorer_v2": {
            "orig_tau_grid": orig_tau_y_grid,
            "ext_tau_grid": ext_tau_y_grid,
            "orig_nested_cv": res_sc_orig["nested_cv"],
            "orig_global_best": res_sc_orig["global_best"],
            "orig_chosen_taus": res_sc_orig["chosen_taus"],
            "ext_nested_cv": res_sc_ext["nested_cv"],
            "ext_global_best": res_sc_ext["global_best"],
            "ext_chosen_taus": res_sc_ext["chosen_taus"],
            "bootstrap_ext_vs_orig": {
                "point_delta": boot_scorer["point_delta"],
                "ci_lower": boot_scorer["ci_lower"],
                "ci_upper": boot_scorer["ci_upper"],
                "p_b_better": boot_scorer["p_b_better"],
            }
        }
    }
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved complete grid extension results to {out_json} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
