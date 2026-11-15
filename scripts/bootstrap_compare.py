#!/usr/bin/env python3
"""
scripts/bootstrap_compare.py - Paired bootstrap hypothesis test for model comparison.

Resamples validation images (observation stems) with replacement to estimate
the distribution of Delta PQ = PQ(Model B) - PQ(Model A) and computes:
- Point estimate: Delta PQ on the full validation fold
- Mean Delta PQ over bootstrap resamples
- 95% Confidence Interval (2.5% to 97.5% percentiles)
- P(Model B > Model A)
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import find_dataset_paths


def precompute_stem_matrices(
    pred_a_df: pd.DataFrame,
    pred_b_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    stems: List[str],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    Precomputes pairwise IoU matrices between GT and predictions for each stem.
    Returns:
      matrices_a: stem -> list of IoU matrices (one per annotator-image)
      matrices_b: stem -> list of IoU matrices (one per annotator-image)
    """
    pred_a_img = pred_a_df["filament_id"].str.split("_", n=1).str[0]
    pred_b_img = pred_b_df["filament_id"].str.split("_", n=1).str[0]

    rles_a = pred_a_df.groupby(pred_a_img)["segmentation_rle"].apply(list).to_dict()
    rles_b = pred_b_df.groupby(pred_b_img)["segmentation_rle"].apply(list).to_dict()

    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]

    matrices_a = defaultdict(list)
    matrices_b = defaultdict(list)

    for ai, rows in gt_df.groupby(gt_ai, sort=False):
        stem = ai.split("-", 1)[1]
        if stem not in stems:
            continue
        gt_strings = rows["segmentation_rle"].tolist()
        pred_a_strings = rles_a.get(stem, [])
        pred_b_strings = rles_b.get(stem, [])

        mat_a = pq_eval.iou_matrix(gt_strings, pred_a_strings)
        mat_b = pq_eval.iou_matrix(gt_strings, pred_b_strings)

        matrices_a[stem].append(mat_a)
        matrices_b[stem].append(mat_b)

    return matrices_a, matrices_b


def run_paired_bootstrap(
    pred_a_path: str,
    pred_b_path: str,
    gt_path: str,
    folds_path: Optional[str] = None,
    fold: Optional[int] = 0,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict:
    df_a = pd.read_csv(pred_a_path)
    df_b = pd.read_csv(pred_b_path)
    gt_df = pd.read_csv(gt_path)

    # Filter fold if provided
    if folds_path and os.path.isfile(folds_path) and fold is not None:
        folds_df = pd.read_csv(folds_path)
        val_obs = set(folds_df.loc[folds_df["fold"] == fold, "observation"])
        gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
        gt_img = gt_ai.str.split("-", n=1).str[1]
        gt_df = gt_df[gt_img.isin(val_obs)].copy()
        stems = sorted(list(val_obs))
    else:
        gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
        stems = sorted(list(gt_ai.str.split("-", n=1).str[1].unique()))

    print(f"Precomputing IoU matrices across {len(stems)} unique stems...")
    matrices_a, matrices_b = precompute_stem_matrices(df_a, df_b, gt_df, stems)

    # Global point evaluation
    all_m_a = [m for s in stems for m in matrices_a[s]]
    all_m_b = [m for s in stems for m in matrices_b[s]]

    full_a = pq_eval.pq_from_matrices(all_m_a)
    full_b = pq_eval.pq_from_matrices(all_m_b)
    point_delta = full_b["pq"] - full_a["pq"]

    print(f"Running {n_boot} paired bootstrap iterations with seed {seed}...")
    rng = np.random.default_rng(seed)
    n_stems = len(stems)
    stems_arr = np.array(stems)

    boot_deltas = []
    boot_pqs_a = []
    boot_pqs_b = []

    for _ in range(n_boot):
        sampled_indices = rng.choice(n_stems, size=n_stems, replace=True)
        sampled_stems = stems_arr[sampled_indices]

        sampled_m_a = [m for s in sampled_stems for m in matrices_a[s]]
        sampled_m_b = [m for s in sampled_stems for m in matrices_b[s]]

        pq_a = pq_eval.pq_from_matrices(sampled_m_a)["pq"]
        pq_b = pq_eval.pq_from_matrices(sampled_m_b)["pq"]

        boot_pqs_a.append(pq_a)
        boot_pqs_b.append(pq_b)
        boot_deltas.append(pq_b - pq_a)

    boot_deltas = np.array(boot_deltas)
    mean_delta = float(np.mean(boot_deltas))
    std_delta = float(np.std(boot_deltas))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_b_better = float(np.mean(boot_deltas > 0))

    return {
        "pred_a": pred_a_path,
        "pred_b": pred_b_path,
        "n_stems": n_stems,
        "n_boot": n_boot,
        "full_a": full_a,
        "full_b": full_b,
        "point_delta": point_delta,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_b_better": p_b_better,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap test over images (stem = unit)")
    parser.add_argument("--pred_a", type=str, required=True, help="Baseline predictions CSV")
    parser.add_argument("--pred_b", type=str, required=True, help="Comparison predictions CSV")
    parser.add_argument("--gt", type=str, default="work/gt_all.csv", help="Ground truth CSV")
    parser.add_argument("--folds", type=str, default="work/folds.csv", help="Folds CSV")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate (default: 0)")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    res = run_paired_bootstrap(
        pred_a_path=args.pred_a,
        pred_b_path=args.pred_b,
        gt_path=args.gt,
        folds_path=args.folds,
        fold=args.fold,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print("\n==================================================================")
    print("           PAIRED BOOTSTRAP COMPARISON REPORT                     ")
    print("==================================================================")
    print(f"Model A (Baseline):  {res['pred_a']}")
    print(f"Model B (Candidate): {res['pred_b']}")
    print(f"Number of Stems:     {res['n_stems']}")
    print(f"Bootstrap Iterations:{res['n_boot']}")
    print("------------------------------------------------------------------")
    print(f"Model A PQ:          {res['full_a']['pq']:.4f}  (SQ: {res['full_a']['sq']:.4f}, RQ: {res['full_a']['rq']:.4f})")
    print(f"Model B PQ:          {res['full_b']['pq']:.4f}  (SQ: {res['full_b']['sq']:.4f}, RQ: {res['full_b']['rq']:.4f})")
    print(f"Point Delta PQ:      {res['point_delta']:+.4f}")
    print("------------------------------------------------------------------")
    print(f"Mean Delta PQ:       {res['mean_delta']:+.4f}")
    print(f"Std Error:           {res['std_delta']:.4f}")
    print(f"95% Bootstrap CI:    [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]")
    print(f"P(Model B > Model A):{res['p_b_better'] * 100:.1f}%")
    print("==================================================================")
    if res["ci_lower"] > 0:
        print("Conclusion: Model B is STATISTICALLY SIGNIFICANTLY BETTER than Model A (p < 0.05).")
    elif res["ci_upper"] < 0:
        print("Conclusion: Model B is STATISTICALLY SIGNIFICANTLY WORSE than Model A (p < 0.05).")
    else:
        print("Conclusion: No statistically significant difference at 95% confidence level.")
    print("==================================================================")


if __name__ == "__main__":
    main()
