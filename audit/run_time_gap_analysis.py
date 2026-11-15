#!/usr/bin/env python3
"""
audit/run_time_gap_analysis.py - Analyzes per-image PQ against nearest training
observation time gap in 5-fold cross-validation.

Outputs:
  - Plot: audit/per_image_pq_vs_time_gap.png
  - JSON: audit/per_image_pq_vs_time_gap.json
"""
import argparse
import os
import sys
import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLAR_ROOT = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))).resolve()
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import find_dataset_paths


def compute_per_image_pqs(gt_df: pd.DataFrame, pred_df: pd.DataFrame, obs_list: list) -> dict:
    """Computes per-image Panoptic Quality for each observation in obs_list."""
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df["obs"] = gt_df["filament_id"].apply(lambda s: s.split("_")[0].split("-", 1)[1])
    pred_df["obs"] = pred_df["filament_id"].apply(lambda s: s.split("_")[0])

    gt_by_obs = {obs: sub for obs, sub in gt_df.groupby("obs")}
    pred_by_obs = {obs: sub for obs, sub in pred_df.groupby("obs")}

    per_image_pq = {}
    for obs in obs_list:
        g = gt_by_obs.get(obs, pd.DataFrame(columns=gt_df.columns))
        p = pred_by_obs.get(obs, pd.DataFrame(columns=pred_df.columns))
        if len(p) == 0 and len(g) == 0:
            per_image_pq[obs] = 1.0
        elif len(p) == 0 or len(g) == 0:
            per_image_pq[obs] = 0.0
        else:
            res = pq_eval.score(g, p)
            per_image_pq[obs] = float(res["pq"])

    return per_image_pq


def main():
    argparse.ArgumentParser(
        description="Analyze per-image PQ against nearest training observation time gap."
    ).parse_args()
    os.chdir(str(SOLAR_ROOT))
    paths = find_dataset_paths()
    folds = pd.read_csv(paths["folds_csv"])
    gt_df = pd.read_csv(paths["gt_all_csv"])

    train_stems = folds["observation"].tolist()
    train_dts = {o: datetime.datetime.strptime(o[:14], "%Y%m%d%H%M%S") for o in train_stems}
    fold_map = dict(zip(folds["observation"], folds["fold"]))

    # Compute validation gap to nearest training image in other folds
    print("Computing validation time gaps to nearest training observation in fold split...")
    val_gaps_hours = {}
    for v in train_stems:
        f = fold_map[v]
        other_train = [train_dts[u] for u in train_stems if fold_map[u] != f]
        min_diff = min(abs((train_dts[v] - dt).total_seconds()) for dt in other_train) / 3600.0
        val_gaps_hours[v] = min_diff

    # Load OOF predictions for Scorer v2 and Threshold Rule
    sc_pred_path = REPO_ROOT / "work/oof/oof_predictions_instance_scorer.csv"
    tr_pred_path = REPO_ROOT / "work/oof/oof_predictions_nested_cv.csv"

    print("Scoring Scorer v2 per-image PQs...")
    sc_pred_df = pd.read_csv(sc_pred_path)
    sc_pqs = compute_per_image_pqs(gt_df, sc_pred_df, train_stems)

    print("Scoring Threshold Rule per-image PQs...")
    tr_pred_df = pd.read_csv(tr_pred_path)
    tr_pqs = compute_per_image_pqs(gt_df, tr_pred_df, train_stems)

    # Prepare DataFrame
    df = pd.DataFrame({
        "observation": train_stems,
        "fold": [fold_map[o] for o in train_stems],
        "time_gap_hours": [val_gaps_hours[o] for o in train_stems],
        "scorer_v2_pq": [sc_pqs[o] for o in train_stems],
        "threshold_rule_pq": [tr_pqs[o] for o in train_stems],
    })
    df["time_gap_days"] = df["time_gap_hours"] / 24.0

    # Define bins
    bin_edges = [0, 6, 24, 72, 168, 336, 2000]
    bin_labels = ["< 6h", "6 - 24h", "1 - 3d", "3 - 7d", "7 - 14d", "> 14d"]
    df["time_bin"] = pd.cut(df["time_gap_hours"], bins=bin_edges, labels=bin_labels, right=False)

    # Bin statistics
    bin_stats = {}
    for b_label in bin_labels:
        sub = df[df["time_bin"] == b_label]
        n = len(sub)
        if n == 0:
            continue
        sc_vals = sub["scorer_v2_pq"].values
        tr_vals = sub["threshold_rule_pq"].values
        bin_stats[b_label] = {
            "count": int(n),
            "pct_of_total": round(100.0 * n / len(df), 2),
            "gap_hours_min": float(sub["time_gap_hours"].min()),
            "gap_hours_max": float(sub["time_gap_hours"].max()),
            "gap_hours_median": float(sub["time_gap_hours"].median()),
            "scorer_v2": {
                "mean_pq": float(np.mean(sc_vals)),
                "std_pq": float(np.std(sc_vals)),
                "sem_pq": float(np.std(sc_vals) / np.sqrt(n)),
                "median_pq": float(np.median(sc_vals)),
                "p25_pq": float(np.percentile(sc_vals, 25)),
                "p75_pq": float(np.percentile(sc_vals, 75)),
            },
            "threshold_rule": {
                "mean_pq": float(np.mean(tr_vals)),
                "std_pq": float(np.std(tr_vals)),
                "sem_pq": float(np.std(tr_vals) / np.sqrt(n)),
                "median_pq": float(np.median(tr_vals)),
                "p25_pq": float(np.percentile(tr_vals, 25)),
                "p75_pq": float(np.percentile(tr_vals, 75)),
            },
        }

    # Save JSON summary
    out_json = REPO_ROOT / "audit/per_image_pq_vs_time_gap.json"
    summary_data = {
        "dataset_size": len(df),
        "overall_scorer_v2_mean_per_image_pq": float(df["scorer_v2_pq"].mean()),
        "overall_threshold_rule_mean_per_image_pq": float(df["threshold_rule_pq"].mean()),
        "time_gap_hours_summary": {
            "min": float(df["time_gap_hours"].min()),
            "p25": float(df["time_gap_hours"].quantile(0.25)),
            "median": float(df["time_gap_hours"].median()),
            "p75": float(df["time_gap_hours"].quantile(0.75)),
            "max": float(df["time_gap_hours"].max()),
            "mean": float(df["time_gap_hours"].mean()),
        },
        "binned_statistics": bin_stats,
    }
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved stats to {out_json}")

    # Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=False, facecolor="#ffffff"
    )

    x_indices = np.arange(len(bin_labels))
    counts = [bin_stats[b]["count"] for b in bin_labels]
    sc_means = [bin_stats[b]["scorer_v2"]["mean_pq"] for b in bin_labels]
    sc_sems = [bin_stats[b]["scorer_v2"]["sem_pq"] for b in bin_labels]
    tr_means = [bin_stats[b]["threshold_rule"]["mean_pq"] for b in bin_labels]
    tr_sems = [bin_stats[b]["threshold_rule"]["sem_pq"] for b in bin_labels]

    # Top Panel: Binned Means + SEM + Scatter overlay
    # Scatter points with jitter
    rng = np.random.RandomState(2026)
    bin_map = {lbl: i for i, lbl in enumerate(bin_labels)}
    x_scatter = [bin_map[b] + rng.uniform(-0.18, 0.18) for b in df["time_bin"]]

    ax1.scatter(
        x_scatter,
        df["scorer_v2_pq"],
        color="#1f77b4",
        alpha=0.18,
        s=22,
        edgecolors="none",
        label="Validation Images (N=707)",
    )

    # Error bars and lines
    ax1.errorbar(
        x_indices - 0.08,
        sc_means,
        yerr=sc_sems,
        fmt="o-",
        color="#1f77b4",
        linewidth=2.4,
        markersize=8,
        capsize=5,
        capthick=1.5,
        label=f"Instance Scorer v2 (Mean $\\pm$ SEM, Overall={df['scorer_v2_pq'].mean():.4f})",
        zorder=5,
    )
    ax1.errorbar(
        x_indices + 0.08,
        tr_means,
        yerr=tr_sems,
        fmt="s--",
        color="#d62728",
        linewidth=2.0,
        markersize=7,
        capsize=5,
        capthick=1.5,
        label=f"Threshold Rule (Mean $\\pm$ SEM, Overall={df['threshold_rule_pq'].mean():.4f})",
        zorder=5,
    )

    # Reference overall line
    ax1.axhline(
        df["scorer_v2_pq"].mean(),
        color="#1f77b4",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )

    ax1.set_ylabel("Per-Image Panoptic Quality (PQ)", fontsize=12, fontweight="bold")
    ax1.set_title("5-Fold OOF Validation: Per-Image PQ vs Time Gap to Nearest Training Observation", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(bin_labels, fontsize=11)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper right", frameon=True, fontsize=10, facecolor="#f8f9fa")

    # Annotate mean values above points
    for i, (m, sem) in enumerate(zip(sc_means, sc_sems)):
        ax1.annotate(
            f"{m:.3f}",
            (i - 0.08, m + sem + 0.035),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1f77b4",
        )

    # Bottom Panel: Sample Count Histogram
    bars = ax2.bar(x_indices, counts, color="#4a5568", alpha=0.75, width=0.55, edgecolor="#2d3748")
    ax2.set_xlabel("Time Gap to Nearest Training Observation in Split", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Image Count", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(bin_labels, fontsize=11)
    ax2.set_ylim(0, max(counts) * 1.25)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")

    for bar, count in zip(bars, counts):
        pct = 100.0 * count / len(df)
        ax2.annotate(
            f"N={count}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 5),
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    plt.tight_layout()
    out_plot = REPO_ROOT / "audit/per_image_pq_vs_time_gap.png"
    plt.savefig(out_plot, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved publication figure to {out_plot}")


if __name__ == "__main__":
    main()
