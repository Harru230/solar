#!/usr/bin/env python3
"""
scripts/rebuild_time_gap_distribution.py - Generates time_gap_distribution.png with log-spaced bins.
"""
import argparse
import glob
import os
import json
import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    argparse.ArgumentParser(
        description="Generate audit/time_gap_distribution.png from fold timestamps."
    ).parse_args()
    upstream_dir = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar")))
    folds_path = REPO_ROOT / "work/folds.csv"
    if not folds_path.exists():
        folds_path = upstream_dir / "work/folds.csv"
    folds = pd.read_csv(folds_path)
    train_stems = folds["observation"].tolist()
    train_dts = {o: datetime.datetime.strptime(o[:14], "%Y%m%d%H%M%S") for o in train_stems}
    fold_map = dict(zip(folds["observation"], folds["fold"]))

    data_dir = os.environ.get("SOLAR_DATA_DIR")
    search_dirs = [Path(data_dir)] if data_dir else []
    search_dirs.extend([
        REPO_ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        upstream_dir / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    ])
    test_files = []
    for d in search_dirs:
        if d.is_dir():
            test_files = glob.glob(str(d / "*.jp*g"))
            if test_files:
                break
    test_stems = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
    test_dts = {o: datetime.datetime.strptime(o[:14], "%Y%m%d%H%M%S") for o in test_stems}

    val_gaps_fold_split = []
    val_gaps_all = []
    all_train_dts = list(train_dts.values())

    for v in train_stems:
        f = fold_map[v]
        other_train = [train_dts[u] for u in train_stems if fold_map[u] != f]
        min_diff = min(abs((train_dts[v] - dt).total_seconds()) for dt in other_train) / 3600.0
        val_gaps_fold_split.append(min_diff)

        other_all = [train_dts[u] for u in train_stems if u != v]
        min_diff_all = min(abs((train_dts[v] - dt).total_seconds()) for dt in other_all) / 3600.0
        val_gaps_all.append(min_diff_all)

    test_gaps = []
    for t in test_stems:
        min_diff = min(abs((test_dts[t] - dt).total_seconds()) for dt in all_train_dts) / 3600.0
        test_gaps.append(min_diff)

    val_gaps_fold_split = np.array(val_gaps_fold_split)
    test_gaps = np.array(test_gaps)
    val_gaps_all = np.array(val_gaps_all)

    # 40 Log-spaced bins between 0.01 hours (~36 sec) and 2000 hours (~83 days)
    min_val = 0.01
    max_val = 2000.0
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 42)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#ffffff")
    
    # Histograms with density / log-spaced bins
    ax.hist(
        val_gaps_fold_split,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color="#1f77b4",
        label=f"Val in 5-Fold Split (N=707, Median={np.median(val_gaps_fold_split):.1f}h)",
        edgecolor="#1f77b4",
        linewidth=1.5,
    )
    ax.hist(
        test_gaps,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="#d62728",
        label=f"Test vs All Train (N=180, Median={np.median(test_gaps):.1f}h)",
    )
    ax.hist(
        val_gaps_all,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="#2ca02c",
        label=f"Train Internal Baseline (N=707, Median={np.median(val_gaps_all):.1f}h)",
    )

    # Median lines
    ax.axvline(np.median(val_gaps_fold_split), color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.axvline(np.median(test_gaps), color="#d62728", linestyle=":", linewidth=1.5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlim(min_val, max_val)
    
    # Custom ticks on log scale
    ticks = [0.01, 0.1, 1, 6, 24, 72, 168, 720, 2000]
    tick_labels = ["36s", "6m", "1h", "6h", "24h (1d)", "3d", "1w", "1mo", "83d"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=10)

    ax.set_xlabel("Time Gap to Nearest Training Observation (Log-Spaced Bins)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Temporal Separation to Nearest Training Observation\n"
        "Comparison of Validation (in 5-Fold Split) vs Hidden Test Set",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.3, color="#cccccc")
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#cfd8dc", fontsize=10, loc="upper left")

    out_path = REPO_ROOT / "audit/time_gap_distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved rebuilt time_gap_distribution.png to: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
