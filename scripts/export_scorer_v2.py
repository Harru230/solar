#!/usr/bin/env python3
"""
scripts/export_scorer_v2.py - Trains and exports the 5 cross-fitted LightGBM Scorer v2 models.

This reproduces the exact 5-fold cross-fitted LightGBM regressor ensemble used for Submission A:
- 19 view-invariant candidate & cluster features (SCORER_V2_VIEW_INVARIANT_FEATURES)
- Trained on 12-view 3-family fused OOF detections across all 707 training observations (seed=2026)
- Saved to work/checkpoints/scorer_v2_models.joblib
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

try:
    import joblib
    import lightgbm as lgb
except ImportError:
    joblib = None
    lgb = None
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import find_dataset_paths
from scripts.multifamily_pipeline import (
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    extract_scorer_v2_dataset,
    load_fused_oof_pool,
)


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export Scorer v2 LightGBM models")
    parser.add_argument("--output", type=str, default="work/checkpoints/scorer_v2_models.joblib", help="Output joblib file")
    parser.add_argument("--upstream_work", type=str, default=None, help="Path to upstream solar/work directory")
    args = parser.parse_args()
    if joblib is None or lgb is None:
        raise ImportError("joblib and lightgbm are required to run scripts/export_scorer_v2.py")

    t0 = time.time()
    upstream_dir = Path(args.upstream_work) if args.upstream_work else (REPO_ROOT.parent / "solar" / "work")
    paths = find_dataset_paths(str(REPO_ROOT))

    folds_csv = upstream_dir / "folds.csv" if not paths.get("folds_csv") else Path(paths["folds_csv"])
    gt_csv = upstream_dir / "gt_all.csv" if not paths.get("gt_all_csv") else Path(paths["gt_all_csv"])
    train_json = paths.get("train_json")
    if not train_json:
        train_json = str(REPO_ROOT.parent / "solar/data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    print(f"Loading datasets...")
    folds_df = pd.read_csv(folds_csv)
    gt_df = pd.read_csv(gt_csv)

    oof_clahe = str(upstream_dir / "oof")
    oof_a5 = str(upstream_dir / "oof_a5")
    oof_dice = str(upstream_dir / "oof_dice")

    specs_12v = [
        ("clahe", oof_clahe, 4, 0.0),
        ("a5", oof_a5, 4, 0.0),
        ("dice", oof_dice, 4, 0.0),
    ]

    print(f"Loading 12-view 3-family OOF candidate pool across 5 folds...")
    pool_12v = load_fused_oof_pool(folds_df, specs_12v)
    print(f"Loaded {len(pool_12v)} observations in {time.time() - t0:.1f}s.")

    print(f"Extracting 19 view-invariant Scorer v2 features (tau_mask=0.525)...")
    feat_0525, pool_0525 = extract_scorer_v2_dataset(
        pool_12v,
        folds_df,
        gt_df,
        train_json,
        num_views_total=12,
        tau_mask=0.525,
        use_tight_bbox=True,
    )
    print(f"Extracted {len(feat_0525)} candidates. Feature count: {len(SCORER_V2_VIEW_INVARIANT_FEATURES)}")

    print(f"Cross-fitting 5 LightGBM Regressors (seed=2026)...")
    models = []
    seed = 2026
    for val_f in range(5):
        tr_m = (feat_0525["fold"] != val_f).values
        va_m = (feat_0525["fold"] == val_f).values

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
        X_tr = feat_0525.loc[tr_m, SCORER_V2_VIEW_INVARIANT_FEATURES].values
        y_tr = feat_0525.loc[tr_m, "target_y"].values
        model.fit(X_tr, y_tr)
        models.append(model)
        r2_val = model.score(feat_0525.loc[va_m, SCORER_V2_VIEW_INVARIANT_FEATURES].values, feat_0525.loc[va_m, "target_y"].values)
        print(f"  Fold {val_f}: fitted {len(X_tr)} samples, val R^2 = {r2_val:.4f}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, out_path)
    file_size_kb = out_path.stat().st_size / 1024
    chksum = sha256_file(out_path)

    print(f"\n[SUCCESS] Saved 5 Scorer v2 models to: {out_path}")
    print(f"Size:   {file_size_kb:.1f} KB")
    print(f"SHA256: {chksum}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
