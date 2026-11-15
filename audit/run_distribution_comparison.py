#!/usr/bin/env python3
"""
audit/run_distribution_comparison.py - Compares predictions per image, area
distributions, and predicted quality distributions between 5-fold OOF and the
Test Ensemble for both Submission A (Scorer v2) and Submission B (Threshold Rule).

Outputs:
  - Plot: audit/distribution_comparison_oof_vs_test.png
  - JSON: audit/distribution_comparison.json
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLAR_ROOT = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))).resolve()
sys.path.insert(0, str(REPO_ROOT))

from scripts.regenerate_submission import (
    load_ensemble_config,
    weighted_mask_fusion_tagged,
    apply_scorer_v2_decision,
    apply_threshold_decision,
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    FRAME_SIZE,
)
try:
    import joblib
except ImportError:
    joblib = None


def get_areas_from_rles(rle_strings: list) -> np.ndarray:
    """Computes area in pixels from a list of RLE strings in chunks to prevent uint8 pycocotools overflow."""
    valid_rles = [s for s in rle_strings if pd.notna(s) and str(s).strip() != ""]
    if not valid_rles:
        return np.array([], dtype=np.int64)
    coco_rles = [{"size": [FRAME_SIZE, FRAME_SIZE], "counts": str(s).encode("ascii")} for s in valid_rles]
    areas = []
    chunk_size = 100
    for i in range(0, len(coco_rles), chunk_size):
        chunk = coco_rles[i : i + chunk_size]
        areas.extend(mu.area(chunk))
    return np.asarray(areas, dtype=np.int64)


def get_counts_per_image(df: pd.DataFrame, expected_images: list) -> dict:
    """Counts valid (non-empty) filament predictions per image."""
    valid_df = df[df["segmentation_rle"].notna() & (df["segmentation_rle"] != "")].copy()
    valid_df["img"] = valid_df["filament_id"].apply(lambda s: s.split("_")[0])
    counts_map = valid_df.groupby("img").size().to_dict()
    return {img: counts_map.get(img, 0) for img in expected_images}


def main():
    argparse.ArgumentParser(
        description="Compare OOF vs Test prediction count, area, and quality distributions."
    ).parse_args()
    os.chdir(str(SOLAR_ROOT))
    t0 = time.time()
    print("Loading OOF and Test CSV predictions...")

    # Paths
    oof_folds_csv = SOLAR_ROOT / "work/folds.csv"
    folds_df = pd.read_csv(oof_folds_csv)
    oof_obs = folds_df["observation"].tolist()

    oof_A_path = SOLAR_ROOT / "work/oof/oof_predictions_instance_scorer.csv"
    oof_B_path = SOLAR_ROOT / "work/oof/oof_predictions_nested_cv.csv"
    test_A_path = SOLAR_ROOT / "work/oof/submission_A_scorer_v2_multifam_8model.csv"
    test_B_path = SOLAR_ROOT / "work/oof/submission_B_threshold_multifam_8model.csv"

    df_oof_A = pd.read_csv(oof_A_path)
    df_oof_B = pd.read_csv(oof_B_path)
    df_test_A = pd.read_csv(test_A_path)
    df_test_B = pd.read_csv(test_B_path)

    test_images = sorted(list(set(df_test_A["filament_id"].apply(lambda s: s.split("_")[0]))))
    print(f"OOF images: {len(oof_obs)}, Test images: {len(test_images)}")

    # 1. Predictions per image
    print("Extracting predictions per image...")
    counts_oof_A = get_counts_per_image(df_oof_A, oof_obs)
    counts_oof_B = get_counts_per_image(df_oof_B, oof_obs)
    counts_test_A = get_counts_per_image(df_test_A, test_images)
    counts_test_B = get_counts_per_image(df_test_B, test_images)

    # 2. Area distributions
    print("Computing filament area distributions...")
    areas_oof_A = get_areas_from_rles(df_oof_A["segmentation_rle"].tolist())
    areas_oof_B = get_areas_from_rles(df_oof_B["segmentation_rle"].tolist())
    areas_test_A = get_areas_from_rles(df_test_A["segmentation_rle"].tolist())
    areas_test_B = get_areas_from_rles(df_test_B["segmentation_rle"].tolist())

    # 3. Predicted Quality Scores
    print("Extracting candidate quality scores for OOF and Test...")
    # For Test: extract scores using test_cache streaming
    cfg = load_ensemble_config(REPO_ROOT / "configs/final_ensemble.json")
    model_list = [(m["id"], m["family"], float(m.get("prob_shift", 0.0))) for m in cfg["models"]]
    scorer_models = joblib.load(SOLAR_ROOT / "work/checkpoints/scorer_v2_models.joblib")
    cache_dir = SOLAR_ROOT / "work/test_cache"

    test_scores_A = []
    test_scores_B = []

    def _process_test_stem(stem: str):
        combined = []
        for m_name, fam, p_shift in model_list:
            pt_path = os.path.join(cache_dir, m_name, f"{stem}.pt")
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            for c in data.get("candidates", []):
                pc = c["prob_crop"]
                if p_shift != 0.0:
                    pc = np.clip(pc.astype(np.float32) + p_shift, 0.0, 1.0)
                combined.append({"box": c["box"], "score": float(c["score"]), "prob_crop": pc, "family": fam})
        fused = weighted_mask_fusion_tagged(combined, num_views=cfg["total_views"], iou_thresh=0.40, family_views=cfg["family_views"])

        _, b_scores = apply_threshold_decision(fused, stem, tau_score=0.875, tau_mask=0.550, return_scores=True)
        _, a_scores = apply_scorer_v2_decision(fused, stem, scorer_models, tau_scorer=0.2440, tau_mask=0.525, return_scores=True)

        return a_scores, b_scores

    with ThreadPoolExecutor(max_workers=8) as executor:
        for a_sc, b_sc in executor.map(_process_test_stem, test_images):
            test_scores_A.extend(a_sc)
            test_scores_B.extend(b_sc)

    # For OOF: Extract scores from 3-family pool
    print("Extracting OOF candidate scores...")
    from scripts.multifamily_pipeline import load_fused_oof_pool, extract_scorer_v2_dataset
    specs_12v_raw = [
        ("clahe", "work/oof", 4, 0.0),
        ("a5", "work/oof_a5", 4, 0.0),
        ("dice", "work/oof_dice", 4, 0.0),
    ]
    pool_12v = load_fused_oof_pool(folds_df, specs_12v_raw)
    feat_12v, obs_pool_12v = extract_scorer_v2_dataset(
        pool_12v, folds_df, pd.read_csv(SOLAR_ROOT / "work/gt_all.csv"),
        SOLAR_ROOT / "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        num_views_total=12, tau_mask=0.525
    )

    # Cross-predict pred_y on OOF
    for val_f in range(5):
        tr_mask = (feat_12v["fold"] != val_f).values
        va_mask = (feat_12v["fold"] == val_f).values
        m = scorer_models[val_f]
        feat_12v.loc[va_mask, "pred_y"] = m.predict(feat_12v.loc[va_mask, SCORER_V2_VIEW_INVARIANT_FEATURES].values)

    oof_scores_A = []
    obs_pred_map = feat_12v.groupby("obs")["pred_y"].apply(list).to_dict()
    for obs, pred_list in obs_pred_map.items():
        cands = obs_pool_12v.get(obs, [])
        for c, py in zip(cands, pred_list):
            c["pred_y"] = float(py)
        cands_sorted = sorted(cands, key=lambda c: c["pred_y"], reverse=True)
        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        for c in cands_sorted:
            if c["pred_y"] <= 0.2440:
                break
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
            oof_scores_A.append(float(c["pred_y"]))

    oof_scores_B = []
    for obs, cands in pool_12v.items():
        _, b_sc = apply_threshold_decision(cands, obs, tau_score=0.875, tau_mask=0.550, return_scores=True)
        oof_scores_B.extend(b_sc)

    print(f"Extracted scores in {time.time() - t0:.1f}s.")
    print(f"  Submission A (Scorer v2): OOF N={len(oof_scores_A)}, Test N={len(test_scores_A)}")
    print(f"  Submission B (Threshold): OOF N={len(oof_scores_B)}, Test N={len(test_scores_B)}")

    # Compute Summary Statistics
    stats = {
        "submission_A_scorer_v2": {
            "predictions_per_image": {
                "oof": {
                    "count": len(oof_obs),
                    "mean": float(np.mean(list(counts_oof_A.values()))),
                    "median": float(np.median(list(counts_oof_A.values()))),
                    "std": float(np.std(list(counts_oof_A.values()))),
                    "zero_count": int(sum(1 for v in counts_oof_A.values() if v == 0)),
                },
                "test": {
                    "count": len(test_images),
                    "mean": float(np.mean(list(counts_test_A.values()))),
                    "median": float(np.median(list(counts_test_A.values()))),
                    "std": float(np.std(list(counts_test_A.values()))),
                    "zero_count": int(sum(1 for v in counts_test_A.values() if v == 0)),
                },
            },
            "area_pixels": {
                "oof": {
                    "count": len(areas_oof_A),
                    "median": float(np.median(areas_oof_A)),
                    "mean": float(np.mean(areas_oof_A)),
                    "p25": float(np.percentile(areas_oof_A, 25)),
                    "p75": float(np.percentile(areas_oof_A, 75)),
                },
                "test": {
                    "count": len(areas_test_A),
                    "median": float(np.median(areas_test_A)),
                    "mean": float(np.mean(areas_test_A)),
                    "p25": float(np.percentile(areas_test_A, 25)),
                    "p75": float(np.percentile(areas_test_A, 75)),
                },
            },
            "predicted_quality_pred_y": {
                "oof": {
                    "count": len(oof_scores_A),
                    "mean": float(np.mean(oof_scores_A)),
                    "median": float(np.median(oof_scores_A)),
                    "std": float(np.std(oof_scores_A)),
                },
                "test": {
                    "count": len(test_scores_A),
                    "mean": float(np.mean(test_scores_A)),
                    "median": float(np.median(test_scores_A)),
                    "std": float(np.std(test_scores_A)),
                },
            },
        },
        "submission_B_threshold": {
            "predictions_per_image": {
                "oof": {
                    "count": len(oof_obs),
                    "mean": float(np.mean(list(counts_oof_B.values()))),
                    "median": float(np.median(list(counts_oof_B.values()))),
                    "std": float(np.std(list(counts_oof_B.values()))),
                    "zero_count": int(sum(1 for v in counts_oof_B.values() if v == 0)),
                },
                "test": {
                    "count": len(test_images),
                    "mean": float(np.mean(list(counts_test_B.values()))),
                    "median": float(np.median(list(counts_test_B.values()))),
                    "std": float(np.std(list(counts_test_B.values()))),
                    "zero_count": int(sum(1 for v in counts_test_B.values() if v == 0)),
                },
            },
            "area_pixels": {
                "oof": {
                    "count": len(areas_oof_B),
                    "median": float(np.median(areas_oof_B)),
                    "mean": float(np.mean(areas_oof_B)),
                    "p25": float(np.percentile(areas_oof_B, 25)),
                    "p75": float(np.percentile(areas_oof_B, 75)),
                },
                "test": {
                    "count": len(areas_test_B),
                    "median": float(np.median(areas_test_B)),
                    "mean": float(np.mean(areas_test_B)),
                    "p25": float(np.percentile(areas_test_B, 25)),
                    "p75": float(np.percentile(areas_test_B, 75)),
                },
            },
            "detection_score_S": {
                "oof": {
                    "count": len(oof_scores_B),
                    "mean": float(np.mean(oof_scores_B)),
                    "median": float(np.median(oof_scores_B)),
                    "std": float(np.std(oof_scores_B)),
                },
                "test": {
                    "count": len(test_scores_B),
                    "mean": float(np.mean(test_scores_B)),
                    "median": float(np.median(test_scores_B)),
                    "std": float(np.std(test_scores_B)),
                },
            },
        },
    }

    out_json = REPO_ROOT / "audit/distribution_comparison.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {out_json}")

    # Generate 6-panel Publication Figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), facecolor="#ffffff")

    # Column 0: Submission A (Scorer v2)
    # Row 0: Predictions per image
    ax = axes[0, 0]
    bins_cnt = np.arange(0, 25, 1)
    ax.hist(list(counts_oof_A.values()), bins=bins_cnt, density=True, alpha=0.4, color="#1f77b4", label=f"OOF (Mean={stats['submission_A_scorer_v2']['predictions_per_image']['oof']['mean']:.2f})")
    ax.hist(list(counts_test_A.values()), bins=bins_cnt, density=True, histtype="step", linewidth=2.2, color="#d62728", label=f"Test (Mean={stats['submission_A_scorer_v2']['predictions_per_image']['test']['mean']:.2f})")
    ax.set_title("Submission A (Scorer v2): Predictions per Image", fontweight="bold")
    ax.set_xlabel("Filaments per Image")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Row 1: Area distribution (log10 scale)
    ax = axes[1, 0]
    log_oof_A = np.log10(np.clip(areas_oof_A, 10, None))
    log_test_A = np.log10(np.clip(areas_test_A, 10, None))
    bins_area = np.linspace(1.0, 5.5, 45)
    ax.hist(log_oof_A, bins=bins_area, density=True, alpha=0.4, color="#1f77b4", label=f"OOF (Median={stats['submission_A_scorer_v2']['area_pixels']['oof']['median']:.0f} px)")
    ax.hist(log_test_A, bins=bins_area, density=True, histtype="step", linewidth=2.2, color="#d62728", label=f"Test (Median={stats['submission_A_scorer_v2']['area_pixels']['test']['median']:.0f} px)")
    ax.set_title("Submission A (Scorer v2): Area Distribution (log10 px)", fontweight="bold")
    ax.set_xlabel("log10(Filament Area in Pixels)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Row 2: Predicted quality distribution
    ax = axes[2, 0]
    bins_q_A = np.linspace(0.244, 0.70, 35)
    ax.hist(oof_scores_A, bins=bins_q_A, density=True, alpha=0.4, color="#1f77b4", label=f"OOF (Mean={stats['submission_A_scorer_v2']['predicted_quality_pred_y']['oof']['mean']:.3f})")
    ax.hist(test_scores_A, bins=bins_q_A, density=True, histtype="step", linewidth=2.2, color="#d62728", label=f"Test (Mean={stats['submission_A_scorer_v2']['predicted_quality_pred_y']['test']['mean']:.3f})")
    ax.set_title("Submission A (Scorer v2): Quality Score (y_hat >= 0.244)", fontweight="bold")
    ax.set_xlabel("Continuous Utility Score (y_hat)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Column 1: Submission B (Threshold Rule)
    # Row 0: Predictions per image
    ax = axes[0, 1]
    ax.hist(list(counts_oof_B.values()), bins=bins_cnt, density=True, alpha=0.4, color="#2ca02c", label=f"OOF (Mean={stats['submission_B_threshold']['predictions_per_image']['oof']['mean']:.2f})")
    ax.hist(list(counts_test_B.values()), bins=bins_cnt, density=True, histtype="step", linewidth=2.2, color="#e377c2", label=f"Test (Mean={stats['submission_B_threshold']['predictions_per_image']['test']['mean']:.2f})")
    ax.set_title("Submission B (Threshold Rule): Predictions per Image", fontweight="bold")
    ax.set_xlabel("Filaments per Image")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Row 1: Area distribution (log10 scale)
    ax = axes[1, 1]
    log_oof_B = np.log10(np.clip(areas_oof_B, 10, None))
    log_test_B = np.log10(np.clip(areas_test_B, 10, None))
    ax.hist(log_oof_B, bins=bins_area, density=True, alpha=0.4, color="#2ca02c", label=f"OOF (Median={stats['submission_B_threshold']['area_pixels']['oof']['median']:.0f} px)")
    ax.hist(log_test_B, bins=bins_area, density=True, histtype="step", linewidth=2.2, color="#e377c2", label=f"Test (Median={stats['submission_B_threshold']['area_pixels']['test']['median']:.0f} px)")
    ax.set_title("Submission B (Threshold Rule): Area Distribution (log10 px)", fontweight="bold")
    ax.set_xlabel("log10(Filament Area in Pixels)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Row 2: Detection score distribution
    ax = axes[2, 1]
    bins_q_B = np.linspace(0.875, 1.0, 25)
    ax.hist(oof_scores_B, bins=bins_q_B, density=True, alpha=0.4, color="#2ca02c", label=f"OOF (Mean={stats['submission_B_threshold']['detection_score_S']['oof']['mean']:.3f})")
    ax.hist(test_scores_B, bins=bins_q_B, density=True, histtype="step", linewidth=2.2, color="#e377c2", label=f"Test (Mean={stats['submission_B_threshold']['detection_score_S']['test']['mean']:.3f})")
    ax.set_title("Submission B (Threshold Rule): Detection Score (S >= 0.875)", fontweight="bold")
    ax.set_xlabel("Detection Confidence Score S")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_plot = REPO_ROOT / "audit/distribution_comparison_oof_vs_test.png"
    plt.savefig(out_plot, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved distribution comparison figure to {out_plot}")
    print(f"Distribution comparison finished in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
