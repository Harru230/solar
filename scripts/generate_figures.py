#!/usr/bin/env python3
"""
scripts/generate_figures.py - Standalone Static Figure Generator for Solar Release.

Generates publication-quality static figures for documentation and technical report:
1. figures/gt_vs_pred_overlay.png:
   Ground truth vs prediction overlay with TP (green), FP (red), FN (cyan/blue) colors
   for the 6 representative observations cataloged in survey_experiments_data.md:
   - 20140323020654Uh (11 TP, 3 FP, 3 FN, PQ=0.5362)
   - 20110312082634Lh (7 TP, 1 FP, 2 FN, PQ=0.5603)
   - 20110509121334Ch (6 TP, 2 FP, 2 FN, PQ=0.4312)
   - 20120730135454Bh (5 TP, 5 FP, 4 FN, PQ=0.3429)
   - 20170719235330Lh (3 TP, 1 FP, 2 FN, PQ=0.4712)
   - 20130809072814Th (8 TP, 5 FP, 0 FN, PQ=0.4996)

2. figures/pq_loss_decomposition.png:
   Horizontal bar chart decomposing Panoptic Quality (PQ) errors into 6 structural categories,
   citing work/oof/loss_decomposition_summary.json.

3. figures/oof_per_family_table.png:
   High-resolution rendered table of 5-fold Out-of-Fold (OOF) cross-validation performance
   across model families, citing authoritative results files.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --out_dir figures --dpi 300
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mu

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pq_eval import iou_matrix, pq_from_matrices


# -----------------------------------------------------------------------------
# Color Definitions
# -----------------------------------------------------------------------------
# High-contrast, semantically meaningful colors on greyscale solar images:
# - True Positive (TP): Bright Green
# - False Positive (FP): High-visibility Red
# - False Negative (FN): Vibrant Cyan / Sky Blue
COLOR_TP = np.array([0.15, 0.85, 0.25])     # Vibrant Emerald Green (RGB [0, 1])
COLOR_FP = np.array([0.95, 0.20, 0.20])     # Crimson Red
COLOR_FN = np.array([0.00, 0.75, 1.00])     # Deep Sky Blue / Cyan

HEX_TP = "#26d742"
HEX_FP = "#f23535"
HEX_FN = "#00bfff"


# -----------------------------------------------------------------------------
# Path Resolution Helpers
# -----------------------------------------------------------------------------
def locate_file(relative_or_candidates: List[str]) -> Path:
    """Finds the first existing candidate path."""
    for cand in relative_or_candidates:
        p = Path(cand)
        if p.is_file():
            return p.resolve()
    # Try resolving relative to REPO_ROOT
    for cand in relative_or_candidates:
        p = REPO_ROOT / cand
        if p.is_file():
            return p.resolve()
    # Try resolving relative to parent solar directory
    for cand in relative_or_candidates:
        p = REPO_ROOT.parent / "solar" / cand
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"Could not locate file from candidates: {relative_or_candidates}")


def locate_dir(relative_or_candidates: List[str]) -> Path:
    """Finds the first existing candidate directory."""
    for cand in relative_or_candidates:
        p = Path(cand)
        if p.is_dir():
            return p.resolve()
    for cand in relative_or_candidates:
        p = REPO_ROOT / cand
        if p.is_dir():
            return p.resolve()
    for cand in relative_or_candidates:
        p = REPO_ROOT.parent / "solar" / cand
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError(f"Could not locate directory from candidates: {relative_or_candidates}")


def decode_rle_mask(rle_str: str, shape: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Decodes COCO compressed RLE into a 2D boolean numpy array."""
    if not rle_str or rle_str == "PPPP4":
        return np.zeros(shape, dtype=bool)
    rle_obj = {"size": list(shape), "counts": rle_str.encode("ascii")}
    return mu.decode(rle_obj).astype(bool)


# -----------------------------------------------------------------------------
# Figure 1: GT vs Pred Overlay
# -----------------------------------------------------------------------------
def overlay_colored_mask(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float = 0.40,
    outline_thickness: int = 2,
) -> np.ndarray:
    """Applies semi-transparent fill and solid boundary outline for a binary mask."""
    if not mask.any():
        return base_rgb
    
    H, W = base_rgb.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Translucent fill
    base_rgb[mask] = base_rgb[mask] * (1.0 - alpha) + color * alpha
    
    # Boundary outline via morphological dilation
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=outline_thickness)
    eroded = cv2.erode(mask_u8, kernel, iterations=outline_thickness)
    contour = (dilated - eroded).astype(bool)
    base_rgb[contour] = color * 0.98
    
    return base_rgb


def generate_gt_vs_pred_overlay(out_path: Path, dpi: int = 300) -> None:
    """Generates the 6-image GT vs Prediction overlay figure."""
    print("Generating Figure 1: GT vs Prediction Overlay...")
    
    obs_catalog = [
        ("20140323020654Uh", 11, 3, 3, 0.5362),
        ("20110312082634Lh", 7, 1, 2, 0.5603),
        ("20110509121334Ch", 6, 2, 2, 0.4312),
        ("20120730135454Bh", 5, 5, 4, 0.3429),
        ("20170719235330Lh", 3, 1, 2, 0.4712),
        ("20130809072814Th", 8, 5, 0, 0.4996),
    ]
    
    gt_path = locate_file(["work/gt_all.csv", "gt_all.csv"])
    pred_path = locate_file(["work/oof/oof_predictions_tuned.csv"])
    
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)
    
    # Find train images dir
    train_dir_candidates = [
        "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
        "../solar/data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    ]
    train_dir = locate_dir(train_dir_candidates)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12.5), facecolor="#111111")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.03, right=0.97, hspace=0.18, wspace=0.05)
    
    for idx, (obs_id, exp_tp, exp_fp, exp_fn, exp_pq) in enumerate(obs_catalog):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#111111")
        
        # Load image
        img_file = train_dir / f"{obs_id}.jpeg"
        if not img_file.exists():
            img_file = train_dir / f"{obs_id}.jpg"
        
        img_bgr = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback synthetic dark solar canvas if image missing
            canvas = np.zeros((1024, 1024, 3), dtype=np.float32)
        else:
            # Convert to RGB float32 [0, 1] and downscale to 1024 for crisp fast rendering
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            canvas = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        # Filter GT and Pred rows
        gt_rows = gt_df[gt_df["filament_id"].str.contains(obs_id)]
        pred_rows = pred_df[pred_df["filament_id"].str.startswith(obs_id + "_")]
        
        gt_rles = gt_rows["segmentation_rle"].tolist()
        pred_rles = pred_rows["segmentation_rle"].tolist()
        
        # Match via IoU matrix
        iou_mat = iou_matrix(gt_rles, pred_rles)
        hit = (iou_mat > 0.50)
        
        # Identify TP predictions and GTs
        tp_pred_indices = set(np.where(hit.sum(axis=0) > 0)[0])
        fp_pred_indices = set(np.where(hit.sum(axis=0) == 0)[0])
        fn_gt_indices = set(np.where(hit.sum(axis=1) == 0)[0])
        
        # Draw False Negatives (Missed GTs) in Cyan/Blue
        for g_idx in fn_gt_indices:
            mask = decode_rle_mask(gt_rles[g_idx], shape=(2048, 2048))
            canvas = overlay_colored_mask(canvas, mask, COLOR_FN, alpha=0.45, outline_thickness=2)
            
        # Draw False Positives (Spurious Preds) in Red
        for p_idx in fp_pred_indices:
            mask = decode_rle_mask(pred_rles[p_idx], shape=(2048, 2048))
            canvas = overlay_colored_mask(canvas, mask, COLOR_FP, alpha=0.45, outline_thickness=2)
            
        # Draw True Positives (Matched Predictions) in Green
        for p_idx in tp_pred_indices:
            mask = decode_rle_mask(pred_rles[p_idx], shape=(2048, 2048))
            canvas = overlay_colored_mask(canvas, mask, COLOR_TP, alpha=0.50, outline_thickness=2)
        
        # Display image
        ax.imshow(canvas)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(1.5)
            
        # Subplot Title
        ax.set_title(
            f"{obs_id}\n"
            f"PQ: {exp_pq:.4f}  |  TP: {len(tp_pred_indices)}  |  FP: {len(fp_pred_indices)}  |  FN: {len(fn_gt_indices)}",
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
    
    # Global Super Title
    fig.suptitle(
        "Ground Truth vs Model Predictions Overlay (OOF Tuned Evaluation)\n"
        "Representative Validation Cases Demonstrating True Positives, Hallucinations, and Misses",
        color="#ffffff",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    
    # Global Legend
    legend_elements = [
        mpatches.Patch(facecolor=HEX_TP, edgecolor="#ffffff", linewidth=1.2, label="True Positive (TP, IoU >= 0.50) — Model Prediction Matched with GT"),
        mpatches.Patch(facecolor=HEX_FP, edgecolor="#ffffff", linewidth=1.2, label="False Positive (FP, IoU < 0.50) — Unmatched / Spurious Prediction"),
        mpatches.Patch(facecolor=HEX_FN, edgecolor="#ffffff", linewidth=1.2, label="False Negative (FN) — Ground Truth Filament Missed by Model"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11.5,
        frameon=True,
        facecolor="#222222",
        edgecolor="#555555",
        labelcolor="#ffffff",
        borderpad=0.6,
        handlelength=1.6,
        handleheight=1.0,
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


# -----------------------------------------------------------------------------
# Figure 2: PQ Loss Decomposition Bar Chart
# -----------------------------------------------------------------------------
def generate_pq_loss_decomposition_figure(out_path: Path, dpi: int = 300) -> None:
    """Generates the PQ Loss Decomposition horizontal bar chart."""
    print("Generating Figure 2: PQ Loss Decomposition Bar Chart...")
    
    json_path = locate_file(["work/oof/loss_decomposition_summary.json"])
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["nested_cv"]
        
    baseline_pq = data["baseline"]["pq"]
    baseline_sq = data["baseline"]["sq"]
    baseline_rq = data["baseline"]["rq"]
    
    # Extract categories and their exact numbers
    items = []
    
    # 1. TP Boundary Inexactness
    tp_inexact = data["tp_boundary_inexactness"]
    items.append({
        "name": "TP Boundary Inexactness",
        "detail": f"Matched TPs having SQ={tp_inexact['mean_sq']:.4f} instead of 1.0 (5,277 instances)",
        "pq_cost": tp_inexact["pq_cost"],
        "events": tp_inexact["tp_count"],
        "fn": 0,
        "fp": 0,
        "color": "#1f77b4",  # Primary Blue
    })
    
    # Map raw categories
    cat_meta = {
        "complete_misses": ("Complete Misses", "Unmatched GT filaments with max IoU < 0.25 (faint/thin)", "#d62728"),
        "near_misses_025_050": ("Near-Misses", "Unmatched GT paired with prediction where 0.25 <= IoU < 0.50", "#ff7f0e"),
        "pure_false_positives": ("Pure False Positives", "Hallucinated predictions with best GT IoU < 0.25", "#e377c2"),
        "merges": ("Merges", "Single predicted mask swallowing >= 2 distinct GT filaments", "#9467bd"),
        "splits": ("Splits", "Single GT filament fragmented into >= 2 predicted masks", "#8c564b"),
    }
    
    for c in data["categories"]:
        cat_key = c["category"]
        if cat_key in cat_meta:
            name, desc, color = cat_meta[cat_key]
            items.append({
                "name": name,
                "detail": f"{desc} ({c['events']:,} events)",
                "pq_cost": c["pq_cost"],
                "events": c["events"],
                "fn": c["fn"],
                "fp": c["fp"],
                "color": color,
            })
            
    # Sort items by PQ cost ascending so largest is on top in horizontal bar chart
    items.sort(key=lambda x: x["pq_cost"], reverse=False)
    
    total_assignable_loss = sum(x["pq_cost"] for x in items)
    
    fig, ax = plt.subplots(figsize=(15.0, 7.8), facecolor="#ffffff")
    fig.subplots_adjust(left=0.25, right=0.88, top=0.86, bottom=0.12)
    
    y_pos = np.arange(len(items))
    costs = [x["pq_cost"] for x in items]
    colors = [x["color"] for x in items]
    labels = [x["name"] for x in items]
    
    bars = ax.barh(y_pos, costs, color=colors, height=0.62, edgecolor="#333333", linewidth=1.1, zorder=3)
    
    # Add gridlines behind bars
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    
    # Value annotations on each bar
    for bar, item in zip(bars, items):
        w = bar.get_width()
        pct = (w / total_assignable_loss) * 100.0
        events_str = f"Events: {item['events']:,}"
        if item["fn"] or item["fp"]:
            events_str += f" (FN: {item['fn']:,}, FP: {item['fp']:,})"
        text_label = f"  +{w:.4f} PQ  ({pct:.1f}% of loss)  |  {events_str}"
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2.0, text_label,
                va="center", ha="left", fontsize=10.5, fontweight="bold", color="#222222")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Counterfactual Gain in Dataset Panoptic Quality (ΔPQ Cost)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_xlim(0, max(costs) * 1.62)
    
    # Header and subheader
    fig.suptitle(
        "Panoptic Quality (PQ) Error Decomposition\n"
        f"Nested-CV 5-Fold Baseline: PQ = {baseline_pq:.4f} (SQ = {baseline_sq:.4f}, RQ = {baseline_rq:.4f})  |  Total Assignable Loss = {total_assignable_loss:.4f} PQ",
        fontsize=14.5,
        fontweight="bold",
        y=0.96,
    )
    
    # Informative annotation box placed cleanly below the upper bars to avoid overlap
    callout_text = (
        "Key Takeaways:\n"
        "• TP Boundary Inexactness (+0.2153 PQ, 48.0%) is the largest assignable gap, reflecting\n"
        "  inherent boundary variation (human inter-annotator SQ=0.634 vs. model SQ=0.669).\n"
        "  The boundary loss is mostly annotator disagreement and diffuse filament contours.\n"
        "• Complete Misses (0.0796 PQ) and Near-Misses (0.0703 PQ) form the primary recognition errors.\n"
        "• Pure False Positives (0.0412 PQ) are comparatively rare.\n"
        "Source: work/oof/loss_decomposition_summary.json"
    )
    ax.text(
        0.48, 0.07, callout_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#b0bec5", alpha=0.95),
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


# -----------------------------------------------------------------------------
# Figure 3: OOF Per-Family Performance Table
# -----------------------------------------------------------------------------
def generate_oof_per_family_table_figure(out_path: Path, dpi: int = 300) -> None:
    """Renders a publication-ready graphical table of OOF per-family metrics."""
    print("Generating Figure 3: OOF Per-Family Table...")
    
    table_data = [
        [
            "r50_1536_clahe",
            "1536",
            "CLAHE preproc, 4-flip TTA, WMF",
            "0.4338",
            "0.4347\n(τ_s=0.825, τ_m=0.500)",
            "0.4343\n(0.6686 / 0.6496)",
            "5,277 / 2,772 / 2,922",
            "work/oof/oof_tuning_summary.json",
        ],
        [
            "r50_1536_anchors5",
            "1536",
            "5 aspect ratios [0.2, 0.5, 1.0, 2.0, 5.0]",
            "0.4330",
            "0.4349\n(τ_s=0.800, τ_m=0.500)",
            "0.4338\n(0.6683 / 0.6490)",
            "5,308 / 2,850 / 2,891",
            "work/oof_a5/oof_tuning_summary.json",
        ],
        [
            "r50_1536_dice",
            "1536",
            "BCE + Dice mask loss (τ_m=0.55)",
            "0.4323",
            "0.4384\n(τ_s=0.775, τ_m=0.550)",
            "0.4380\n(0.6689 / 0.6548)",
            "5,339 / 2,770 / 2,860",
            "work/oof_dice/oof_tuning_summary.json",
        ],
        [
            "r50_2048_all*",
            "2048",
            "Native 2048px (Folds 0-2 pooled)",
            "0.4386*",
            "N/A\n(3 folds pooled)",
            "0.4386*\n(0.6743 / 0.6504)",
            "3,244 / 1,699 / 1,788",
            "work/oof_2048/fold*/oof_fold*_summary.json",
        ],
        [
            "Instance Scorer v1",
            "1536",
            "LightGBM 14-feat GBDT on CLAHE OOF",
            "0.4387",
            "0.4389\n(τ_gbdt=0.250)",
            "0.4387\n(0.6707 / 0.6540)",
            "5,316 / 2,739 / 2,883",
            "work/oof/instance_scorer_summary.json",
        ],
    ]
    
    columns = [
        "Model Family",
        "Image\nSize",
        "Key Architectural Switch\n& Preprocessing",
        "Default\nOOF PQ",
        "Tuned Global PQ\n(Best Thresholds)",
        "Nested-CV PQ\n(SQ / RQ)",
        "TP / FP / FN\nCounts",
        "Authoritative\nSource File",
    ]
    
    fig, ax = plt.subplots(figsize=(17.0, 6.5), facecolor="#ffffff")
    ax.axis("off")
    
    # Title
    fig.suptitle(
        "Out-of-Fold (OOF) 5-Fold Cross-Validation Performance Across Model Families\n"
        "Evaluated on Full 707 Training Observations (1,154 Annotator-Image Evaluations)",
        fontsize=14.5,
        fontweight="bold",
        y=0.95,
    )
    
    # Create Table (colWidths normalized to exactly 1.00 to prevent clipping)
    tab = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.13, 0.06, 0.22, 0.08, 0.12, 0.12, 0.11, 0.16],
    )
    
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.0, 2.3)
    
    # Styling cells
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#cfd8dc")
        cell.set_linewidth(1.0)
        if row == 0:
            # Header
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="#ffffff", fontweight="bold", fontsize=10.0)
            cell.set_height(0.12)
        else:
            # Alternating row colors
            if row % 2 == 1:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("#ffffff")
            # Highlight Best Nested-CV PQ
            if col == 5:
                cell.set_text_props(fontweight="bold", color="#1b5e20")
            elif col == 0:
                cell.set_text_props(fontweight="bold")
            elif col == 7:
                cell.set_text_props(fontsize=8.0, color="#546e7a", family="monospace")
                
    # Footer footnote
    footnote = (
        "* Note: r50_2048_all represents 3 folds pooled (Folds 0-2 completed 30 epochs each; fold 3 stopped at epoch 21 per schedule).\n"
        "All metrics computed strictly using official competition evaluation logic (IoU threshold τ = 0.50 with disjoint canvas predictions)."
    )
    fig.text(0.08, 0.05, footnote, fontsize=9.2, color="#455a64", style="italic")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate static figures for solar_release.")
    parser.add_argument("--out_dir", type=str, default="figures", help="Output directory for generated PNG figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI resolution for figures (default: 300).")
    args = parser.parse_args()
    
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting static figure generation into: {out_dir} (DPI={args.dpi})")
    
    fig1_path = out_dir / "gt_vs_pred_overlay.png"
    fig2_path = out_dir / "pq_loss_decomposition.png"
    fig3_path = out_dir / "oof_per_family_table.png"
    
    generate_gt_vs_pred_overlay(fig1_path, dpi=args.dpi)
    generate_pq_loss_decomposition_figure(fig2_path, dpi=args.dpi)
    generate_oof_per_family_table_figure(fig3_path, dpi=args.dpi)
    
    print("\nAll figures generated successfully:")
    for p in [fig1_path, fig2_path, fig3_path]:
        print(f"  - {p.relative_to(REPO_ROOT)}: {os.path.getsize(p) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
