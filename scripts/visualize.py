#!/usr/bin/env python3
"""
scripts/visualize.py - Side-by-side visualisation of validation predictions vs GT.

For N randomly-chosen validation images from the held-out fold, saves one PNG per image
to work/vis/<run_name>/ with three panels:

  [original 1024px] | [predicted masks overlay] | [GT masks overlay (one colour per annotator)]

Usage:
  python scripts/visualize.py --run_name base_all_e30 --fold 0 --n 12
  python scripts/visualize.py --run_name exp_1536 --img_size 1536 --n 6

The script reads best_thresholds.json from work/runs/<run_name>/ (or --score_thresh /
--mask_thresh overrides) and uses the best checkpoint found in the same run dir.
If --checkpoint is not provided it looks for maskrcnn_fold0_best.pt in the run
checkpoints/ sub-directory.

pq_eval.py is NOT imported or modified.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    EMPTY_RLE,
    apply_mask_nms,
    extract_state_dict,
    find_dataset_paths,
    get_maskrcnn_model,
    get_run_dir,
    make_masks_disjoint,
    preprocess_solar_image,
    rle_encode_mask,
    set_seed,
    setup_logger,
)


# ── colour palettes ───────────────────────────────────────────────────────────

# Single colour for predicted masks (cyan-ish, high contrast on greyscale solar images)
PRED_COLOUR = np.array([0.0, 0.85, 1.0])      # RGB [0,1]

# Per-annotator colours (up to 12 distinct annotators per image in practice ≤3)
ANNOTATOR_COLOURS = [
    np.array([1.0,  0.27, 0.0 ]),   # orange-red
    np.array([0.12, 0.47, 0.71]),   # steel blue
    np.array([0.17, 0.63, 0.17]),   # green
    np.array([0.84, 0.15, 0.16]),   # red
    np.array([0.58, 0.40, 0.74]),   # purple
    np.array([0.55, 0.34, 0.29]),   # brown
    np.array([0.89, 0.47, 0.76]),   # pink
    np.array([0.50, 0.50, 0.50]),   # grey
    np.array([0.74, 0.74, 0.13]),   # olive-yellow
    np.array([0.09, 0.75, 0.81]),   # teal
    np.array([1.0,  0.60, 0.0 ]),   # amber
    np.array([0.65, 0.34, 0.16]),   # sienna
]


# ── RLE helpers ───────────────────────────────────────────────────────────────

def decode_rle(rle_str: str, size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Decode a compressed COCO RLE string to a binary (H, W) uint8 array."""
    rle_obj = {"size": list(size), "counts": rle_str.encode("ascii")}
    return mu.decode(rle_obj).astype(np.uint8)


# ── overlay helpers ───────────────────────────────────────────────────────────

def overlay_masks_on_image(
    base_img_rgb: np.ndarray,          # (H, W, 3) float32 [0,1]
    masks: List[np.ndarray],           # list of (H, W) uint8 binary masks (at base_img size)
    colours: List[np.ndarray],         # list of RGB [0,1] colours, one per mask
    alpha: float = 0.40,
    outline_only: bool = False,
    outline_thickness: int = 1,
) -> np.ndarray:
    """
    Overlay binary masks on a base RGB image.
    Returns a new (H, W, 3) float32 image.
    Draws solid translucent fill + a bright outline for each mask.
    """
    result = base_img_rgb.copy()
    H, W = result.shape[:2]

    for mask, colour in zip(masks, colours):
        if mask.shape != (H, W):
            # Resize mask to match base image if needed
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue

        if not outline_only:
            # Translucent fill
            result[mask_bool] = (
                result[mask_bool] * (1 - alpha) + colour * alpha
            )

        # Bright outline via dilate - erode contour trick
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=outline_thickness)
        eroded  = cv2.erode(mask, kernel, iterations=outline_thickness)
        outline_mask = (dilated - eroded).astype(bool)
        result[outline_mask] = colour * 0.95  # nearly full colour for outline

    return result


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference_single(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    score_thresh: float,
    mask_thresh: float,
    mask_nms_thresh: float,
    img_size: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run model inference on one image.
    Returns (bin_masks_2048, scores) — masks are already upsampled to 2048×2048.
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_preprocessed = preprocess_solar_image(img_gray, img_size)
    img_tensor = torch.from_numpy(img_preprocessed).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model([img_tensor])[0]

    scores_np = output["scores"].detach().cpu().numpy()
    masks_t   = output["masks"]  # [K, 1, img_size, img_size]

    keep = scores_np >= score_thresh
    if not np.any(keep):
        return [], []

    scores_kept = scores_np[keep]
    masks_kept  = masks_t[keep]

    # Upsample to 2048×2048
    masks_up = F.interpolate(
        masks_kept, size=(2048, 2048), mode="bilinear", align_corners=False
    )
    bin_masks_np = (masks_up.squeeze(1) >= mask_thresh).byte().cpu().numpy()  # [M, 2048, 2048]

    # RLE-encode for NMS
    rles, valid_scores = [], []
    for i in range(bin_masks_np.shape[0]):
        m = bin_masks_np[i]
        if m.sum() > 0:
            rles.append(rle_encode_mask(m))
            valid_scores.append(float(scores_kept[i]))

    if not rles:
        return [], []

    kept_idx  = apply_mask_nms(rles, valid_scores, iou_thresh=mask_nms_thresh)
    kept_masks  = [bin_masks_np[rles.index(rles[i])] for i in kept_idx]
    kept_scores = [valid_scores[i] for i in kept_idx]

    clean_masks, clean_scores = make_masks_disjoint(kept_masks, kept_scores)
    return clean_masks, clean_scores


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt_masks_by_annotator(
    gt_df: pd.DataFrame,
    obs_id: str,
) -> Dict[str, List[np.ndarray]]:
    """
    Returns {annotator_id: [mask_2048, ...]} for the given observation.
    Masks are decoded from RLE and stored as (2048, 2048) uint8 arrays.
    """
    # filament_id format: "<annotator>-<obs>_<n>"
    gt_for_obs = gt_df[
        gt_df["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1] == obs_id
    ]

    annotators: Dict[str, List[np.ndarray]] = {}
    for _, row in gt_for_obs.iterrows():
        fid = row["filament_id"]
        ann_obs = fid.split("_", 1)[0]          # e.g. "010101-20130425135954Bh"
        annotator = ann_obs.split("-", 1)[0]    # e.g. "010101"
        rle_str = row["segmentation_rle"]
        if rle_str == EMPTY_RLE:
            continue
        m = decode_rle(rle_str, size=(2048, 2048))
        if m.sum() > 0:
            annotators.setdefault(annotator, []).append(m)

    return annotators


# ── figure rendering ──────────────────────────────────────────────────────────

def render_triplet(
    img_path: str,
    obs_id: str,
    pred_masks: List[np.ndarray],      # list of (2048, 2048) uint8
    pred_scores: List[float],
    gt_by_annotator: Dict[str, List[np.ndarray]],
    out_path: str,
    vis_size: int = 1024,
    alpha: float = 0.40,
    dpi: int = 150,
) -> None:
    """
    Save a three-panel PNG:
      Panel 1: original image (greyscale, resized to vis_size)
      Panel 2: predicted masks overlaid on image (single colour)
      Panel 3: GT masks per annotator overlaid (each annotator a different colour)
    """
    # Load & resize base image
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_small = cv2.resize(img_gray, (vis_size, vis_size), interpolation=cv2.INTER_AREA)
    # Convert to 3-channel float for overlay
    img_rgb = np.stack([img_small] * 3, axis=-1).astype(np.float32) / 255.0  # (vis_size, vis_size, 3)

    # ── Panel 1: original ─────────────────────────────────────────────────────
    panel_orig = img_rgb.copy()

    # ── Panel 2: predictions ──────────────────────────────────────────────────
    # Downscale pred masks to vis_size for display (they're 2048 originals)
    pred_masks_small = []
    for m in pred_masks:
        ms = cv2.resize(m, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        pred_masks_small.append(ms)

    pred_colours = [PRED_COLOUR] * len(pred_masks_small)
    panel_pred = overlay_masks_on_image(img_rgb, pred_masks_small, pred_colours, alpha=alpha)

    # ── Panel 3: GT per annotator ─────────────────────────────────────────────
    ann_ids = sorted(gt_by_annotator.keys())
    gt_masks_small = []
    gt_colours     = []
    legend_patches = []

    for ann_idx, ann_id in enumerate(ann_ids):
        colour = ANNOTATOR_COLOURS[ann_idx % len(ANNOTATOR_COLOURS)]
        for m in gt_by_annotator[ann_id]:
            ms = cv2.resize(m, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
            gt_masks_small.append(ms)
            gt_colours.append(colour)
        # Build legend entry
        label = f"Ann {ann_id} ({len(gt_by_annotator[ann_id])} masks)"
        legend_patches.append(
            mpatches.Patch(facecolor=colour, edgecolor="white", linewidth=0.5, label=label)
        )

    panel_gt = overlay_masks_on_image(img_rgb, gt_masks_small, gt_colours, alpha=alpha)

    # ── Compose figure ────────────────────────────────────────────────────────
    n_pred = len(pred_masks)
    n_gt   = sum(len(v) for v in gt_by_annotator.values())
    n_ann  = len(ann_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=dpi)
    fig.patch.set_facecolor("#1a1a1a")

    titles = [
        f"Original ({vis_size}px)\n{obs_id}",
        f"Predictions: {n_pred} mask(s)",
        f"GT: {n_gt} mask(s), {n_ann} annotator(s)",
    ]
    panels = [panel_orig, panel_pred, panel_gt]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")
        # Thin white border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("white")
            spine.set_linewidth(0.5)

    # Prediction colour legend on panel 2
    pred_patch = mpatches.Patch(facecolor=PRED_COLOUR, edgecolor="white",
                                linewidth=0.5, label=f"Predictions ({n_pred})")
    axes[1].legend(handles=[pred_patch], loc="lower left", fontsize=7,
                   framealpha=0.6, facecolor="#222222", labelcolor="white")

    # Annotator legend on panel 3
    if legend_patches:
        axes[2].legend(handles=legend_patches, loc="lower left", fontsize=7,
                       framealpha=0.6, facecolor="#222222", labelcolor="white")

    fig.suptitle(f"Fold 0 — {obs_id}", color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise fold validation predictions vs GT annotations (3-panel PNG per image)"
    )
    parser.add_argument(
        "--run_name", type=str, default="default",
        help="Run name; reads best_thresholds.json and checkpoint from work/runs/<run_name>/. "
             "Saves PNGs to work/vis/<run_name>/. (default: 'default')"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint .pt file. If omitted, auto-discovered from "
             "work/runs/<run_name>/checkpoints/maskrcnn_fold<fold>_best.pt "
             "or the legacy work/checkpoints/ path."
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Held-out validation fold to visualise (default: 0)"
    )
    parser.add_argument(
        "--n", type=int, default=12,
        help="Number of validation images to visualise (default: 12)"
    )
    parser.add_argument(
        "--img_size", type=int, default=1024,
        help="Inference resolution matching the checkpoint (default: 1024)"
    )
    parser.add_argument(
        "--vis_size", type=int, default=1024,
        help="Size of each panel in the output PNG (default: 1024)"
    )
    parser.add_argument(
        "--score_thresh", type=float, default=None,
        help="Detection score threshold (default: from best_thresholds.json or 0.35)"
    )
    parser.add_argument(
        "--mask_thresh", type=float, default=None,
        help="Mask binarisation threshold (default: from best_thresholds.json or 0.5)"
    )
    parser.add_argument(
        "--mask_nms_thresh", type=float, default=0.5,
        help="Mask NMS IoU threshold (default: 0.5)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.40,
        help="Mask overlay transparency 0=invisible 1=opaque (default: 0.40)"
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed for image sampling (default: 2026)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for PNGs (default: work/vis/<run_name>/)"
    )
    args = parser.parse_args()

    logger = setup_logger("visualize")
    paths  = find_dataset_paths()
    repo_root = paths["repo_root"]

    # ── Resolve run directory ─────────────────────────────────────────────────
    run_dir = get_run_dir(repo_root, args.run_name)
    out_dir = args.out_dir or os.path.join(repo_root, "work", "vis", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load thresholds ───────────────────────────────────────────────────────
    thresh_file = os.path.join(run_dir, "best_thresholds.json")
    # fallback to legacy location
    if not os.path.exists(thresh_file):
        thresh_file = os.path.join(repo_root, "work", "best_thresholds.json")

    if os.path.exists(thresh_file):
        try:
            with open(thresh_file) as f:
                td = json.load(f)
            if args.score_thresh is None and "best_score_threshold" in td:
                args.score_thresh = float(td["best_score_threshold"])
                logger.info(f"Loaded score_thresh={args.score_thresh} from {thresh_file}")
            if args.mask_thresh is None and "best_mask_threshold" in td:
                args.mask_thresh = float(td["best_mask_threshold"])
                logger.info(f"Loaded mask_thresh={args.mask_thresh} from {thresh_file}")
        except Exception as e:
            logger.warning(f"Could not read {thresh_file}: {e}")

    if args.score_thresh is None:
        args.score_thresh = 0.35
        logger.info(f"Using default score_thresh={args.score_thresh}")
    if args.mask_thresh is None:
        args.mask_thresh = 0.5
        logger.info(f"Using default mask_thresh={args.mask_thresh}")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            os.path.join(run_dir, "checkpoints", f"maskrcnn_fold{args.fold}_best.pt"),
            # legacy flat path kept for base_all_e30 compatibility
            os.path.join(repo_root, "work", "checkpoints", f"maskrcnn_fold{args.fold}_best.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
    if ckpt_path is None or not os.path.exists(ckpt_path):
        logger.error(
            f"No checkpoint found. Tried: {candidates}. "
            "Pass --checkpoint explicitly."
        )
        sys.exit(1)
    logger.info(f"Using checkpoint: {ckpt_path}")

    # ── Load fold-0 images ────────────────────────────────────────────────────
    folds_df   = pd.read_csv(paths["folds_csv"])
    val_obs    = folds_df.loc[folds_df["fold"] == args.fold, "observation"].tolist()
    train_dir  = paths["train_dir"]

    image_paths: List[Tuple[str, str]] = []  # (obs_id, img_path)
    for obs in val_obs:
        for ext in (".jpeg", ".jpg"):
            p = os.path.join(train_dir, f"{obs}{ext}")
            if os.path.exists(p):
                image_paths.append((obs, p))
                break

    logger.info(f"Found {len(image_paths)} images in fold {args.fold}")

    # Sample N images deterministically
    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    selected = image_paths[: args.n]
    logger.info(f"Visualising {len(selected)} images → {out_dir}")

    # ── Load GT CSV ───────────────────────────────────────────────────────────
    gt_df = pd.read_csv(paths["gt_all_csv"])

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on {device} (img_size={args.img_size})...")
    model = get_maskrcnn_model(
        pretrained=False,
        min_size=args.img_size,
        max_size=args.img_size,
        num_classes=2,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # ── Process each image ────────────────────────────────────────────────────
    logger.info("==================================================")
    for i, (obs_id, img_path) in enumerate(selected):
        logger.info(f"[{i+1:02d}/{len(selected)}] {obs_id}")

        # Run inference
        pred_masks, pred_scores = run_inference_single(
            model=model,
            img_path=img_path,
            device=device,
            score_thresh=args.score_thresh,
            mask_thresh=args.mask_thresh,
            mask_nms_thresh=args.mask_nms_thresh,
            img_size=args.img_size,
        )
        logger.info(f"  Predicted masks: {len(pred_masks)}")

        # Load GT masks grouped by annotator
        gt_by_ann = load_gt_masks_by_annotator(gt_df, obs_id)
        n_gt_total = sum(len(v) for v in gt_by_ann.values())
        logger.info(f"  GT masks: {n_gt_total} from {len(gt_by_ann)} annotator(s): "
                    f"{sorted(gt_by_ann.keys())}")

        # Render and save
        out_path = os.path.join(out_dir, f"{obs_id}.png")
        render_triplet(
            img_path=img_path,
            obs_id=obs_id,
            pred_masks=pred_masks,
            pred_scores=pred_scores,
            gt_by_annotator=gt_by_ann,
            out_path=out_path,
            vis_size=args.vis_size,
            alpha=args.alpha,
        )
        logger.info(f"  Saved → {out_path}")

    logger.info("==================================================")
    logger.info(f"Done. {len(selected)} PNGs written to {out_dir}")


if __name__ == "__main__":
    main()
