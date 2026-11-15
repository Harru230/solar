#!/usr/bin/env python3
"""
scripts/predict.py - Inference and submission generation for Solar Filament Segmentation Challenge 2026.

Features:
- Runs a trained Mask R-CNN checkpoint on test images or validation folds.
- Configurable inference resolution via --img_size (default 1024); predicted masks are always
  upsampled back to 2048x2048 before RLE encoding.
- Optional TTA via --tta {none,flip}: for "flip", runs horizontal + vertical + both-flip passes,
  un-flips the predicted masks, and merges via Mask NMS (keeps pred_multi_hit near 0).
- Run isolation via --run_name: loads best_thresholds.json from work/runs/<run_name>/ by default
  and writes the submission CSV there as well.
- Upsamples soft mask probabilities from img_size to full resolution (2048x2048) with bilinear interpolation.
- Applies greedy Mask NMS to eliminate overlapping duplicate predictions (keeps gt_multi_hit and pred_multi_hit at 0).
- Encodes masks to compressed COCO RLE strings.
- Handles images with zero predictions: writes a row with empty-mask RLE ("PPPP4" for 2048x2048) so every image is present.
- Reports the count of zero-prediction images.
- Optionally evaluates against ground truth (using pq_eval.score) and reports PQ, SQ, RQ, TP, FP, FN, gt_multi_hit, pred_multi_hit.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
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
    run_tta_inference,
    setup_logger,
)


def load_image(img_path: str, img_size: int = 1024) -> Tuple[torch.Tensor, str]:
    """Loads 2048x2048 image, applies Multi-CLAHE preprocessing to img_size x img_size, returns 3-channel float tensor and base image_id."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_preprocessed = preprocess_solar_image(img_gray, img_size)
    image_id = Path(img_path).stem
    return torch.from_numpy(img_preprocessed), image_id


# Backward-compat alias used by tune_threshold.py
def load_image_1024(img_path: str) -> Tuple[torch.Tensor, str]:
    """Backward-compat alias: loads image resized to 1024x1024."""
    return load_image(img_path, img_size=1024)


def predict_batch(
    model: torch.nn.Module,
    batch_tensors: List[torch.Tensor],
    batch_ids: List[str],
    device: torch.device,
    score_thresh: float,
    mask_thresh: float,
    mask_nms_thresh: float,
    img_size: int = 1024,
    tta_mode: str = "none",
) -> Tuple[List[dict], int]:
    """
    Runs inference on a batch of images and generates submission rows.
    When tta_mode == "flip", runs 4 passes (original + H + V + HV) per image,
    un-flips predicted masks, and merges with Mask NMS.
    Returns: (list of prediction rows, count of images with zero predictions).
    """
    rows = []
    zero_pred_count = 0

    if tta_mode == "none":
        # Standard single-pass inference
        batch_gpu = [t.to(device) for t in batch_tensors]
        with torch.no_grad():
            outputs = model(batch_gpu)

        for i, output in enumerate(outputs):
            img_id = batch_ids[i]
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"]  # [K, 1, img_size, img_size]

            keep_score = scores >= score_thresh
            if not np.any(keep_score):
                zero_pred_count += 1
                rows.append({"filament_id": f"{img_id}_1", "segmentation_rle": EMPTY_RLE})
                continue

            scores_kept = scores[keep_score]
            masks_kept = masks[keep_score]

            # Bilinear upsampling soft mask probabilities to full 2048x2048 resolution in chunks
            if masks_kept.shape[0] > 0:
                bin_masks_list = []
                chunk_size = 32
                for c_start in range(0, masks_kept.shape[0], chunk_size):
                    c_chunk = masks_kept[c_start : c_start + chunk_size]
                    c_up = F.interpolate(c_chunk, size=(2048, 2048), mode="bilinear", align_corners=False)
                    bin_masks_list.append((c_up.squeeze(1) >= mask_thresh).byte().cpu().numpy())
                    del c_up
                bin_masks = np.concatenate(bin_masks_list, axis=0)
            else:
                bin_masks = np.zeros((0, 2048, 2048), dtype=np.uint8)

            valid_masks = []
            valid_scores = []
            valid_rles = []
            for m_idx in range(bin_masks.shape[0]):
                m = bin_masks[m_idx]
                if m.sum() > 0:
                    valid_masks.append(m)
                    valid_scores.append(float(scores_kept[m_idx]))
                    valid_rles.append(rle_encode_mask(m))

            # Discard any empty masks produced after thresholding
            if not valid_masks:
                zero_pred_count += 1
                rows.append({"filament_id": f"{img_id}_1", "segmentation_rle": EMPTY_RLE})
                continue

            # Apply greedy Mask NMS to eliminate duplicate overlapping predictions
            kept_nms = apply_mask_nms(valid_rles, valid_scores, iou_thresh=mask_nms_thresh)

            if not kept_nms:
                zero_pred_count += 1
                rows.append({"filament_id": f"{img_id}_1", "segmentation_rle": EMPTY_RLE})
                continue

            nms_masks = [valid_masks[kept_i] for kept_i in kept_nms]
            nms_scores = [valid_scores[kept_i] for kept_i in kept_nms]

            # Enforce strictly pixel-wise disjoint masks
            clean_masks, clean_scores = make_masks_disjoint(nms_masks, nms_scores)

            if not clean_masks:
                zero_pred_count += 1
                rows.append({"filament_id": f"{img_id}_1", "segmentation_rle": EMPTY_RLE})
                continue

            # Assertion right before writing RLE for each image verifying that the sum of all binary masks has max <= 1
            mask_sum = np.sum(np.stack(clean_masks, axis=0), axis=0)
            assert mask_sum.max() <= 1, f"Assertion failed: overlapping pixels detected for {img_id}: max overlap is {mask_sum.max()}"

            for k_idx, m in enumerate(clean_masks, start=1):
                rows.append({
                    "filament_id": f"{img_id}_{k_idx}",
                    "segmentation_rle": rle_encode_mask(m),
                })

    else:
        # TTA: process images one-by-one (each requires 4 forward passes)
        for tensor, img_id in zip(batch_tensors, batch_ids):
            merged_rles, merged_scores = run_tta_inference(
                model=model,
                img_tensor=tensor,
                device=device,
                score_thresh=score_thresh,
                mask_thresh=mask_thresh,
                mask_nms_thresh=mask_nms_thresh,
                img_size=img_size,
                tta_mode=tta_mode,
            )
            if not merged_rles:
                zero_pred_count += 1
                rows.append({"filament_id": f"{img_id}_1", "segmentation_rle": EMPTY_RLE})
            else:
                for k_idx, rle in enumerate(merged_rles, start=1):
                    rows.append({"filament_id": f"{img_id}_{k_idx}", "segmentation_rle": rle})

    return rows, zero_pred_count


def main():
    parser = argparse.ArgumentParser(description="Generate predictions and submission CSV")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name; used to locate best_thresholds.json at "
                             "work/runs/<run_name>/best_thresholds.json and default output path. "
                             "If omitted, falls back to the legacy work/best_thresholds.json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--images", type=str, nargs="*", default=None, help="Explicit list of image file paths")
    parser.add_argument("--folds_csv", type=str, default="work/folds.csv", help="Path to folds.csv for fold prediction")
    parser.add_argument("--fold", type=int, default=None, help="If specified, predict on this held-out fold from train_images")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output CSV (default: work/runs/<run_name>/submission.csv or work/submission.csv)")
    parser.add_argument("--score_thresh", type=float, default=None,
                        help="Detection score threshold (default: load from best_thresholds.json or 0.35)")
    parser.add_argument("--mask_thresh", type=float, default=None,
                        help="Mask binarization threshold (default: load from best_thresholds.json or 0.5)")
    parser.add_argument("--mask_nms_thresh", type=float, default=0.5, help="Mask NMS IoU threshold (default: 0.5)")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Inference resolution (default: read from best_thresholds.json or checkpoint config, fallback 1024)")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--tta", type=str, choices=["none", "flip"], default="none",
                        help="Test-Time Augmentation mode. 'flip': runs H-flip, V-flip, and HV-flip passes, "
                             "un-flips predicted masks, merges via Mask NMS (pred_multi_hit stays near 0). "
                             "Default: 'none' (single forward pass).")
    parser.add_argument("--eval", action="store_true", help="Evaluate against GT if GT is available")
    parser.add_argument("--gt_csv", type=str, default="work/gt_all.csv", help="Ground truth CSV for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to predict (for testing)")
    args = parser.parse_args()

    logger = setup_logger("predict")
    paths = find_dataset_paths()
    repo_root = paths["repo_root"]

    # ── Resolve run-scoped paths ───────────────────────────────────────────────
    if args.run_name:
        run_dir = get_run_dir(repo_root, args.run_name)
        best_thresh_file = os.path.join(run_dir, "best_thresholds.json")
    else:
        run_dir = os.path.join(repo_root, "work")
        best_thresh_file = os.path.join(repo_root, "work/best_thresholds.json")

    if args.output is None:
        args.output = os.path.join(run_dir, "submission.csv")

    # Load thresholds and img_size from best_thresholds.json if not explicitly specified
    if os.path.exists(best_thresh_file):
        try:
            with open(best_thresh_file) as f:
                thresh_data = json.load(f)
            if args.score_thresh is None and "best_score_threshold" in thresh_data:
                args.score_thresh = float(thresh_data["best_score_threshold"])
                logger.info(f"Loaded tuned score_threshold from {best_thresh_file}: {args.score_thresh}")
            if args.mask_thresh is None and "best_mask_threshold" in thresh_data:
                args.mask_thresh = float(thresh_data["best_mask_threshold"])
                logger.info(f"Loaded tuned mask_threshold from {best_thresh_file}: {args.mask_thresh}")
            if args.img_size is None and "img_size" in thresh_data:
                args.img_size = int(thresh_data["img_size"])
                logger.info(f"Loaded img_size from {best_thresh_file}: {args.img_size}")
        except Exception as e:
            logger.warning(f"Could not read {best_thresh_file}: {e}")

    if args.score_thresh is None:
        args.score_thresh = 0.35
    if args.mask_thresh is None:
        args.mask_thresh = 0.5

    # Resolve device and checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint '{args.checkpoint}' onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_config = checkpoint.get("config", {})

    if args.img_size is None:
        args.img_size = ckpt_config.get("img_size", 1024)
        logger.info(f"Inferred img_size from checkpoint config: {args.img_size}")

    logger.info("==================================================")
    logger.info("Starting Solar Filament Prediction Pipeline")
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Score Threshold: {args.score_thresh}")
    logger.info(f"Mask Threshold:  {args.mask_thresh}")
    logger.info(f"Mask NMS IoU:    {args.mask_nms_thresh}")
    logger.info(f"Img Size:        {args.img_size}x{args.img_size}")
    logger.info(f"Preprocessing:   3-channel Multi-CLAHE [ch0: raw/255, ch1: CLAHE(2, 8x8), ch2: CLAHE(4, 16x16)]")
    logger.info(f"TTA Mode:        {args.tta}")
    logger.info(f"Output File:     {args.output}")
    logger.info("==================================================")

    # Collect images to process
    image_paths = []
    if args.fold is not None:
        # Predict on validation fold
        if not os.path.exists(args.folds_csv):
            logger.error(f"Folds file not found at {args.folds_csv}")
            sys.exit(1)
        folds_df = pd.read_csv(args.folds_csv)
        val_obs = folds_df.loc[folds_df["fold"] == args.fold, "observation"].tolist()
        train_dir = paths["train_dir"]
        for o in val_obs:
            p = os.path.join(train_dir, f"{o}.jpeg")
            if not os.path.exists(p):
                p = os.path.join(train_dir, f"{o}.jpg")
            if os.path.exists(p):
                image_paths.append(p)
        logger.info(f"Loaded {len(image_paths)} images from fold {args.fold}")
    elif args.images:
        image_paths = args.images
    elif args.image_dir:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jp*g")))
    else:
        # Default to test directory
        test_dir = paths["test_dir"]
        if test_dir and os.path.exists(test_dir):
            image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jp*g")))
            logger.info(f"Using test images directory: {test_dir} ({len(image_paths)} images)")
        else:
            logger.error("No images specified and test_dir not found!")
            sys.exit(1)

    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]
        logger.info(f"Limiting prediction to first {args.limit} images")

    if not image_paths:
        logger.error("No images found to predict!")
        sys.exit(1)

    # Initialize model
    ckpt_anchors5 = ckpt_config.get("anchors5", False)
    model = get_maskrcnn_model(
        pretrained=False,
        min_size=args.img_size,
        max_size=args.img_size,
        num_classes=2,
        anchors5=ckpt_anchors5,
    )
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()

    # Process in batches
    all_rows = []
    total_zero_preds = 0
    t0 = time.time()
    n_images = len(image_paths)

    for i in range(0, n_images, args.batch_size):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_tensors = []
        batch_ids = []
        for p in batch_paths:
            tensor, img_id = load_image(p, img_size=args.img_size)
            batch_tensors.append(tensor)
            batch_ids.append(img_id)

        rows, zero_cnt = predict_batch(
            model=model,
            batch_tensors=batch_tensors,
            batch_ids=batch_ids,
            device=device,
            score_thresh=args.score_thresh,
            mask_thresh=args.mask_thresh,
            mask_nms_thresh=args.mask_nms_thresh,
            img_size=args.img_size,
            tta_mode=args.tta,
        )
        all_rows.extend(rows)
        total_zero_preds += zero_cnt

        if (i // args.batch_size + 1) % 10 == 0 or (i + len(batch_paths)) == n_images:
            elapsed = time.time() - t0
            logger.info(
                f"Progress: [{min(i + len(batch_paths), n_images):04d}/{n_images:04d}] images "
                f"({elapsed:.1f}s, {elapsed / max(1, min(i + len(batch_paths), n_images)):.3f}s/img)"
            )

    # Save output CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pred_df = pd.DataFrame(all_rows, columns=["filament_id", "segmentation_rle"])
    pred_df.to_csv(args.output, index=False)

    logger.info("--------------------------------------------------")
    logger.info(f"Predictions saved to: {args.output}")
    logger.info(f"Total prediction rows: {len(pred_df)}")
    logger.info(
        f"Images with zero predictions: {total_zero_preds} / {n_images} "
        f"(written as empty-mask '{EMPTY_RLE}' rows to ensure every image is present)"
    )
    logger.info(f"Total elapsed time: {time.time() - t0:.2f}s")
    logger.info("--------------------------------------------------")

    # Evaluation if requested
    if args.eval:
        if not os.path.exists(args.gt_csv):
            logger.error(f"Cannot evaluate: GT file not found at {args.gt_csv}")
            return

        logger.info(f"Evaluating predictions against GT ({args.gt_csv})...")
        gt_df = pd.read_csv(args.gt_csv)
        if args.fold is not None and os.path.exists(args.folds_csv):
            folds_df = pd.read_csv(args.folds_csv)
            keep_obs = set(folds_df.loc[folds_df["fold"] == args.fold, "observation"])
            gt_obs = gt_df["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1]
            gt_df = gt_df[gt_obs.isin(keep_obs)].copy()

        eval_res = pq_eval.score(gt_df, pred_df)
        logger.info("================ EVALUATION RESULTS ================")
        for key, val in eval_res.items():
            if isinstance(val, float):
                logger.info(f"{key:26s} {val:.4f}")
            else:
                logger.info(f"{key:26s} {val}")
        logger.info("====================================================")


if __name__ == "__main__":
    main()
