#!/usr/bin/env python3
"""
scripts/tune_threshold.py - Threshold Sweeper for Panoptic Quality Optimization.

Sweeps detection score threshold (0.05 ... 0.90) and mask probability thresholds
on the held-out validation fold. Reports PQ, SQ, RQ, TP, FP, FN per threshold.

Advanced post-processing features:
- Expanded mask thresholds: 0.40 to 0.80 with step 0.05.
- Efficient IoU matrix precomputation: IoU matrices computed once per mask threshold,
  allowing instant column slicing for all score threshold evaluations.
- Maskness rescoring (--rescore):
    new_score = (cls_score ** gamma) * (mean_mask_prob ** (1 - gamma))
    where mean_mask_prob is the average soft mask probability strictly over pixels >= mask_threshold.
    Evaluated over gamma in {0.3, 0.5, 0.7}.
- Size-bin thresholds (--size_bins):
    Categorizes instances into Small, Medium, Large based on predicted mask pixel area.
    Bin edges (P33, P66) are computed from predictions on training folds (fold != val_fold).
    Independent thresholds per bin optimized using 2-round coordinate descent.
- Selection bias control:
    Validation observations split into halves A and B (seed 2026).
    Evaluates generalization by tuning on A and testing on B, and vice versa.
    Highlights variants only if they beat standard sweep by > +0.01 on held-out in both directions.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    rle_encode_mask,
    run_tta_inference,
    setup_logger,
)
from scripts.predict import load_image


def compute_size_bin_edges(
    model: torch.nn.Module,
    train_image_paths: List[str],
    device: torch.device,
    img_size: int,
    base_score_thresh: float = 0.05,
    mask_thresh: float = 0.5,
    sample_size: int = 60,
    seed: int = 2026,
    logger=None,
) -> Tuple[float, float]:
    """
    Computes P33 and P66 area percentiles strictly from training fold images.
    Returns: (p33, p66) in pixel area (at 2048x2048 resolution).
    """
    model.eval()
    rng = random.Random(seed)
    sampled_paths = list(train_image_paths)
    rng.shuffle(sampled_paths)
    sampled_paths = sampled_paths[:sample_size]

    if logger:
        logger.info(f"Computing size-bin edges from {len(sampled_paths)} training fold images (seed {seed})...")

    t0 = time.time()
    areas = []
    batch_size = 4

    for i in range(0, len(sampled_paths), batch_size):
        batch = sampled_paths[i : i + batch_size]
        batch_tensors = []
        for p in batch:
            tensor, _ = load_image(p, img_size=img_size)
            batch_tensors.append(tensor.to(device))

        with torch.no_grad():
            outputs = model(batch_tensors)

        for output in outputs:
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"]  # [K, 1, img_size, img_size]
            keep = scores >= base_score_thresh
            if not np.any(keep):
                continue

            scores_kept = scores[keep]
            masks_kept = masks[keep]

            # Upsample to 2048x2048
            masks_up = F.interpolate(
                masks_kept.float(), size=(2048, 2048), mode="bilinear", align_corners=False
            ).squeeze(1)
            bin_masks_gpu = masks_up >= mask_thresh
            bin_masks = bin_masks_gpu.byte().cpu().numpy()

            valid_idx = [k for k in range(len(scores_kept)) if bin_masks[k].sum() > 0]
            if not valid_idx:
                continue

            vm = [bin_masks[k] for k in valid_idx]
            vs = [float(scores_kept[k]) for k in valid_idx]
            vr = [rle_encode_mask(m) for m in vm]

            kept_nms = apply_mask_nms(vr, vs, iou_thresh=0.5)
            nm = [vm[ki] for ki in kept_nms]
            ns = [vs[ki] for ki in kept_nms]

            clean_m, _ = make_masks_disjoint(nm, ns)
            for m in clean_m:
                a = int(m.sum())
                if a > 0:
                    areas.append(a)

    if not areas:
        if logger:
            logger.warning("No training instances found for size bins, using defaults [500.0, 1500.0]")
        return 500.0, 1500.0

    p33 = float(np.percentile(areas, 100.0 / 3.0))
    p66 = float(np.percentile(areas, 200.0 / 3.0))
    elapsed = time.time() - t0
    if logger:
        logger.info(
            f"Size-bin edges computed in {elapsed:.2f}s over {len(areas)} instances: "
            f"P33 = {p33:.1f} px, P66 = {p66:.1f} px"
        )

    return p33, p66


def precompute_threshold_cache(
    model: torch.nn.Module,
    val_image_paths: List[str],
    val_gt_df: pd.DataFrame,
    mask_thresholds: List[float],
    device: torch.device,
    img_size: int,
    base_score_thresh: float = 0.05,
    bin_edges: Optional[Tuple[float, float]] = None,
    batch_size: int = 4,
    logger=None,
) -> Dict[float, Dict[str, Tuple[np.ndarray, List[dict]]]]:
    """
    Runs model inference ONCE across all validation images and precomputes
    IoU matrices and candidate attributes for each mask threshold.

    Returns:
        cache[mask_thresh][annotator_image] = (iou_matrix, candidate_records)
        where candidate_records is a list of dicts with keys:
          'cls_score', 'mean_mask_prob', 'scores_gamma', 'area', 'size_bin', 'rle'
    """
    model.eval()
    p33, p66 = bin_edges if bin_edges is not None else (0.0, 0.0)

    # Pre-index GT annotations by observation and annotator-image
    gt_ai = val_gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_by_ai = {}
    obs_to_ais = {}
    for ai, rows in val_gt_df.groupby(gt_ai, sort=False):
        obs = ai.split("-", 1)[1]
        rles = rows["segmentation_rle"].tolist()
        gt_by_ai[ai] = rles
        obs_to_ais.setdefault(obs, []).append(ai)

    cache = {m_th: {} for m_th in mask_thresholds}
    n_images = len(val_image_paths)
    t0 = time.time()

    if logger:
        logger.info(
            f"Precomputing candidates and IoU matrices for {len(mask_thresholds)} mask thresholds "
            f"across {n_images} validation images..."
        )

    for b_idx in range(0, n_images, batch_size):
        batch_paths = val_image_paths[b_idx : b_idx + batch_size]
        batch_tensors = []
        batch_obs = []
        for p in batch_paths:
            tensor, obs = load_image(p, img_size=img_size)
            batch_tensors.append(tensor.to(device))
            batch_obs.append(obs)

        with torch.no_grad():
            outputs = model(batch_tensors)

        for img_idx, output in enumerate(outputs):
            obs = batch_obs[img_idx]
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"]  # [K, 1, img_size, img_size]

            keep = scores >= base_score_thresh
            has_candidates = np.any(keep)

            if has_candidates:
                scores_kept = scores[keep]
                masks_kept = masks[keep]
                K = len(scores_kept)

                # Upsample once to 2048x2048 for this image
                masks_up = F.interpolate(
                    masks_kept.float(), size=(2048, 2048), mode="bilinear", align_corners=False
                ).squeeze(1)
                # 512x512 downsampling for accelerated GPU NMS
                masks_512 = F.interpolate(
                    masks_kept.float(), size=(512, 512), mode="nearest"
                ).squeeze(1)

            ais_for_obs = obs_to_ais.get(obs, [])

            for m_th in mask_thresholds:
                clean_candidates = []

                if has_candidates:
                    # Accelerated GPU pairwise NMS
                    bin_512 = (masks_512 >= m_th).float()
                    areas_512 = bin_512.sum(dim=(-2, -1))
                    valid_512 = areas_512 > 0

                    if valid_512.any():
                        flat = bin_512.view(K, -1)
                        inter = torch.mm(flat, flat.t())
                        union = areas_512.unsqueeze(1) + areas_512.unsqueeze(0) - inter
                        iou_gpu = torch.where(union > 0, inter / union, torch.zeros_like(inter)).cpu().numpy()

                        order = np.argsort(scores_kept)[::-1]
                        keep_nms = []
                        for i in range(K):
                            if not valid_512[order[i]]:
                                continue
                            orig_i = order[i]
                            overlap = False
                            for k in keep_nms:
                                if iou_gpu[orig_i, order[k]] > 0.5:
                                    overlap = True
                                    break
                            if not overlap:
                                keep_nms.append(i)

                        kept_indices = [int(order[i]) for i in keep_nms]

                        if kept_indices:
                            # 2048 binary masks & mean_mask_prob computation
                            bin_gpu = masks_up[kept_indices] >= m_th
                            counts = bin_gpu.float().sum(dim=(-2, -1))
                            sums = (masks_up[kept_indices] * bin_gpu.float()).sum(dim=(-2, -1))
                            mean_probs = (
                                torch.where(counts > 0, sums / counts, torch.zeros_like(counts))
                                .cpu()
                                .numpy()
                            )

                            bin_masks_cpu = bin_gpu.byte().cpu().numpy()
                            scores_nms = [float(scores_kept[idx]) for idx in kept_indices]

                            # Disjoint enforcement
                            clean_m, clean_s = make_masks_disjoint(list(bin_masks_cpu), scores_nms)

                            # Map back to mean_prob for clean masks
                            # clean_m matches clean_s order
                            score_to_mean_prob = {
                                scores_nms[idx]: float(mean_probs[idx])
                                for idx in range(len(scores_nms))
                            }

                            for c_m, c_s in zip(clean_m, clean_s):
                                area = int(c_m.sum())
                                mp = score_to_mean_prob.get(c_s, 0.5)

                                # Size-bin assignment
                                if area < p33:
                                    sz_bin = "small"
                                elif area < p66:
                                    sz_bin = "medium"
                                else:
                                    sz_bin = "large"

                                # Rescored variants
                                s_g03 = (c_s ** 0.3) * (mp ** 0.7)
                                s_g05 = (c_s ** 0.5) * (mp ** 0.5)
                                s_g07 = (c_s ** 0.7) * (mp ** 0.3)

                                rle = rle_encode_mask(c_m)
                                clean_candidates.append({
                                    "cls_score": c_s,
                                    "mean_mask_prob": mp,
                                    "scores_gamma": {0.3: s_g03, 0.5: s_g05, 0.7: s_g07},
                                    "area": area,
                                    "size_bin": sz_bin,
                                    "rle": rle,
                                })

                pred_rles = [c["rle"] for c in clean_candidates]
                for ai in ais_for_obs:
                    gt_rles = gt_by_ai[ai]
                    m_full = pq_eval.iou_matrix(gt_rles, pred_rles)
                    cache[m_th][ai] = (m_full, clean_candidates)

        if logger and ((b_idx // batch_size + 1) % 5 == 0 or (b_idx + len(batch_paths)) == n_images):
            processed = min(b_idx + len(batch_paths), n_images)
            elapsed = time.time() - t0
            rate = elapsed / processed
            rem = rate * (n_images - processed)
            logger.info(
                f"Candidate precomputation: [{processed:03d}/{n_images:03d}] images "
                f"({elapsed:.1f}s, ~{rem:.1f}s remaining)"
            )

    return cache


def evaluate_from_cache(
    cached_ai_data: Dict[str, Tuple[np.ndarray, List[dict]]],
    filter_fn,
    split_obs: Optional[set] = None,
) -> dict:
    """
    Evaluates PQ from precomputed IoU matrices using column slicing.
    Zero RLE decoding or IoU recomputation.
    """
    matrices = []
    for ai, (m_full, candidates) in cached_ai_data.items():
        if split_obs is not None:
            obs = ai.split("-", 1)[1]
            if obs not in split_obs:
                continue

        if len(candidates) == 0:
            matrices.append(m_full)
        else:
            keep = [k for k, c in enumerate(candidates) if filter_fn(c)]
            matrices.append(m_full[:, keep])

    return pq_eval.pq_from_matrices(matrices)


def run_standard_sweep(
    cache: Dict[float, Dict[str, Tuple[np.ndarray, List[dict]]]],
    mask_thresholds: List[float],
    score_thresholds: List[float],
    split_obs: Optional[set] = None,
) -> Tuple[dict, List[dict]]:
    """
    Runs standard score sweep across all mask thresholds using precomputed IoU matrices.
    """
    best_result = None
    best_pq = -1.0
    results_table = []

    for mask_th in mask_thresholds:
        ai_data = cache[mask_th]
        for score_th in score_thresholds:
            metrics = evaluate_from_cache(
                ai_data,
                filter_fn=lambda c: c["cls_score"] >= score_th,
                split_obs=split_obs,
            )

            pq = metrics["pq"]
            row = {
                "method": "standard",
                "mask_threshold": mask_th,
                "score_threshold": score_th,
                "gamma": None,
                "score_thresh_small": None,
                "score_thresh_med": None,
                "score_thresh_large": None,
                "pq": pq,
                "sq": metrics["sq"],
                "rq": metrics["rq"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "gt_multi_hit": metrics.get("gt_multi_hit", 0),
                "pred_multi_hit": metrics.get("pred_multi_hit", 0),
            }
            results_table.append(row)

            if pq > best_pq:
                best_pq = pq
                best_result = row

    return best_result, results_table


def run_rescore_sweep(
    cache: Dict[float, Dict[str, Tuple[np.ndarray, List[dict]]]],
    mask_thresholds: List[float],
    score_thresholds: List[float],
    gammas: List[float] = [0.3, 0.5, 0.7],
    split_obs: Optional[set] = None,
) -> Tuple[dict, List[dict]]:
    """
    Runs maskness rescoring sweep over gamma in {0.3, 0.5, 0.7}.
    """
    best_result = None
    best_pq = -1.0
    results_table = []

    for gamma in gammas:
        for mask_th in mask_thresholds:
            ai_data = cache[mask_th]
            for score_th in score_thresholds:
                metrics = evaluate_from_cache(
                    ai_data,
                    filter_fn=lambda c: c["scores_gamma"][gamma] >= score_th,
                    split_obs=split_obs,
                )

                pq = metrics["pq"]
                row = {
                    "method": "rescore",
                    "mask_threshold": mask_th,
                    "score_threshold": score_th,
                    "gamma": gamma,
                    "score_thresh_small": None,
                    "score_thresh_med": None,
                    "score_thresh_large": None,
                    "pq": pq,
                    "sq": metrics["sq"],
                    "rq": metrics["rq"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "gt_multi_hit": metrics.get("gt_multi_hit", 0),
                    "pred_multi_hit": metrics.get("pred_multi_hit", 0),
                }
                results_table.append(row)

                if pq > best_pq:
                    best_pq = pq
                    best_result = row

    return best_result, results_table


def run_size_bins_coordinate_descent(
    cache: Dict[float, Dict[str, Tuple[np.ndarray, List[dict]]]],
    mask_thresholds: List[float],
    score_thresholds: List[float],
    best_standard_score: float = 0.35,
    rounds: int = 2,
    split_obs: Optional[set] = None,
) -> Tuple[dict, List[dict]]:
    """
    Tunes per-bin score thresholds (Small, Medium, Large) using coordinate descent.
    """
    best_result = None
    best_pq = -1.0
    results_table = []

    for mask_th in mask_thresholds:
        ai_data = cache[mask_th]

        # Initialize thresholds from baseline best score
        t_small = best_standard_score
        t_med = best_standard_score
        t_large = best_standard_score

        for r in range(rounds):
            # 1. Sweep t_small
            best_t_small = t_small
            best_pq_step = -1.0
            for s_val in score_thresholds:
                m = evaluate_from_cache(
                    ai_data,
                    filter_fn=lambda c, s=s_val, tm=t_med, tl=t_large: (
                        c["cls_score"] >= s if c["size_bin"] == "small"
                        else (c["cls_score"] >= tm if c["size_bin"] == "medium" else c["cls_score"] >= tl)
                    ),
                    split_obs=split_obs,
                )
                if m["pq"] > best_pq_step:
                    best_pq_step = m["pq"]
                    best_t_small = s_val
            t_small = best_t_small

            # 2. Sweep t_med
            best_t_med = t_med
            best_pq_step = -1.0
            for s_val in score_thresholds:
                m = evaluate_from_cache(
                    ai_data,
                    filter_fn=lambda c, ts=t_small, s=s_val, tl=t_large: (
                        c["cls_score"] >= ts if c["size_bin"] == "small"
                        else (c["cls_score"] >= s if c["size_bin"] == "medium" else c["cls_score"] >= tl)
                    ),
                    split_obs=split_obs,
                )
                if m["pq"] > best_pq_step:
                    best_pq_step = m["pq"]
                    best_t_med = s_val
            t_med = best_t_med

            # 3. Sweep t_large
            best_t_large = t_large
            best_pq_step = -1.0
            for s_val in score_thresholds:
                m = evaluate_from_cache(
                    ai_data,
                    filter_fn=lambda c, ts=t_small, tm=t_med, s=s_val: (
                        c["cls_score"] >= ts if c["size_bin"] == "small"
                        else (c["cls_score"] >= tm if c["size_bin"] == "medium" else c["cls_score"] >= s)
                    ),
                    split_obs=split_obs,
                )
                if m["pq"] > best_pq_step:
                    best_pq_step = m["pq"]
                    best_t_large = s_val
            t_large = best_t_large

        # Final evaluation for this mask_th
        final_metrics = evaluate_from_cache(
            ai_data,
            filter_fn=lambda c, ts=t_small, tm=t_med, tl=t_large: (
                c["cls_score"] >= ts if c["size_bin"] == "small"
                else (c["cls_score"] >= tm if c["size_bin"] == "medium" else c["cls_score"] >= tl)
            ),
            split_obs=split_obs,
        )

        row = {
            "method": "size_bins",
            "mask_threshold": mask_th,
            "score_threshold": None,
            "gamma": None,
            "score_thresh_small": t_small,
            "score_thresh_med": t_med,
            "score_thresh_large": t_large,
            "pq": final_metrics["pq"],
            "sq": final_metrics["sq"],
            "rq": final_metrics["rq"],
            "tp": final_metrics["tp"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "gt_multi_hit": final_metrics.get("gt_multi_hit", 0),
            "pred_multi_hit": final_metrics.get("pred_multi_hit", 0),
        }
        results_table.append(row)

        if final_metrics["pq"] > best_pq:
            best_pq = final_metrics["pq"]
            best_result = row

    return best_result, results_table


def print_ascii_table(title: str, headers: List[str], rows: List[List[str]]):
    """Prints a formatted ASCII table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    print("\n" + "=" * (sum(widths) + 3 * len(widths) + 1), flush=True)
    print(f" {title}", flush=True)
    print("=" * (sum(widths) + 3 * len(widths) + 1), flush=True)

    header_str = " | ".join(f"{h:^{widths[i]}}" for i, h in enumerate(headers))
    print(f"| {header_str} |", flush=True)
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(f"+-{sep_str}-+", flush=True)

    for row in rows:
        row_str = " | ".join(f"{str(cell):^{widths[i]}}" for i, cell in enumerate(row))
        print(f"| {row_str} |", flush=True)

    print("=" * (sum(widths) + 3 * len(widths) + 1) + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extended threshold sweep with rescoring and size bins")
    parser.add_argument("--run_name", type=str, default="default",
                        help="Run name; outputs go to work/runs/<run_name>/")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--fold", type=int, default=0, help="Held-out validation fold to tune on (default: 0)")
    parser.add_argument("--folds_csv", type=str, default="work/folds.csv", help="Path to folds.csv")
    parser.add_argument("--gt_csv", type=str, default="work/gt_all.csv", help="Path to gt_all.csv")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Path to save best thresholds JSON")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Path to save full sweep table CSV (default: work/runs/<run_name>/threshold_sweep_extended.csv)")
    parser.add_argument("--score_min", type=float, default=0.05, help="Min score threshold")
    parser.add_argument("--score_max", type=float, default=0.90, help="Max score threshold")
    parser.add_argument("--score_step", type=float, default=0.05, help="Score threshold step")
    parser.add_argument("--mask_thresholds", type=float, nargs="+",
                        default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                        help="Mask probability thresholds to evaluate")
    parser.add_argument("--mask_nms_thresh", type=float, default=0.5, help="Mask NMS IoU threshold")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Inference resolution; if None, read from checkpoint config (default: 1536)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for candidate extraction")
    parser.add_argument("--rescore", action="store_true", help="Evaluate maskness rescoring sweep")
    parser.add_argument("--size_bins", action="store_true", help="Evaluate size-bin coordinate descent")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for A/B split and training sampling")
    parser.add_argument("--limit", type=int, default=None, help="Limit images (for smoke testing)")
    args = parser.parse_args()

    paths = find_dataset_paths()
    repo_root = paths["repo_root"]
    run_dir = get_run_dir(repo_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.out_json is None:
        args.out_json = os.path.join(run_dir, "best_thresholds.json")
    if args.out_csv is None:
        args.out_csv = os.path.join(run_dir, "threshold_sweep_extended.csv")

    logger = setup_logger("tune_threshold")

    logger.info("==================================================")
    logger.info("Extended Post-Processing Tuning for Panoptic Quality")
    logger.info(f"Run name:     {args.run_name}")
    logger.info(f"Run dir:      {run_dir}")
    logger.info(f"Checkpoint:   {args.checkpoint}")
    logger.info(f"Fold:         {args.fold}")
    logger.info(f"Rescore:      {args.rescore}")
    logger.info(f"Size bins:    {args.size_bins}")
    logger.info(f"Output CSV:   {args.out_csv}")
    logger.info(f"Output JSON:  {args.out_json}")
    logger.info("==================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint '{args.checkpoint}' onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    # Consistency verification: read img_size from config
    ckpt_img_size = config.get("img_size", 1024)
    if args.img_size is None:
        args.img_size = ckpt_img_size

    ckpt_anchors5 = config.get("anchors5", False)
    model = get_maskrcnn_model(
        pretrained=False,
        min_size=args.img_size,
        max_size=args.img_size,
        num_classes=2,
        anchors5=ckpt_anchors5,
    )
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)

    # Log preprocessing consistency
    logger.info("---------------- PREPROCESSING CONSISTENCY ----------------")
    logger.info(f"Model resolution:         {args.img_size}x{args.img_size} (from config: {ckpt_img_size})")
    logger.info(f"Color input pipeline:     3-channel Multi-CLAHE [ch0: raw/255, ch1: CLAHE(2, 8x8), ch2: CLAHE(4, 16x16)]")
    logger.info(f"Mask head RoI pool size:  {model.roi_heads.mask_roi_pool.output_size} -> 56x56 masks")
    logger.info(f"RoI detections_per_img:   {model.roi_heads.detections_per_img}")
    logger.info("-----------------------------------------------------------")

    # Load validation fold images
    folds_df = pd.read_csv(args.folds_csv)
    val_obs = folds_df.loc[folds_df["fold"] == args.fold, "observation"].tolist()
    if args.limit and args.limit > 0:
        val_obs = val_obs[:args.limit]

    train_dir = paths["train_dir"]
    val_image_paths = []
    for o in val_obs:
        p = os.path.join(train_dir, f"{o}.jpeg")
        if not os.path.exists(p):
            p = os.path.join(train_dir, f"{o}.jpg")
        if os.path.exists(p):
            val_image_paths.append(p)

    logger.info(f"Validation images in fold {args.fold}: {len(val_image_paths)}")

    # Load GT for validation fold
    gt_df = pd.read_csv(args.gt_csv)
    keep_obs = set(val_obs)
    gt_obs = gt_df["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1]
    val_gt_df = gt_df[gt_obs.isin(keep_obs)].copy()
    logger.info(
        f"GT annotations in validation fold: {len(val_gt_df)} rows across "
        f"{val_gt_df['filament_id'].str.split('_').str[0].nunique()} annotator-images"
    )

    # Selection bias control: split 142 observations into Halves A and B
    val_obs_sorted = sorted(val_obs)
    rng = random.Random(args.seed)
    rng.shuffle(val_obs_sorted)
    half_idx = len(val_obs_sorted) // 2
    obs_set_A = set(val_obs_sorted[:half_idx])
    obs_set_B = set(val_obs_sorted[half_idx:])
    logger.info(
        f"Selection Bias Control: Split {len(val_obs_sorted)} observations into "
        f"Half A ({len(obs_set_A)} obs) and Half B ({len(obs_set_B)} obs) with seed {args.seed}"
    )

    # Compute size bin edges from training folds if --size_bins is active
    bin_edges = None
    if args.size_bins:
        train_obs = folds_df.loc[folds_df["fold"] != args.fold, "observation"].tolist()
        train_image_paths = []
        for o in train_obs:
            p = os.path.join(train_dir, f"{o}.jpeg")
            if not os.path.exists(p):
                p = os.path.join(train_dir, f"{o}.jpg")
            if os.path.exists(p):
                train_image_paths.append(p)

        bin_edges = compute_size_bin_edges(
            model=model,
            train_image_paths=train_image_paths,
            device=device,
            img_size=args.img_size,
            base_score_thresh=args.score_min,
            sample_size=60,
            seed=args.seed,
            logger=logger,
        )

    score_thresholds = np.arange(args.score_min, args.score_max + 1e-5, args.score_step).round(3).tolist()
    mask_thresholds = [round(m, 3) for m in args.mask_thresholds]

    # Precompute IoU matrices and candidates cache across all mask thresholds
    cache = precompute_threshold_cache(
        model=model,
        val_image_paths=val_image_paths,
        val_gt_df=val_gt_df,
        mask_thresholds=mask_thresholds,
        device=device,
        img_size=args.img_size,
        base_score_thresh=args.score_min,
        bin_edges=bin_edges,
        batch_size=args.batch_size,
        logger=logger,
    )

    all_sweep_records = []

    # ──────────────── 1. STANDARD SWEEP ────────────────
    logger.info("Running Standard Sweep (9 mask thresholds x 18 score thresholds)...")
    t_std = time.time()
    best_std_full, records_std = run_standard_sweep(
        cache, mask_thresholds, score_thresholds, split_obs=None
    )
    all_sweep_records.extend(records_std)

    # Standard A/B generalization
    best_std_A, _ = run_standard_sweep(cache, mask_thresholds, score_thresholds, split_obs=obs_set_A)
    best_std_B, _ = run_standard_sweep(cache, mask_thresholds, score_thresholds, split_obs=obs_set_B)

    # Evaluate best A on B, and best B on A
    eval_std_A_on_B = evaluate_from_cache(
        cache[best_std_A["mask_threshold"]],
        filter_fn=lambda c: c["cls_score"] >= best_std_A["score_threshold"],
        split_obs=obs_set_B,
    )
    eval_std_B_on_A = evaluate_from_cache(
        cache[best_std_B["mask_threshold"]],
        filter_fn=lambda c: c["cls_score"] >= best_std_B["score_threshold"],
        split_obs=obs_set_A,
    )
    logger.info(f"Standard sweep finished in {time.time() - t_std:.2f}s")

    # ──────────────── 2. RESCORING SWEEP ────────────────
    best_rescore_full = None
    eval_rescore_A_on_B = None
    eval_rescore_B_on_A = None
    best_rescore_A = None
    best_rescore_B = None

    if args.rescore:
        logger.info("Running Maskness Rescoring Sweep (3 gammas x 9 mask thresholds x 18 score thresholds)...")
        t_rescore = time.time()
        best_rescore_full, records_rescore = run_rescore_sweep(
            cache, mask_thresholds, score_thresholds, gammas=[0.3, 0.5, 0.7], split_obs=None
        )
        all_sweep_records.extend(records_rescore)

        # Rescore A/B generalization
        best_rescore_A, _ = run_rescore_sweep(
            cache, mask_thresholds, score_thresholds, gammas=[0.3, 0.5, 0.7], split_obs=obs_set_A
        )
        best_rescore_B, _ = run_rescore_sweep(
            cache, mask_thresholds, score_thresholds, gammas=[0.3, 0.5, 0.7], split_obs=obs_set_B
        )

        eval_rescore_A_on_B = evaluate_from_cache(
            cache[best_rescore_A["mask_threshold"]],
            filter_fn=lambda c: c["scores_gamma"][best_rescore_A["gamma"]] >= best_rescore_A["score_threshold"],
            split_obs=obs_set_B,
        )
        eval_rescore_B_on_A = evaluate_from_cache(
            cache[best_rescore_B["mask_threshold"]],
            filter_fn=lambda c: c["scores_gamma"][best_rescore_B["gamma"]] >= best_rescore_B["score_threshold"],
            split_obs=obs_set_A,
        )
        logger.info(f"Rescoring sweep finished in {time.time() - t_rescore:.2f}s")

    # ──────────────── 3. SIZE-BIN SWEEP ────────────────
    best_size_full = None
    eval_size_A_on_B = None
    eval_size_B_on_A = None
    best_size_A = None
    best_size_B = None

    if args.size_bins:
        logger.info("Running Size-Bin Coordinate Descent (2 rounds per mask threshold)...")
        t_size = time.time()
        best_size_full, records_size = run_size_bins_coordinate_descent(
            cache, mask_thresholds, score_thresholds,
            best_standard_score=best_std_full["score_threshold"],
            rounds=2, split_obs=None
        )
        all_sweep_records.extend(records_size)

        # Size-bin A/B generalization
        best_size_A, _ = run_size_bins_coordinate_descent(
            cache, mask_thresholds, score_thresholds,
            best_standard_score=best_std_A["score_threshold"],
            rounds=2, split_obs=obs_set_A
        )
        best_size_B, _ = run_size_bins_coordinate_descent(
            cache, mask_thresholds, score_thresholds,
            best_standard_score=best_std_B["score_threshold"],
            rounds=2, split_obs=obs_set_B
        )

        eval_size_A_on_B = evaluate_from_cache(
            cache[best_size_A["mask_threshold"]],
            filter_fn=lambda c, ts=best_size_A["score_thresh_small"], tm=best_size_A["score_thresh_med"], tl=best_size_A["score_thresh_large"]: (
                c["cls_score"] >= ts if c["size_bin"] == "small"
                else (c["cls_score"] >= tm if c["size_bin"] == "medium" else c["cls_score"] >= tl)
            ),
            split_obs=obs_set_B,
        )
        eval_size_B_on_A = evaluate_from_cache(
            cache[best_size_B["mask_threshold"]],
            filter_fn=lambda c, ts=best_size_B["score_thresh_small"], tm=best_size_B["score_thresh_med"], tl=best_size_B["score_thresh_large"]: (
                c["cls_score"] >= ts if c["size_bin"] == "small"
                else (c["cls_score"] >= tm if c["size_bin"] == "medium" else c["cls_score"] >= tl)
            ),
            split_obs=obs_set_A,
        )
        logger.info(f"Size-bin sweep finished in {time.time() - t_size:.2f}s")

    # ──────────────── WRITE CSV ────────────────
    sweep_df = pd.DataFrame(all_sweep_records)
    sweep_df.to_csv(args.out_csv, index=False)
    logger.info(f"Saved full extended sweep results to '{args.out_csv}' ({len(sweep_df)} evaluations)")

    # ──────────────── WRITE JSON ────────────────
    # Standard sweep best is preserved as default baseline
    out_payload = {
        "best_score_threshold": best_std_full["score_threshold"],
        "best_mask_threshold": best_std_full["mask_threshold"],
        "best_pq": best_std_full["pq"],
        "sq": best_std_full["sq"],
        "rq": best_std_full["rq"],
        "tp": best_std_full["tp"],
        "fp": best_std_full["fp"],
        "fn": best_std_full["fn"],
        "gt_multi_hit": best_std_full["gt_multi_hit"],
        "pred_multi_hit": best_std_full["pred_multi_hit"],
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "img_size": args.img_size,
        "tuning_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_bias_control": {
            "seed": args.seed,
            "standard": {
                "tune_A": {"mask_th": best_std_A["mask_threshold"], "score_th": best_std_A["score_threshold"], "pq_same_set": best_std_A["pq"]},
                "eval_A_on_B": {"pq_held_out": eval_std_A_on_B["pq"], "sq": eval_std_A_on_B["sq"], "rq": eval_std_A_on_B["rq"]},
                "tune_B": {"mask_th": best_std_B["mask_threshold"], "score_th": best_std_B["score_threshold"], "pq_same_set": best_std_B["pq"]},
                "eval_B_on_A": {"pq_held_out": eval_std_B_on_A["pq"], "sq": eval_std_B_on_A["sq"], "rq": eval_std_B_on_A["rq"]},
            }
        }
    }

    if args.rescore and best_rescore_full is not None:
        out_payload["selection_bias_control"]["rescore"] = {
            "tune_A": {
                "mask_th": best_rescore_A["mask_threshold"],
                "score_th": best_rescore_A["score_threshold"],
                "gamma": best_rescore_A["gamma"],
                "pq_same_set": best_rescore_A["pq"],
            },
            "eval_A_on_B": {
                "pq_held_out": eval_rescore_A_on_B["pq"],
                "sq": eval_rescore_A_on_B["sq"],
                "rq": eval_rescore_A_on_B["rq"],
                "delta_vs_std": eval_rescore_A_on_B["pq"] - eval_std_A_on_B["pq"],
            },
            "tune_B": {
                "mask_th": best_rescore_B["mask_threshold"],
                "score_th": best_rescore_B["score_threshold"],
                "gamma": best_rescore_B["gamma"],
                "pq_same_set": best_rescore_B["pq"],
            },
            "eval_B_on_A": {
                "pq_held_out": eval_rescore_B_on_A["pq"],
                "sq": eval_rescore_B_on_A["sq"],
                "rq": eval_rescore_B_on_A["rq"],
                "delta_vs_std": eval_rescore_B_on_A["pq"] - eval_std_B_on_A["pq"],
            },
        }
        out_payload["rescore_best"] = best_rescore_full

    if args.size_bins and best_size_full is not None:
        out_payload["selection_bias_control"]["size_bins"] = {
            "bin_edges_p33_p66": [bin_edges[0], bin_edges[1]] if bin_edges else None,
            "tune_A": {
                "mask_th": best_size_A["mask_threshold"],
                "thresh_small": best_size_A["score_thresh_small"],
                "thresh_med": best_size_A["score_thresh_med"],
                "thresh_large": best_size_A["score_thresh_large"],
                "pq_same_set": best_size_A["pq"],
            },
            "eval_A_on_B": {
                "pq_held_out": eval_size_A_on_B["pq"],
                "sq": eval_size_A_on_B["sq"],
                "rq": eval_size_A_on_B["rq"],
                "delta_vs_std": eval_size_A_on_B["pq"] - eval_std_A_on_B["pq"],
            },
            "tune_B": {
                "mask_th": best_size_B["mask_threshold"],
                "thresh_small": best_size_B["score_thresh_small"],
                "thresh_med": best_size_B["score_thresh_med"],
                "thresh_large": best_size_B["score_thresh_large"],
                "pq_same_set": best_size_B["pq"],
            },
            "eval_B_on_A": {
                "pq_held_out": eval_size_B_on_A["pq"],
                "sq": eval_size_B_on_A["sq"],
                "rq": eval_size_B_on_A["rq"],
                "delta_vs_std": eval_size_B_on_A["pq"] - eval_std_B_on_A["pq"],
            },
        }
        out_payload["size_bins_best"] = best_size_full

    with open(args.out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    logger.info(f"Saved configuration to '{args.out_json}'")

    # ──────────────── 4. CLEAN ASCII TABLES ────────────────

    # Table 1: Standard Sweep - Top Configurations per Mask Threshold
    std_rows = []
    # Find best score for each mask_th in standard sweep
    for m_th in mask_thresholds:
        sub = [r for r in records_std if r["mask_threshold"] == m_th]
        sub.sort(key=lambda x: x["pq"], reverse=True)
        top = sub[0]
        std_rows.append([
            f"{top['mask_threshold']:.2f}",
            f"{top['score_threshold']:.2f}",
            f"{top['pq']:.4f}",
            f"{top['sq']:.4f}",
            f"{top['rq']:.4f}",
            f"{top['tp']}",
            f"{top['fp']}",
            f"{top['fn']}",
            f"{top['gt_multi_hit']}",
            f"{top['pred_multi_hit']}",
        ])
    print_ascii_table(
        title="STANDARD SWEEP: Best Score Threshold per Mask Threshold (Fold 0)",
        headers=["MaskTh", "ScoreTh", "PQ", "SQ", "RQ", "TP", "FP", "FN", "gt_multi", "pred_multi"],
        rows=std_rows,
    )

    # Table 2: Rescoring Variant (if run)
    if args.rescore and best_rescore_full is not None:
        rescore_rows = []
        for gamma in [0.3, 0.5, 0.7]:
            sub = [r for r in records_rescore if r["gamma"] == gamma]
            sub.sort(key=lambda x: x["pq"], reverse=True)
            top = sub[0]
            rescore_rows.append([
                f"{gamma:.1f}",
                f"{top['mask_threshold']:.2f}",
                f"{top['score_threshold']:.2f}",
                f"{top['pq']:.4f}",
                f"{top['sq']:.4f}",
                f"{top['rq']:.4f}",
                f"{top['tp']}",
                f"{top['fp']}",
                f"{top['fn']}",
                f"{top['gt_multi_hit']}",
                f"{top['pred_multi_hit']}",
            ])
        print_ascii_table(
            title="MASKNESS RESCORING: Best Thresholds per Gamma (Fold 0)",
            headers=["Gamma", "MaskTh", "ScoreTh", "PQ", "SQ", "RQ", "TP", "FP", "FN", "gt_multi", "pred_multi"],
            rows=rescore_rows,
        )

    # Table 3: Size-Bins Variant (if run)
    if args.size_bins and best_size_full is not None:
        size_rows = []
        for r in records_size:
            size_rows.append([
                f"{r['mask_threshold']:.2f}",
                f"{r['score_thresh_small']:.2f}",
                f"{r['score_thresh_med']:.2f}",
                f"{r['score_thresh_large']:.2f}",
                f"{r['pq']:.4f}",
                f"{r['sq']:.4f}",
                f"{r['rq']:.4f}",
                f"{r['tp']}",
                f"{r['fp']}",
                f"{r['fn']}",
                f"{r['gt_multi_hit']}",
                f"{r['pred_multi_hit']}",
            ])
        print_ascii_table(
            title=f"SIZE-BIN THRESHOLDS: Coordinate Descent (P33={bin_edges[0]:.1f}px, P66={bin_edges[1]:.1f}px)",
            headers=["MaskTh", "Th_Small", "Th_Med", "Th_Large", "PQ", "SQ", "RQ", "TP", "FP", "FN", "gt_multi", "pred_multi"],
            rows=size_rows,
        )

    # Table 4: Selection Bias Control & Held-Out Half Generalization Summary
    summary_headers = [
        "Method", "Best Full Config", "Full PQ",
        "Tune A -> Eval B (PQ)", "Tune B -> Eval A (PQ)",
        "Delta A->B", "Delta B->A", "Pass > +0.01 Both?"
    ]
    summary_rows = []

    # Standard
    cfg_std = f"mask={best_std_full['mask_threshold']:.2f}, sc={best_std_full['score_threshold']:.2f}"
    std_a_to_b = eval_std_A_on_B["pq"]
    std_b_to_a = eval_std_B_on_A["pq"]
    summary_rows.append([
        "Standard Sweep",
        cfg_std,
        f"{best_std_full['pq']:.4f}",
        f"{std_a_to_b:.4f}",
        f"{std_b_to_a:.4f}",
        "---", "---",
        "BASELINE",
    ])

    # Rescore
    if args.rescore and best_rescore_full is not None:
        cfg_rescore = (
            f"g={best_rescore_full['gamma']:.1f}, "
            f"mask={best_rescore_full['mask_threshold']:.2f}, "
            f"sc={best_rescore_full['score_threshold']:.2f}"
        )
        res_a_to_b = eval_rescore_A_on_B["pq"]
        res_b_to_a = eval_rescore_B_on_A["pq"]
        d_a_b = res_a_to_b - std_a_to_b
        d_b_a = res_b_to_a - std_b_to_a
        passes = "YES" if (d_a_b > 0.01 and d_b_a > 0.01) else "NO"
        summary_rows.append([
            "Mask Rescoring",
            cfg_rescore,
            f"{best_rescore_full['pq']:.4f}",
            f"{res_a_to_b:.4f}",
            f"{res_b_to_a:.4f}",
            f"{d_a_b:+.4f}",
            f"{d_b_a:+.4f}",
            passes,
        ])

    # Size Bins
    if args.size_bins and best_size_full is not None:
        cfg_size = (
            f"mask={best_size_full['mask_threshold']:.2f}, "
            f"({best_size_full['score_thresh_small']:.2f}/{best_size_full['score_thresh_med']:.2f}/{best_size_full['score_thresh_large']:.2f})"
        )
        sz_a_to_b = eval_size_A_on_B["pq"]
        sz_b_to_a = eval_size_B_on_A["pq"]
        d_a_b = sz_a_to_b - std_a_to_b
        d_b_a = sz_b_to_a - std_b_to_a
        passes = "YES" if (d_a_b > 0.01 and d_b_a > 0.01) else "NO"
        summary_rows.append([
            "Size Bins",
            cfg_size,
            f"{best_size_full['pq']:.4f}",
            f"{sz_a_to_b:.4f}",
            f"{sz_b_to_a:.4f}",
            f"{d_a_b:+.4f}",
            f"{d_b_a:+.4f}",
            passes,
        ])

    print_ascii_table(
        title="SELECTION BIAS CONTROL: 2-Way Held-Out Generalization (A/B Split)",
        headers=summary_headers,
        rows=summary_rows,
    )

    logger.info("Threshold optimization complete.")


if __name__ == "__main__":
    main()
