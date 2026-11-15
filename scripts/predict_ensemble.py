#!/usr/bin/env python3
"""
scripts/predict_ensemble.py - Test-Time Augmentation (Flip TTA) & Weighted Mask Fusion (WMF).

Features:
1. 4-View Flip TTA (M = 4):
   - identity: original image
   - h: horizontal flip
   - v: vertical flip
   - hv: horizontal + vertical flip (180° rotation equivalent)
   Predictions (boxes, scores, soft mask probability crops) are un-flipped back to original coordinates.
2. Weighted Mask Fusion (WMF):
   - Greedy clustering by mask IoU > 0.40 on union bbox crops.
   - Cluster score: S = sum(s_k) / M (where M = 4, naturally penalizing single-view false positives).
   - Fused soft mask probability: P_fused = sum(s_k * P_k) / sum(s_k).
3. Disjoint Postprocessing:
   - Thresholds P_fused >= tau_mask.
   - Filters clusters with S >= tau_score.
   - Applies disjoint canvas (score desc, subtract stronger masks, drop if < 50% retained or < min_area).
4. Selection Bias Control (A/B split):
   - Splits validation set into Halves A and B with seed 2026.
   - Tunes (tau_score, tau_mask) on A -> evaluates on B; tunes on B -> evaluates on A.
5. Paired bootstrap comparison against no-TTA baseline with 95% confidence interval.
6. CPU unit testing enabled.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.bootstrap_compare import run_paired_bootstrap
from scripts.common import (
    EMPTY_RLE,
    extract_state_dict,
    find_dataset_paths,
    get_maskrcnn_model,
    get_run_dir,
    preprocess_solar_image,
)
from scripts.predict import load_image

FRAME_SIZE = 2048


# ─────────────────────────────── TTA FLIP TRANSFORMS ─────────────────────────

def apply_flip_transform(img: torch.Tensor, flip_v: bool, flip_h: bool) -> torch.Tensor:
    """Applies horizontal / vertical flip to a 3D tensor [C, H, W]."""
    out = img
    if flip_v:
        out = torch.flip(out, dims=[1])
    if flip_h:
        out = torch.flip(out, dims=[2])
    return out


def unflip_boxes(boxes: torch.Tensor, flip_v: bool, flip_h: bool, height: int, width: int) -> torch.Tensor:
    """
    Inverts flip transformations on bounding boxes [N, 4] with format [x1, y1, x2, y2].
    """
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    if flip_h:
        x1 = out[:, 0].clone()
        x2 = out[:, 2].clone()
        out[:, 0] = width - x2
        out[:, 2] = width - x1
    if flip_v:
        y1 = out[:, 1].clone()
        y2 = out[:, 3].clone()
        out[:, 1] = height - y2
        out[:, 3] = height - y1
    return out


def unflip_mask_probabilities(masks: torch.Tensor, flip_v: bool, flip_h: bool) -> torch.Tensor:
    """Inverts flip transformations on mask probability maps [N, H, W] or [N, 1, H, W]."""
    if masks.numel() == 0:
        return masks
    out = masks
    if flip_v:
        out = torch.flip(out, dims=[-2])
    if flip_h:
        out = torch.flip(out, dims=[-1])
    return out


# ──────────────────────── WEIGHTED MASK FUSION (WMF) ─────────────────────────

def compute_crop_mask_iou(
    c1: dict,
    c2: dict,
    mask_thresh: float = 0.5,
) -> float:
    """
    Computes mask IoU between two candidate bounding-box crops on their union bbox.
    c1, c2: dicts containing 'box' [x1, y1, x2, y2] and 'prob_crop' [H_crop, W_crop].
    """
    b1 = c1["box"]
    b2 = c2["box"]
    ux1 = min(b1[0], b2[0])
    uy1 = min(b1[1], b2[1])
    ux2 = max(b1[2], b2[2])
    uy2 = max(b1[3], b2[3])

    if ux2 <= ux1 or uy2 <= uy1:
        return 0.0

    uw = ux2 - ux1
    uh = uy2 - uy1

    m1 = np.zeros((uh, uw), dtype=bool)
    m2 = np.zeros((uh, uw), dtype=bool)

    bin1 = (c1["prob_crop"] >= mask_thresh)
    m1[b1[1] - uy1 : b1[3] - uy1, b1[0] - ux1 : b1[2] - ux1] = bin1

    bin2 = (c2["prob_crop"] >= mask_thresh)
    m2[b2[1] - uy1 : b2[3] - uy1, b2[0] - ux1 : b2[2] - ux1] = bin2

    inter = int(np.logical_and(m1, m2).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(m1, m2).sum())
    return inter / union if union > 0 else 0.0


def weighted_mask_fusion(
    candidate_list: List[dict],
    num_views: int = 4,
    iou_thresh: float = 0.40,
) -> List[dict]:
    """
    Greedy Weighted Mask Fusion across multi-view candidate detections.

    Each candidate dict:
      - 'box': np.ndarray [4] (x1, y1, x2, y2)
      - 'score': float (s_k)
      - 'prob_crop': np.ndarray (H_crop, W_crop) soft mask probabilities
      - 'view_idx': int

    Returns list of fused cluster dicts:
      - 'score': S = sum(s_k) / M
      - 'prob_crop': sum(s_k * P_k) / sum(s_k) within union bbox
      - 'box': union bounding box [ux1, uy1, ux2, uy2]
      - 'cluster_size': count of constituent candidates
    """
    if not candidate_list:
        return []

    sorted_candidates = sorted(candidate_list, key=lambda c: c["score"], reverse=True)
    clusters: List[List[dict]] = []

    for cand in sorted_candidates:
        best_cluster_idx = -1
        best_iou = -1.0
        for c_idx, cluster in enumerate(clusters):
            leader = cluster[0]
            iou = compute_crop_mask_iou(cand, leader, mask_thresh=0.5)
            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = c_idx

        if best_iou > iou_thresh:
            clusters[best_cluster_idx].append(cand)
        else:
            clusters.append([cand])

    fused_clusters = []
    for cluster in clusters:
        scores = [c["score"] for c in cluster]
        score_sum = sum(scores)
        cluster_score = score_sum / float(num_views)

        all_boxes = np.stack([c["box"] for c in cluster], axis=0)
        ux1 = int(np.floor(all_boxes[:, 0].min()))
        uy1 = int(np.floor(all_boxes[:, 1].min()))
        ux2 = int(np.ceil(all_boxes[:, 2].max()))
        uy2 = int(np.ceil(all_boxes[:, 3].max()))
        union_box = np.array([ux1, uy1, ux2, uy2], dtype=np.int32)

        uh = uy2 - uy1
        uw = ux2 - ux1
        fused_prob_crop = np.zeros((uh, uw), dtype=np.float32)

        for c in cluster:
            w = c["score"] / score_sum
            b = c["box"]
            fused_prob_crop[b[1] - uy1 : b[3] - uy1, b[0] - ux1 : b[2] - ux1] += w * c["prob_crop"]

        fused_clusters.append({
            "score": float(cluster_score),
            "prob_crop": fused_prob_crop,
            "box": union_box,
            "cluster_size": len(cluster),
            "member_scores": [float(s) for s in scores],
        })

    fused_clusters.sort(key=lambda x: x["score"], reverse=True)
    return fused_clusters


# ──────────────────────── EVALUATION & THRESHOLD SWEEPER ─────────────────────

def generate_predictions_for_threshold(
    image_clusters: Dict[str, List[dict]],
    tau_score: float,
    tau_mask: float,
    min_area: int = 10,
    area_retention: float = 0.5,
) -> pd.DataFrame:
    """
    Generates a postprocessed predictions DataFrame for specific (tau_score, tau_mask).
    """
    rows = []
    for stem, clusters in image_clusters.items():
        kept_clusters = [c for c in clusters if c["score"] >= tau_score]
        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        kept_rles = []

        for c in kept_clusters:
            b = c["box"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            binary_crop = (c["prob_crop"] >= tau_mask)
            orig_area = int(binary_crop.sum())
            if orig_area == 0:
                continue

            sub_occ = occupied[y1:y2, x1:x2]
            clean_crop = binary_crop & (~sub_occ)
            clean_area = int(clean_crop.sum())

            if clean_area < min_area or (clean_area / orig_area) < area_retention:
                continue

            occupied[y1:y2, x1:x2] |= clean_crop

            full_mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8, order="F")
            full_mask[y1:y2, x1:x2] = clean_crop.astype(np.uint8)
            enc_rle = mu.encode(full_mask)["counts"].decode("ascii")
            kept_rles.append(enc_rle)

        if not kept_rles:
            rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(kept_rles, start=1):
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

    return pd.DataFrame(rows)


def run_ensemble_ab_sweep(
    image_clusters: Dict[str, List[dict]],
    gt_df: pd.DataFrame,
    seed: int = 2026,
    score_threshs: Optional[List[float]] = None,
    mask_threshs: Optional[List[float]] = None,
) -> dict:
    """
    Evaluates 2-way held-out A/B generalization over a grid of (tau_score, tau_mask).
    """
    if score_threshs is None:
        score_threshs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    if mask_threshs is None:
        mask_threshs = [0.45, 0.50, 0.55, 0.60, 0.65]

    all_stems = sorted(list(image_clusters.keys()))
    rng = random.Random(seed)
    shuffled_stems = all_stems.copy()
    rng.shuffle(shuffled_stems)
    half_idx = len(shuffled_stems) // 2
    stems_A = set(shuffled_stems[:half_idx])
    stems_B = set(shuffled_stems[half_idx:])

    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_stem = gt_ai.str.split("-", n=1).str[1]
    gt_A = gt_df[gt_stem.isin(stems_A)].copy()
    gt_B = gt_df[gt_stem.isin(stems_B)].copy()

    clusters_A = {s: image_clusters[s] for s in stems_A}
    clusters_B = {s: image_clusters[s] for s in stems_B}

    best_A = {"pq": -1.0, "tau_score": None, "tau_mask": None}
    best_B = {"pq": -1.0, "tau_score": None, "tau_mask": None}
    best_full = {"pq": -1.0, "tau_score": None, "tau_mask": None}

    sweep_results = []
    print(f"Sweeping grid of {len(score_threshs)} score x {len(mask_threshs)} mask thresholds...")

    for sc in score_threshs:
        for mk in mask_threshs:
            pred_full = generate_predictions_for_threshold(image_clusters, sc, mk)
            res_full = pq_eval.score(gt_df, pred_full[pred_full["segmentation_rle"] != EMPTY_RLE])

            pred_A = generate_predictions_for_threshold(clusters_A, sc, mk)
            res_A = pq_eval.score(gt_A, pred_A[pred_A["segmentation_rle"] != EMPTY_RLE])

            pred_B = generate_predictions_for_threshold(clusters_B, sc, mk)
            res_B = pq_eval.score(gt_B, pred_B[pred_B["segmentation_rle"] != EMPTY_RLE])

            if res_full["pq"] > best_full["pq"]:
                best_full = {"pq": res_full["pq"], "tau_score": sc, "tau_mask": mk, "metrics": res_full}
            if res_A["pq"] > best_A["pq"]:
                best_A = {"pq": res_A["pq"], "tau_score": sc, "tau_mask": mk}
            if res_B["pq"] > best_B["pq"]:
                best_B = {"pq": res_B["pq"], "tau_score": sc, "tau_mask": mk}

            sweep_results.append({
                "tau_score": sc,
                "tau_mask": mk,
                "pq_full": res_full["pq"],
                "pq_A": res_A["pq"],
                "pq_B": res_B["pq"],
            })

    # Cross evaluation: A tuned on B, B tuned on A
    pred_A_on_B = generate_predictions_for_threshold(clusters_B, best_A["tau_score"], best_A["tau_mask"])
    eval_A_on_B = pq_eval.score(gt_B, pred_A_on_B[pred_A_on_B["segmentation_rle"] != EMPTY_RLE])

    pred_B_on_A = generate_predictions_for_threshold(clusters_A, best_B["tau_score"], best_B["tau_mask"])
    eval_B_on_A = pq_eval.score(gt_A, pred_B_on_A[pred_B_on_A["segmentation_rle"] != EMPTY_RLE])

    return {
        "best_full": best_full,
        "best_A": best_A,
        "best_B": best_B,
        "eval_A_on_B": eval_A_on_B,
        "eval_B_on_A": eval_B_on_A,
        "sweep_table": sweep_results,
    }


# ──────────────────────── FULL FOLD 0 EVALUATION PIPELINE ─────────────────────

def run_fold0_evaluation(
    checkpoint_path: str,
    img_size: int = 1536,
    fold: int = 0,
    n_boot: int = 1000,
    seed: int = 2026,
    model_version: Optional[str] = None,
    device_str: str = "cuda:0",
):
    print("==================================================================")
    print("   FLIP TTA + WEIGHTED MASK FUSION: FOLD 0 FULL EVALUATION        ")
    print("==================================================================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image size: {img_size}")
    print(f"Fold:       {fold}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    val_obs = folds_df.loc[folds_df["fold"] == fold, "observation"].tolist()
    train_dir = paths["train_dir"]

    val_image_paths = []
    for o in val_obs:
        p = os.path.join(train_dir, f"{o}.jpeg")
        if not os.path.exists(p):
            p = os.path.join(train_dir, f"{o}.jpg")
        val_image_paths.append((o, p))

    gt_df = pd.read_csv(paths["gt_all_csv"])
    val_obs_set = set(val_obs)
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]
    val_gt = gt_df[gt_img.isin(val_obs_set)].copy()

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    ckpt_anchors5 = config.get("anchors5", False)
    ckpt_version = model_version or config.get("model_version", "v2")
    print(f"Model architecture: Mask R-CNN {ckpt_version}, anchors5={ckpt_anchors5}")
    model = get_maskrcnn_model(
        pretrained=False,
        min_size=img_size,
        max_size=img_size,
        num_classes=2,
        anchors5=ckpt_anchors5,
        model_version=ckpt_version,
    )
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device).eval()

    # 4 TTA Views: (flip_v, flip_h)
    views = [
        ("id", False, False),
        ("h", False, True),
        ("v", True, False),
        ("hv", True, True),
    ]

    scale_to_2048 = 2048.0 / img_size
    fused_clusters_by_stem = {}
    no_tta_candidates_by_stem = {}

    t0 = time.time()
    print(f"\nRunning 4-view Flip TTA inference on {len(val_image_paths)} images...")

    with torch.no_grad():
        for img_idx, (obs, img_path) in enumerate(val_image_paths, start=1):
            if img_idx % 25 == 0 or img_idx == len(val_image_paths):
                print(f"  Processed {img_idx}/{len(val_image_paths)} images ({time.time() - t0:.1f}s)...")

            # Load base image and preprocess once
            base_tensor, _ = load_image(img_path, img_size=img_size)
            all_view_candidates = []

            for v_idx, (v_name, f_v, f_h) in enumerate(views):
                # Apply flip
                flipped_tensor = apply_flip_transform(base_tensor, flip_v=f_v, flip_h=f_h).to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output = model([flipped_tensor])[0]

                scores = output["scores"]
                keep = scores >= 0.25
                if not keep.any():
                    continue

                scores = scores[keep]
                boxes = output["boxes"][keep]
                masks = output["masks"][keep]  # [K, 1, H, W]

                # Unflip
                boxes_unflipped = unflip_boxes(boxes, flip_v=f_v, flip_h=f_h, height=img_size, width=img_size)
                masks_unflipped = unflip_mask_probabilities(masks, flip_v=f_v, flip_h=f_h)

                # Scale boxes to 2048
                boxes_2048 = (boxes_unflipped * scale_to_2048).detach().cpu().numpy()
                scores_np = scores.detach().cpu().numpy()

                # Upsample masks to 2048 if needed
                if img_size == 2048:
                    masks_up = masks_unflipped.squeeze(1).detach().cpu().numpy()
                else:
                    masks_up = F.interpolate(
                        masks_unflipped.float(), size=(2048, 2048), mode="bilinear", align_corners=False
                    ).squeeze(1).detach().cpu().numpy()

                for k in range(len(scores_np)):
                    b = boxes_2048[k]
                    x1 = max(0, int(np.floor(b[0])))
                    y1 = max(0, int(np.floor(b[1])))
                    x2 = min(2048, int(np.ceil(b[2])) + 1)
                    y2 = min(2048, int(np.ceil(b[3])) + 1)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float32)

                    cand = {
                        "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                        "score": float(scores_np[k]),
                        "prob_crop": crop_soft,
                        "view_idx": v_idx,
                    }
                    all_view_candidates.append(cand)

                    if v_idx == 0:  # Identity view (no-TTA)
                        if obs not in no_tta_candidates_by_stem:
                            no_tta_candidates_by_stem[obs] = []
                        no_tta_candidates_by_stem[obs].append(cand)

            # Fuse across the 4 views for this image
            fused = weighted_mask_fusion(all_view_candidates, num_views=4, iou_thresh=0.40)
            fused_clusters_by_stem[obs] = fused
            del all_view_candidates
            if img_idx % 10 == 0:
                torch.cuda.empty_cache()

    print(f"\nInference & Fusion completed in {time.time() - t0:.1f}s.")

    # 1. No-TTA Baseline Evaluation
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    no_tta_csv = os.path.join(run_dir, "pred_fold0_score0.85_mask0.55.csv")
    if not os.path.isfile(no_tta_csv):
        print("Generating no-TTA baseline predictions from identity view...")
        fused_no_tta = {s: weighted_mask_fusion(cands, num_views=1, iou_thresh=0.40) for s, cands in no_tta_candidates_by_stem.items()}
        for s in val_obs:
            if s not in fused_no_tta:
                fused_no_tta[s] = []
        df_no_tta = generate_predictions_for_threshold(fused_no_tta, tau_score=0.85, tau_mask=0.55)
        df_no_tta.to_csv(no_tta_csv, index=False)
        print(f"Saved generated no-TTA baseline to {no_tta_csv}")

    res_no_tta = pq_eval.score(val_gt, pd.read_csv(no_tta_csv))
    print("\n------------------------------------------------------------------")
    print(f"No-TTA Baseline (Score 0.85, Mask 0.55):")
    print(f"PQ: {res_no_tta['pq']:.4f} | SQ: {res_no_tta['sq']:.4f} | RQ: {res_no_tta['rq']:.4f}")
    print(f"TP: {res_no_tta['tp']} | FP: {res_no_tta['fp']} | FN: {res_no_tta['fn']}")
    print("------------------------------------------------------------------")

    # 2. TTA + WMF Threshold Sweep
    sweep_out = run_ensemble_ab_sweep(fused_clusters_by_stem, val_gt, seed=seed)
    best_full = sweep_out["best_full"]
    best_sc = best_full["tau_score"]
    best_mk = best_full["tau_mask"]
    print(f"\nOptimal TTA+WMF Thresholds (Full): tau_score = {best_sc:.2f}, tau_mask = {best_mk:.2f}")
    print(f"Optimal TTA+WMF PQ: {best_full['pq']:.4f}")
    print(f"A/B Generalization: Half A optimal ({sweep_out['best_A']['tau_score']}, {sweep_out['best_A']['tau_mask']}) on B -> PQ {sweep_out['eval_A_on_B']['pq']:.4f}")
    print(f"A/B Generalization: Half B optimal ({sweep_out['best_B']['tau_score']}, {sweep_out['best_B']['tau_mask']}) on A -> PQ {sweep_out['eval_B_on_A']['pq']:.4f}")

    # Generate and save TTA+WMF predictions
    df_tta = generate_predictions_for_threshold(fused_clusters_by_stem, best_sc, best_mk)
    tta_csv = os.path.join(run_dir, "pred_fold0_tta_wmf.csv")
    df_tta.to_csv(tta_csv, index=False)
    print(f"Saved TTA+WMF predictions to: {tta_csv}")

    # 3. Paired Bootstrap Comparison
    print(f"\nRunning Paired Bootstrap Comparison (1000 resamples)...")
    boot_res = run_paired_bootstrap(
        pred_a_path=no_tta_csv,
        pred_b_path=tta_csv,
        gt_path=paths["gt_all_csv"],
        folds_path=paths["folds_csv"],
        fold=fold,
        n_boot=n_boot,
        seed=seed,
    )

    print("\n==================================================================")
    print("                  FINAL COMPARISON RESULTS                         ")
    print("==================================================================")
    print(f"Baseline (no-TTA) PQ:      {boot_res['full_a']['pq']:.4f}")
    print(f"Ensemble (Flip TTA+WMF) PQ:{boot_res['full_b']['pq']:.4f}")
    print(f"Point Delta PQ:            {boot_res['point_delta']:+.4f}")
    print(f"Bootstrap Mean Delta PQ:   {boot_res['mean_delta']:+.4f} (std: {boot_res['std_delta']:.4f})")
    print(f"95% Confidence Interval:   [{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}]")
    print(f"P(TTA+WMF > Baseline):     {boot_res['p_b_better'] * 100:.1f}%")
    print("==================================================================")

    # Save summary JSON
    summary_path = os.path.join(run_dir, "tta_wmf_eval.json")
    out_dict = {
        "checkpoint": checkpoint_path,
        "img_size": img_size,
        "fold": fold,
        "no_tta": {
            "pq": boot_res["full_a"]["pq"],
            "sq": boot_res["full_a"]["sq"],
            "rq": boot_res["full_a"]["rq"],
            "tp": boot_res["full_a"]["tp"],
            "fp": boot_res["full_a"]["fp"],
            "fn": boot_res["full_a"]["fn"],
        },
        "tta_wmf": {
            "best_tau_score": best_sc,
            "best_tau_mask": best_mk,
            "pq": boot_res["full_b"]["pq"],
            "sq": boot_res["full_b"]["sq"],
            "rq": boot_res["full_b"]["rq"],
            "tp": boot_res["full_b"]["tp"],
            "fp": boot_res["full_b"]["fp"],
            "fn": boot_res["full_b"]["fn"],
            "eval_A_on_B": sweep_out["eval_A_on_B"]["pq"],
            "eval_B_on_A": sweep_out["eval_B_on_A"]["pq"],
        },
        "bootstrap": {
            "point_delta": boot_res["point_delta"],
            "mean_delta": boot_res["mean_delta"],
            "std_delta": boot_res["std_delta"],
            "ci_lower": boot_res["ci_lower"],
            "ci_upper": boot_res["ci_upper"],
            "p_b_better": boot_res["p_b_better"],
        },
    }
    with open(summary_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved evaluation summary to: {summary_path}")


# ──────────────────────── CPU UNIT TESTS ─────────────────────────────────────

def run_cpu_unit_tests():
    """Unit tests for Flip TTA and Weighted Mask Fusion on CPU."""
    print("\n==================================================================")
    print("      RUNNING CPU UNIT TESTS FOR FLIP TTA & WMF ENSEMBLE          ")
    print("==================================================================")

    H, W = 100, 120
    box_orig = torch.tensor([[20.0, 30.0, 60.0, 70.0]])  # x1, y1, x2, y2

    # Horizontal flip
    box_h = unflip_boxes(box_orig, flip_v=False, flip_h=True, height=H, width=W)
    assert box_h[0, 0] == W - 60.0 and box_h[0, 2] == W - 20.0
    box_h_restored = unflip_boxes(box_h, flip_v=False, flip_h=True, height=H, width=W)
    assert torch.allclose(box_h_restored, box_orig)
    print("[PASS] Unit Test 1A: Horizontal box flip and un-flip")

    # Vertical flip
    box_v = unflip_boxes(box_orig, flip_v=True, flip_h=False, height=H, width=W)
    assert box_v[0, 1] == H - 70.0 and box_v[0, 3] == H - 30.0
    box_v_restored = unflip_boxes(box_v, flip_v=True, flip_h=False, height=H, width=W)
    assert torch.allclose(box_v_restored, box_orig)
    print("[PASS] Unit Test 1B: Vertical box flip and un-flip")

    # Both flips (hv)
    box_hv = unflip_boxes(box_orig, flip_v=True, flip_h=True, height=H, width=W)
    box_hv_restored = unflip_boxes(box_hv, flip_v=True, flip_h=True, height=H, width=W)
    assert torch.allclose(box_hv_restored, box_orig)
    print("[PASS] Unit Test 1C: Both-flip (HV) box flip and un-flip")

    # Test 2: Mask un-flipping
    m_test = torch.zeros((1, H, W))
    m_test[0, 30:70, 20:60] = 1.0
    m_unflipped = unflip_mask_probabilities(
        unflip_mask_probabilities(m_test, flip_v=True, flip_h=True),
        flip_v=True, flip_h=True
    )
    assert torch.allclose(m_unflipped, m_test)
    print("[PASS] Unit Test 2: Mask probability un-flip restoration")

    # Test 3: Weighted Mask Fusion (WMF) clustering and scoring
    crop_dummy = np.zeros((40, 40), dtype=np.float32)
    crop_dummy.fill(0.9)
    box_np = np.array([20, 30, 60, 70], dtype=np.int32)

    cands_4views = [
        {"box": box_np, "score": 0.8, "prob_crop": crop_dummy.copy(), "view_idx": i}
        for i in range(4)
    ]
    fused_4v = weighted_mask_fusion(cands_4views, num_views=4, iou_thresh=0.40)
    assert len(fused_4v) == 1, f"Expected 1 cluster, got {len(fused_4v)}"
    assert abs(fused_4v[0]["score"] - 0.8) < 1e-5, f"Expected score 0.8, got {fused_4v[0]['score']}"
    assert fused_4v[0]["cluster_size"] == 4
    print("[PASS] Unit Test 3A: WMF merges 4-view detection into single cluster with S = 0.8")

    # Solitary detection
    cands_1view = [{"box": box_np, "score": 0.8, "prob_crop": crop_dummy.copy(), "view_idx": 0}]
    fused_1v = weighted_mask_fusion(cands_1view, num_views=4, iou_thresh=0.40)
    assert len(fused_1v) == 1
    assert abs(fused_1v[0]["score"] - 0.2) < 1e-5, f"Expected penalized score 0.2, got {fused_1v[0]['score']}"
    print(f"[PASS] Unit Test 3B: WMF penalizes single-view hallucination S = {fused_1v[0]['score']:.2f}")

    # Two spatially separated detections
    crop_sep = np.zeros((15, 15), dtype=np.float32)
    crop_sep.fill(0.85)
    box_sep = np.array([80, 80, 95, 95], dtype=np.int32)

    cands_2obj = cands_4views + [{"box": box_sep, "score": 0.7, "prob_crop": crop_sep, "view_idx": 0}]
    fused_2obj = weighted_mask_fusion(cands_2obj, num_views=4, iou_thresh=0.40)
    assert len(fused_2obj) == 2, f"Expected 2 separate clusters, got {len(fused_2obj)}"
    print("[PASS] Unit Test 3C: WMF keeps distinct objects in separate clusters")

    # Test 4: Threshold prediction generator, disjoint check, and empty-stem placeholder
    image_clusters = {"test_stem": fused_2obj, "empty_stem": []}
    pred_df = generate_predictions_for_threshold(image_clusters, tau_score=0.3, tau_mask=0.5)
    assert len(pred_df) == 2
    assert pred_df.iloc[0]["filament_id"] == "test_stem_1"
    assert pred_df.iloc[1]["filament_id"] == "empty_stem_1"
    assert pred_df.iloc[1]["segmentation_rle"] == EMPTY_RLE
    print("[PASS] Unit Test 4: Threshold prediction generation, filtering, and EMPTY_RLE placeholder")

    print("\nAll CPU unit tests completed successfully!")
    print("==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Flip TTA and Weighted Mask Fusion Ensemble")
    parser.add_argument("--eval_fold0", action="store_true", help="Run full evaluation on fold 0")
    parser.add_argument("--checkpoint", type=str, default="work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt", help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=1536, help="Inference image resolution")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--model_version", type=str, default=None, help="Model version ('v1' or 'v2', default: read from ckpt or 'v2')")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--unit_test", action="store_true", help="Run CPU unit tests")
    args = parser.parse_args()

    if args.unit_test or len(sys.argv) == 1:
        run_cpu_unit_tests()
        return

    if args.eval_fold0:
        run_fold0_evaluation(
            checkpoint_path=args.checkpoint,
            img_size=args.img_size,
            fold=args.fold,
            n_boot=args.n_boot,
            model_version=args.model_version,
        )


if __name__ == "__main__":
    main()
