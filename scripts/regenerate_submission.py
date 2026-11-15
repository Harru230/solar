#!/usr/bin/env python3
"""
scripts/regenerate_submission.py - Standalone Final Submission Generator for Solar Release.

Reproduces the official 30-model multi-family and Phase 3B competition submissions:
  - Rule A: Uniform WMF (30 models, 120 views, tau_score=0.850, tau_mask=0.550) -> Public LB: 0.39
  - Rule B: Weighted WMF (30 models, 120 views, w_2048=2.0, w_1792=2.0, gamma=1.5, tau_score=0.750, tau_mask=0.550) -> Public LB: 0.40
  - Rule C: Phase 3B Purely Local Refiner + 25-Booster LightGBM Ranker (OOF [nested] PQ: 0.468754)

Supported Modes:
  1. Fast Mode (--mode fast, default):
     Streams precomputed 4-flip detections from work/test_cache/ (CPU-only, ~15-20s for Rules A/B).
  2. Full Mode (--mode full):
     Loads model checkpoints from work/checkpoints/ (or work/runs/) and executes
     4-view Flip TTA inference on test images with CPU fallback or CUDA acceleration.

Usage:
  # Reproduce Submission A (Rule A, Uniform WMF, LB 0.39)
  python scripts/regenerate_submission.py --rule A --validate

  # Reproduce Submission B (Rule B, Weighted WMF, LB 0.40)
  python scripts/regenerate_submission.py --rule B --validate

  # Reproduce Phase 3B Local Winner (Rule C, Refiners + S27 LightGBM)
  python scripts/regenerate_submission.py --rule C --mode full --validate
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

# Resolve relative paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    EMPTY_RLE,
    extract_state_dict,
    find_dataset_paths,
    get_autocast_context,
    get_maskrcnn_model,
    get_stage1_autocast_context,
    preprocess_solar_image,
)
from scripts.instance_scorer import encode_crop_to_rle
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    compute_binary_crop_iou,
    fast_crop_mask_iou,
    weighted_mask_fusion_tagged,
)
from scripts.predict_ensemble import (
    apply_flip_transform,
    unflip_boxes,
    unflip_mask_probabilities,
)


def load_ensemble_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Loads ensemble specifications from JSON configuration file."""
    if config_path is None:
        config_path = REPO_ROOT / "configs/final_ensemble.json"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Ensemble configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg


def check_ensemble_weights_available(
    config_path: Optional[Union[str, Path]] = None,
    weights_dir: Optional[Union[str, Path]] = None,
    strict_weights_only: bool = False,
) -> Tuple[bool, List[str]]:
    """Checks whether all configured ensemble checkpoints exist."""
    cfg = load_ensemble_config(config_path)
    weights_dir = Path(weights_dir) if weights_dir is not None else (REPO_ROOT / "weights")

    missing = []
    upstream_dir = os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))
    for m in cfg.get("models", []):
        p_name = m.get("pruned_name", f"{m['id']}.pt")
        rel = m.get("checkpoint_relpath", "")
        cands = [weights_dir / p_name, weights_dir / f"{m['id']}.pt"]
        if not strict_weights_only:
            cands.extend([
                REPO_ROOT / "weights" / p_name,
                REPO_ROOT / "work/checkpoints" / p_name,
                REPO_ROOT / rel,
                Path(upstream_dir) / "work/checkpoints" / p_name,
                Path(upstream_dir) / rel,
            ])
        if not any(c.is_file() for c in cands):
            missing.append(p_name)

    return len(missing) == 0, missing


# ────────────────────────── 8-MODEL ENSEMBLE SPECIFICATION & HELPERS ──────────

ENSEMBLE_8MODEL_SPECS = [
    # (cache_subdir, pruned_name, run_relpath, family, img_size, is_anchors5)
    ("f0_1536", "r50_1536_clahe_f0.pt", "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt", "clahe", 1536, False),
    ("f1_1536", "r50_1536_clahe_f1.pt", "work/runs/r50_1536_all_f1/checkpoints/checkpoint_fold1_last.pt", "clahe", 1536, False),
    ("f2_1536", "r50_1536_clahe_f2.pt", "work/runs/r50_1536_all_f2/checkpoints/checkpoint_fold2_last.pt", "clahe", 1536, False),
    ("f3_1536", "r50_1536_clahe_f3.pt", "work/runs/r50_1536_all_f3/checkpoints/checkpoint_fold3_last.pt", "clahe", 1536, False),
    ("f4_1536", "r50_1536_clahe_f4.pt", "work/runs/r50_1536_all_f4/checkpoints/checkpoint_fold4_last.pt", "clahe", 1536, False),
    ("f0_2048", "r50_2048_f0.pt", "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt", "clahe", 2048, False),
    ("f0_anchors5", "r50_1536_anchors5_f0.pt", "work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt", "a5", 1536, True),
    ("f0_dice", "r50_1536_dice_f0.pt", "work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt", "dice", 1536, False),
]

FAMILY_VIEWS = {"clahe": 24, "a5": 4, "dice": 4}
TOTAL_VIEWS = 32


def apply_threshold_decision(
    fused_clusters: List[dict],
    stem: str,
    tau_score: float = 0.875,
    tau_mask: float = 0.550,
    return_scores: bool = False,
) -> Union[List[dict], Tuple[List[dict], List[float]]]:
    """Applies pure threshold rule and disjoint canvas painting."""
    occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    rles = []
    kept_scores = []
    for c in fused_clusters:
        if c["score"] < tau_score:
            break
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bc = (c["prob_crop"] >= tau_mask)
        oa = int(bc.sum())
        if oa == 0:
            continue
        sub_occ = occupied[y1:y2, x1:x2]
        clean_crop = bc & (~sub_occ)
        ca = int(clean_crop.sum())
        if ca < 10 or (ca / float(oa)) < 0.50:
            continue
        occupied[y1:y2, x1:x2] |= clean_crop
        rles.append(encode_crop_to_rle(box, clean_crop))
        kept_scores.append(float(c["score"]))

    rows = []
    if not rles:
        rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
    else:
        for k, enc in enumerate(rles, start=1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})
    if return_scores:
        return rows, kept_scores
    return rows


def apply_scorer_v2_decision(
    fused_clusters: List[dict],
    stem: str,
    scorer_models: list,
    tau_scorer: float = 0.2440,
    tau_mask: float = 0.525,
    return_scores: bool = False,
) -> Union[List[dict], Tuple[List[dict], List[float]]]:
    """Extracts 19 view-invariant features, scores with LightGBM, and paints disjoint canvas."""
    all_scores = [float(c["score"]) for c in fused_clusters]
    max_score_in_img = max(all_scores) if all_scores else 1e-6
    n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

    feat_dicts = []
    cands_A = []
    higher_rank_boxes_and_crops = []
    sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

    for rank_0, c in enumerate(fused_clusters):
        sc = float(c["score"])
        if sc < 0.15:
            continue
        prob_crop = c["prob_crop"]
        box = c["box"]
        binary_crop = (prob_crop >= tau_mask)
        area = int(binary_crop.sum())
        if area < 10:
            continue

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        ys, xs = np.nonzero(binary_crop)
        bw = max(1, int(xs.max() - xs.min() + 1))
        bh = max(1, int(ys.max() - ys.min() + 1))
        bbox_area = float(bw * bh)

        masked_probs = prob_crop[binary_crop].astype(np.float32)
        prob_mean = float(masked_probs.mean())
        prob_std = float(masked_probs.std()) if area > 1 else 0.0
        core_pixels = int((masked_probs >= 0.75).sum())
        prob_sharpness_core = float(core_pixels) / float(area)
        q90, q10 = np.percentile(masked_probs, [90, 10])
        prob_sharpness_iqr = float(q90 - q10)

        m_scores = c.get("member_scores", [sc])
        m_arr = np.asarray(m_scores, dtype=np.float32)
        member_mean = float(m_arr.mean()) if len(m_arr) > 0 else sc
        member_std = float(m_arr.std()) if len(m_arr) > 1 else 0.0

        log_area = float(np.log10(area + 1.0))
        bbox_fill = float(area) / bbox_area
        aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        rank_in_image = rank_0 + 1
        rel_score_to_max = sc / max(1e-6, max_score_in_img)

        fam_sc = c.get("family_scores", {})
        sc_clahe = float(fam_sc.get("clahe", sc))
        sc_a5 = float(fam_sc.get("a5", sc))
        sc_dice = float(fam_sc.get("dice", sc))
        fam_arr = np.array([sc_clahe, sc_a5, sc_dice], dtype=np.float32)
        family_score_std = float(fam_arr.std())
        n_families_active = float(sum(1 for x in fam_arr if x > 0.0))

        sub_occ = sim_occupied[y1:y2, x1:x2]
        unocc_area = int((binary_crop & (~sub_occ)).sum())
        unoccluded_area_ratio = float(unocc_area) / float(area)

        max_iou_higher = 0.0
        for h_box, h_crop in higher_rank_boxes_and_crops:
            iou_h = compute_binary_crop_iou(box, binary_crop, h_box, h_crop)
            if iou_h > max_iou_higher:
                max_iou_higher = iou_h

        if sc >= 0.50 and unoccluded_area_ratio >= 0.50:
            sim_occupied[y1:y2, x1:x2] |= (binary_crop & (~sub_occ))
        higher_rank_boxes_and_crops.append((box, binary_crop))

        feat_dicts.append({
            "cluster_score": sc,
            "member_score_mean": member_mean,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "prob_sharpness_core": prob_sharpness_core,
            "prob_sharpness_iqr": prob_sharpness_iqr,
            "log_area": log_area,
            "bbox_fill": bbox_fill,
            "aspect_ratio": aspect_ratio,
            "rank_in_image": float(rank_in_image),
            "rel_score_to_max": rel_score_to_max,
            "n_cands_gt_05": float(n_cands_gt_05),
            "score_clahe": sc_clahe,
            "score_a5": sc_a5,
            "score_dice": sc_dice,
            "family_score_std": family_score_std,
            "n_families_active": n_families_active,
            "unoccluded_area_ratio": unoccluded_area_ratio,
            "max_iou_higher_rank": max_iou_higher,
        })
        cands_A.append({
            "box": box,
            "binary_crop": binary_crop,
            "orig_area": area,
            "pred_y": 0.0,
        })

    rles_A = []
    scores_A = []
    if cands_A:
        X_stem = pd.DataFrame(feat_dicts)[SCORER_V2_VIEW_INVARIANT_FEATURES].values
        pred_y_stem = np.mean([m.predict(X_stem) for m in scorer_models], axis=0)
        for c_item, py in zip(cands_A, pred_y_stem):
            c_item["pred_y"] = float(py)

        kept_A = [c_item for c_item in cands_A if c_item["pred_y"] > tau_scorer]
        kept_A.sort(key=lambda x: x["pred_y"], reverse=True)
        occupied_A = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        for c_item in kept_A:
            box = c_item["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bc = c_item["binary_crop"]
            oa = int(bc.sum())
            if oa == 0:
                continue
            sub_occ = occupied_A[y1:y2, x1:x2]
            clean_crop = bc & (~sub_occ)
            ca = int(clean_crop.sum())
            if ca < 10 or (ca / float(oa)) < 0.50:
                continue
            occupied_A[y1:y2, x1:x2] |= clean_crop
            rles_A.append(encode_crop_to_rle(box, clean_crop))
            scores_A.append(float(c_item["pred_y"]))

    rows = []
    if not rles_A:
        rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
    else:
        for k, enc in enumerate(rles_A, start=1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})
    if return_scores:
        return rows, scores_A
    return rows


# ────────────────────────── FAST FUSION & DISJOINT CANVAS ────────────────────

def load_test_candidates_for_stem(stem: str, model_list: List[tuple], cache_dir: str) -> List[dict]:
    """Loads precomputed TTA detection candidates across all models for a given image stem."""
    all_cands = []
    for model_name, fam, _ in model_list:
        pt_path = os.path.join(cache_dir, model_name, f"{stem}.pt")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        for c in data.get("candidates", []):
            all_cands.append({
                "box": c["box"],
                "score": float(c["score"]),
                "prob_crop": c["prob_crop"].astype(np.float32) if isinstance(c["prob_crop"], np.ndarray) else c["prob_crop"],
                "family": fam,
            })
    return all_cands


def fuse_test_candidates(
    cands: List[dict],
    family_weights: Dict[str, float],
    gamma: float,
    model_list: List[tuple],
    iou_thresh: float = 0.40,
) -> List[dict]:
    """Applies gamma score exponentiation, family weighting, greedy clustering, and WMF."""
    if not cands:
        return []

    weighted_cands = []
    for c in cands:
        w = family_weights.get(c["family"], 1.0)
        adj_score = (float(c["score"]) ** gamma) * w
        weighted_cands.append({
            "box": c["box"],
            "score": adj_score,
            "prob_crop": c["prob_crop"],
            "family": c["family"],
        })

    sorted_cands = sorted(weighted_cands, key=lambda x: x["score"], reverse=True)
    clusters = []

    for cand in sorted_cands:
        best_idx = -1
        best_iou = -1.0
        for idx, cl in enumerate(clusters):
            leader = cl[0]
            iou = fast_crop_mask_iou(cand, leader)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou > iou_thresh:
            clusters[best_idx].append(cand)
        else:
            clusters.append([cand])

    eff_views = sum(family_weights.get(m[1], 1.0) * 4 for m in model_list)

    fused = []
    for cl in clusters:
        scores = [float(x["score"]) for x in cl]
        score_sum = sum(scores)
        cluster_score = score_sum / float(eff_views)

        all_boxes = np.stack([x["box"] for x in cl], axis=0)
        ux1 = int(np.floor(all_boxes[:, 0].min()))
        uy1 = int(np.floor(all_boxes[:, 1].min()))
        ux2 = int(np.ceil(all_boxes[:, 2].max()))
        uy2 = int(np.ceil(all_boxes[:, 3].max()))
        union_box = np.array([ux1, uy1, ux2, uy2], dtype=np.int32)

        uh = uy2 - uy1
        uw = ux2 - ux1
        fused_prob = np.zeros((uh, uw), dtype=np.float32)
        for x in cl:
            w_norm = float(x["score"]) / score_sum
            b = x["box"]
            crop_f32 = x["prob_crop"].astype(np.float32) if x["prob_crop"].dtype != np.float32 else x["prob_crop"]
            fused_prob[b[1] - uy1 : b[3] - uy1, b[0] - ux1 : b[2] - ux1] += w_norm * crop_f32

        fused.append({
            "score": cluster_score,
            "box": union_box,
            "prob_crop": fused_prob,
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def apply_disjoint_canvas(fused_clusters: List[dict], tau_score: float, tau_mask: float) -> List[str]:
    """Paints candidate masks onto disjoint canvas in descending score order."""
    occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    rles = []

    for c in fused_clusters:
        sc = float(c["score"])
        if sc < tau_score:
            break
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bc = (c["prob_crop"] >= tau_mask)
        oa = int(bc.sum())
        if oa < 10:
            continue
        sub_occ = occupied[y1:y2, x1:x2]
        clean_crop = bc & (~sub_occ)
        ca = int(clean_crop.sum())
        if ca < 10 or (ca / float(oa)) < 0.50:
            continue
        occupied[y1:y2, x1:x2] |= clean_crop
        rles.append(encode_crop_to_rle(box, clean_crop))

    return rles


# ────────────────────────── FAST REPRODUCTION ENGINE ─────────────────────────

def run_fast_mode(
    cache_dir: str,
    rule_name: str = "B",
    config_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Fast CPU reproduction streaming precomputed detections from test_cache."""
    cfg = load_ensemble_config(config_path)
    models = [(m["id"], m["family"], m.get("img_size", 1536)) for m in cfg["models"]]

    rule_key = "rule_A" if rule_name.upper() == "A" else "rule_B"
    rule_cfg = cfg.get("decision_rules", {}).get(rule_key, {})
    if not rule_cfg:
        raise ValueError(f"Rule {rule_key} not found in {config_path}")

    tau_score = float(rule_cfg.get("tau_score", 0.850 if rule_name.upper() == "A" else 0.750))
    tau_mask = float(rule_cfg.get("tau_mask", 0.550))
    gamma = float(rule_cfg.get("gamma", 1.0 if rule_name.upper() == "A" else 1.5))
    family_weights = rule_cfg.get("family_weights", {m[1]: 1.0 for m in models})

    first_model = models[0][0]
    first_dir = os.path.join(cache_dir, first_model)
    if not os.path.isdir(first_dir):
        raise FileNotFoundError(f"Cache directory not found: {first_dir}")

    all_stems = sorted([p[:-3] for p in os.listdir(first_dir) if p.endswith(".pt")])
    if limit is not None and limit > 0:
        all_stems = all_stems[:limit]

    print(f"\n{'='*70}")
    print(f"[Fast Mode] Executing Rule {rule_name.upper()}: {rule_cfg.get('name', rule_key)}")
    print(f"Models: {len(models)} | Views: {len(models)*4} | Test images: {len(all_stems)}")
    print(f"Parameters: tau_score={tau_score:.3f}, tau_mask={tau_mask:.3f}, gamma={gamma:.1f}")
    print(f"Family weights: {family_weights}")
    print(f"{'='*70}")

    rows = []
    total_filaments = 0
    areas = []
    t0 = time.time()

    for idx, stem in enumerate(all_stems, start=1):
        cands = load_test_candidates_for_stem(stem, models, cache_dir)
        fused = fuse_test_candidates(cands, family_weights, gamma, models)
        rles = apply_disjoint_canvas(fused, tau_score, tau_mask)

        if not rles:
            rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(rles, start=1):
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})
                total_filaments += 1
                m_dec = mu.decode({"size": [FRAME_SIZE, FRAME_SIZE], "counts": enc.encode("ascii")})
                areas.append(int(m_dec.sum()))

        if idx % 30 == 0 or idx == len(all_stems):
            print(f"  Processed [{idx:03d}/{len(all_stems):03d}] stems ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    mean_a = float(np.mean(areas)) if areas else 0.0
    med_a = float(np.median(areas)) if areas else 0.0

    stats = {
        "rule": rule_name.upper(),
        "num_images": len(all_stems),
        "total_filaments": total_filaments,
        "total_rows": len(df),
        "mean_area": mean_a,
        "median_area": med_a,
        "elapsed_sec": elapsed,
    }

    print(f"\n[Fast Mode Complete] {total_filaments} filaments across {len(df)} rows in {elapsed:.2f}s.")
    print(f"Area stats: Mean={mean_a:.1f} px | Median={med_a:.1f} px | Min={min(areas) if areas else 0} | Max={max(areas) if areas else 0}")
    return df, stats


# ────────────────────────── FULL INFERENCE ENGINE ────────────────────────────

def materialize_stage1_rule(
    test_stems: List[str],
    raw_by_stem: Dict[str, List[dict]],
    cfg: dict,
    rule_name: str,
) -> Tuple[pd.DataFrame, dict]:
    """Fuse precomputed Stage-1 30-model candidates into a disjoint RLE DataFrame for Rule A or B."""
    models = cfg.get("models", [])
    rule_key = "rule_A" if rule_name.upper() == "A" else "rule_B"
    rule_cfg = cfg.get("decision_rules", {}).get(rule_key, {})

    tau_score = float(rule_cfg.get("tau_score", 0.850 if rule_name.upper() == "A" else 0.750))
    tau_mask = float(rule_cfg.get("tau_mask", 0.550))
    gamma = float(rule_cfg.get("gamma", 1.0 if rule_name.upper() == "A" else 1.5))
    family_weights = rule_cfg.get("family_weights", {m["family"]: 1.0 for m in models})
    model_list_tuples = [(m["id"], m["family"], m["img_size"]) for m in models]

    rows = []
    total_filaments = 0
    areas = []
    t0 = time.time()
    for stem in test_stems:
        fused = fuse_test_candidates(raw_by_stem[stem], family_weights, gamma, model_list_tuples)
        rles = apply_disjoint_canvas(fused, tau_score, tau_mask)
        if not rles:
            rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(rles, start=1):
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})
                total_filaments += 1
                m_dec = mu.decode({"size": [FRAME_SIZE, FRAME_SIZE], "counts": enc.encode("ascii")})
                areas.append(int(m_dec.sum()))

    df = pd.DataFrame(rows)
    mean_a = float(np.mean(areas)) if areas else 0.0
    med_a = float(np.median(areas)) if areas else 0.0
    stats = {
        "rule": rule_name.upper(),
        "num_images": len(test_stems),
        "total_filaments": total_filaments,
        "total_rows": len(df),
        "mean_area": mean_a,
        "median_area": med_a,
        "elapsed_sec": time.time() - t0,
    }
    return df, stats


def run_joint_full_mode(
    weights_dir: str,
    test_dir: str,
    rules: Tuple[str, ...] = ("A", "B", "C"),
    device_str: str = "auto",
    limit: Optional[int] = None,
    config_path: Optional[str] = None,
    phase3_config_path: Optional[str] = None,
    strict_weights_only: bool = True,
    verify_fp16: bool = True,
    stage1_cache_dir: Optional[str] = None,
    stage1_precision: str = "bf16",
) -> Tuple[Dict[str, Tuple[pd.DataFrame, dict]], List[str]]:
    """Run Stage-1 30 FP16 Mask R-CNN models once on GPU/CPU and materialize requested rules."""
    from scripts.refiner_pipeline import _collect_stage1_candidates_from_weights, run_phase3b_from_stage1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    cfg = load_ensemble_config(config_path)
    p3_cfg_p = Path(phase3_config_path) if phase3_config_path else (REPO_ROOT / "configs/phase3b_refiner.json")
    with open(p3_cfg_p, "r") as f:
        p3_cfg = json.load(f)
    tau_c = float(p3_cfg["ranker"]["tau_score"])

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    if not test_files:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    if limit is not None and limit > 0:
        test_files = test_files[:limit]
    stems = [Path(p).stem for p in test_files]

    img_by_stem: Dict[str, np.ndarray] = {}
    for p in test_files:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise IOError(f"Could not read image: {p}")
        img_by_stem[Path(p).stem] = g

    t0 = time.time()
    loaded_log: List[str] = []
    w_path = Path(weights_dir)
    print(f"[Joint Full Mode] device={device} | stage1_precision={stage1_precision} | strict_weights={strict_weights_only}", flush=True)
    raw_by_stem = _collect_stage1_candidates_from_weights(
        stems, img_by_stem, cfg, w_path, device,
        strict_weights_only=strict_weights_only, verify_fp16=verify_fp16,
        loaded_log=loaded_log, stage1_cache_dir=stage1_cache_dir,
        stage1_precision=stage1_precision,
    )
    stage1_elapsed = time.time() - t0

    outputs: Dict[str, Tuple[pd.DataFrame, dict]] = {}
    rules_upper = [r.upper() for r in rules]
    for r in ("A", "B"):
        if r in rules_upper:
            df_r, st_r = materialize_stage1_rule(stems, raw_by_stem, cfg, r)
            st_r["elapsed_sec"] = round(stage1_elapsed + st_r["elapsed_sec"], 2)
            outputs[r] = (df_r, st_r)

    if "C" in rules_upper:
        df_c, st_c = run_phase3b_from_stage1(
            stems=stems,
            img_by_stem=img_by_stem,
            raw_by_stem=raw_by_stem,
            cfg=cfg,
            w_dir=w_path,
            device=device,
            tau=tau_c,
            strict_weights_only=strict_weights_only,
            verify_fp16=verify_fp16,
            loaded_log=loaded_log,
            t_start=t0,
            free_raw_after_pool=True,
        )
        outputs["C"] = (df_c, st_c)

    return outputs, loaded_log


def run_full_mode(
    weights_dir: str,
    test_dir: str,
    rule_name: str = "B",
    device_str: str = "auto",
    limit: Optional[int] = None,
    config_path: Optional[str] = None,
    strict_weights_only: bool = False,
    stage1_precision: str = "bf16",
) -> Tuple[pd.DataFrame, dict]:
    """Full neural network inference from model weights with CPU fallback."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[Full Mode] Running inference on device: {device} (stage1_precision={stage1_precision}, strict_weights={strict_weights_only})...")

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    if not test_files:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    if limit is not None and limit > 0:
        test_files = test_files[:limit]

    test_stems = [Path(p).stem for p in test_files]
    print(f"[Full Mode] Found {len(test_files)} test images.")

    cfg = load_ensemble_config(config_path)
    models = cfg.get("models", [])
    rule_key = "rule_A" if rule_name.upper() == "A" else "rule_B"
    rule_cfg = cfg.get("decision_rules", {}).get(rule_key, {})

    tau_score = float(rule_cfg.get("tau_score", 0.850 if rule_name.upper() == "A" else 0.750))
    tau_mask = float(rule_cfg.get("tau_mask", 0.550))
    gamma = float(rule_cfg.get("gamma", 1.0 if rule_name.upper() == "A" else 1.5))
    family_weights = rule_cfg.get("family_weights", {m["family"]: 1.0 for m in models})

    from scripts.refiner_pipeline import resolve_stage1_checkpoint_path

    loaded_models = []
    w_path = Path(weights_dir)
    for m in models:
        subdir = m["id"]
        fam = m["family"]
        sz = m["img_size"]
        is_a5 = m.get("anchors5", False)

        ckpt_path = resolve_stage1_checkpoint_path(m, weights_dir=w_path, strict_weights_only=strict_weights_only)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(ckpt)
        model = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=is_a5)
        model.load_state_dict(state_dict)
        model.float().to(device).eval()
        loaded_models.append((subdir, fam, sz, model))
        print(f"  Loaded model: {subdir} ({fam}, {sz}px) from {ckpt_path.name}")

    views = [
        ("id", False, False),
        ("h", False, True),
        ("v", True, False),
        ("hv", True, True),
    ]

    all_rows = []
    total_filaments = 0
    areas = []
    t0 = time.time()
    model_list_tuples = [(m["id"], m["family"], m["img_size"]) for m in models]

    with torch.no_grad():
        for idx, (stem, img_path) in enumerate(zip(test_stems, test_files), start=1):
            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            needed_sizes = sorted({m[2] for m in loaded_models})
            t_by_size = {}
            for sz in needed_sizes:
                prep = preprocess_solar_image(raw_img, sz)
                t_by_size[sz] = torch.from_numpy(prep).float()

            combined_cands = []
            for subdir, fam, sz, model in loaded_models:
                base_t = t_by_size[sz]
                scale_to_2048 = 2048.0 / float(sz)

                for v_name, f_v, f_h in views:
                    flipped_tensor = apply_flip_transform(base_t, flip_v=f_v, flip_h=f_h).to(device)
                    with get_stage1_autocast_context(device, precision=stage1_precision):
                        output = model([flipped_tensor])[0]

                    scores = output["scores"]
                    keep = scores >= 0.25
                    if not keep.any():
                        continue

                    scores_np = scores[keep].detach().cpu().numpy()
                    boxes = output["boxes"][keep]
                    masks = output["masks"][keep]

                    boxes_unflipped = unflip_boxes(boxes, flip_v=f_v, flip_h=f_h, height=sz, width=sz)
                    masks_unflipped = unflip_mask_probabilities(masks, flip_v=f_v, flip_h=f_h)

                    boxes_2048 = (boxes_unflipped * scale_to_2048).detach().cpu().numpy()
                    if sz == FRAME_SIZE:
                        masks_up = masks_unflipped.squeeze(1).detach().cpu().numpy()
                    else:
                        masks_up = F.interpolate(
                            masks_unflipped.float(), size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
                        ).squeeze(1).detach().cpu().numpy()

                    for k in range(len(scores_np)):
                        b = boxes_2048[k]
                        x1 = max(0, int(np.floor(b[0])))
                        y1 = max(0, int(np.floor(b[1])))
                        x2 = min(FRAME_SIZE, int(np.ceil(b[2])) + 1)
                        y2 = min(FRAME_SIZE, int(np.ceil(b[3])) + 1)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float32)
                        combined_cands.append({
                            "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                            "score": float(scores_np[k]),
                            "prob_crop": crop_soft,
                            "family": fam,
                        })

            fused = fuse_test_candidates(combined_cands, family_weights, gamma, model_list_tuples)
            rles = apply_disjoint_canvas(fused, tau_score, tau_mask)

            if not rles:
                all_rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
            else:
                for k, enc in enumerate(rles, start=1):
                    all_rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})
                    total_filaments += 1
                    m_dec = mu.decode({"size": [FRAME_SIZE, FRAME_SIZE], "counts": enc.encode("ascii")})
                    areas.append(int(m_dec.sum()))

            if idx % 10 == 0 or idx == len(test_stems):
                print(f"  Progress: [{idx:03d}/{len(test_stems):03d}] ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    elapsed = time.time() - t0
    mean_a = float(np.mean(areas)) if areas else 0.0
    med_a = float(np.median(areas)) if areas else 0.0

    stats = {
        "rule": rule_name.upper(),
        "num_images": len(test_stems),
        "total_filaments": total_filaments,
        "total_rows": len(df),
        "mean_area": mean_a,
        "median_area": med_a,
        "elapsed_sec": elapsed,
    }

    print(f"\n[Full Mode Complete] {total_filaments} filaments across {len(df)} rows in {elapsed:.2f}s.")
    return df, stats


# ────────────────────────── MAIN CONTROLLER ──────────────────────────────────

def _resolve_test_dir(test_dir_arg: Optional[str], prefer_full_test: bool = False) -> Optional[str]:
    """Resolve test image directory from explicit CLI arg or repository fallbacks."""
    if test_dir_arg:
        return test_dir_arg
    paths = find_dataset_paths()
    cand_dirs = [
        paths.get("test_dir", ""),
        os.path.join(REPO_ROOT, "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
        os.path.join(REPO_ROOT.parent, "solar/data/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
    ]
    for cand in cand_dirs:
        if cand and os.path.isdir(cand):
            n_imgs = len(glob.glob(cand + "/*.jp*g"))
            if (prefer_full_test and n_imgs == 180) or (not prefer_full_test and n_imgs > 0):
                return cand
    return paths.get("test_dir")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Kaggle Solar Filament Submissions.")
    parser.add_argument("--rule", type=str, choices=["A", "B", "C", "ALL", "a", "b", "c", "all"], default="B",
                        help="Submission rule: 'A', 'B', 'C' (Phase 3B Local), or 'ALL' (single GPU Stage-1 pass)")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast",
                        help="Operation mode: 'fast' (from cache) or 'full' (from weights)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (for ALL, interpreted as output directory)")
    parser.add_argument("--config", type=str, default="configs/final_ensemble.json",
                        help="Path to ensemble configuration JSON")
    parser.add_argument("--cache-dir", type=str, default="work/test_cache",
                        help="Directory with precomputed candidate detections")
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="Directory with model weights (default: repo weights/)")
    parser.add_argument("--strict-weights", action="store_true",
                        help="Enforce strict loading from --weights-dir (FP16) with zero fallback")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory of test images (for full mode or Rule C)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root dataset directory or directory of test images")
    parser.add_argument("--device", type=str, default="auto",
                        help="Inference device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test images to process")
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Run validate_submission.py on the generated CSV(s)")
    parser.add_argument("--stage1-cache-dir", type=str, default=None,
                        help="Optional per-stem checkpoint directory for resumable Stage-1 inference")
    parser.add_argument("--stage1-precision", type=str, choices=["bf16", "fp32"], default="bf16",
                        help="Stage-1 Mask R-CNN inference precision: 'bf16' (default, reproduces Kaggle test_cache) or 'fp32' (strict float32)")
    args = parser.parse_args()

    rule_upper = args.rule.upper()
    default_out_names = {
        "A": "work/submission_A_uniform_30models.csv",
        "B": "work/submission_B_weighted_30models.csv",
        "C": "work/submission_C_phase3b_local.csv",
    }

    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(REPO_ROOT, args.cache_dir)
    if not os.path.isdir(cache_dir):
        alt_cache = os.path.join(REPO_ROOT.parent, "solar/work/test_cache")
        if os.path.isdir(alt_cache):
            cache_dir = alt_cache

    config_arg = "configs/phase3b_refiner.json" if (rule_upper == "C" and args.config == "configs/final_ensemble.json") else args.config
    weights_dir = args.weights_dir if os.path.isabs(args.weights_dir) else os.path.join(REPO_ROOT, args.weights_dir)
    config_path = config_arg if os.path.isabs(config_arg) else os.path.join(REPO_ROOT, config_arg)

    generated: List[Tuple[str, str, pd.DataFrame, dict]] = []
    if rule_upper == "ALL":
        if args.mode != "full":
            raise ValueError("--rule ALL requires --mode full")
        test_dir = _resolve_test_dir(args.test_dir or args.data_dir, prefer_full_test=False)
        if not test_dir or not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        out_dir = Path(args.output if args.output else os.path.join(REPO_ROOT, "work"))
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs, loaded_log = run_joint_full_mode(
            weights_dir=weights_dir,
            test_dir=test_dir,
            rules=("A", "B", "C"),
            device_str=args.device,
            limit=args.limit,
            config_path=config_path,
            strict_weights_only=args.strict_weights,
            verify_fp16=args.strict_weights,
            stage1_cache_dir=args.stage1_cache_dir,
            stage1_precision=args.stage1_precision,
        )
        name_map = {
            "A": "submission_A_uniform_30models.csv",
            "B": "submission_B_weighted_30models.csv",
            "C": "submission_phase3b_local.csv",
        }
        for r in ("A", "B", "C"):
            df_r, st_r = outputs[r]
            p_r = str(out_dir / name_map[r])
            df_r.to_csv(p_r, index=False)
            generated.append((r, p_r, df_r, st_r))
        print(f"[Joint Full Mode] Verified {len(set(loaded_log))} distinct repo weight files loaded.")
    else:
        out_rel = args.output if args.output is not None else default_out_names[rule_upper]
        out_path = out_rel if os.path.isabs(out_rel) else os.path.normpath(os.path.join(REPO_ROOT, out_rel))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if rule_upper == "C":
            from scripts.refiner_pipeline import run_phase3b_pipeline
            test_dir = _resolve_test_dir(args.test_dir or args.data_dir, prefer_full_test=False)
            if not test_dir or not os.path.isdir(test_dir):
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            df_sub, stats = run_phase3b_pipeline(
                test_dir=test_dir,
                weights_dir=weights_dir,
                cache_dir=cache_dir,
                config_path=config_path,
                device_str=args.device,
                limit=args.limit,
                use_cache=(args.mode == "fast"),
                strict_weights_only=args.strict_weights,
                verify_fp16=args.strict_weights,
                stage1_precision=args.stage1_precision,
                stage1_cache_dir=args.stage1_cache_dir,
            )
        elif args.mode == "fast":
            df_sub, stats = run_fast_mode(
                cache_dir=cache_dir,
                rule_name=rule_upper,
                config_path=config_path,
                limit=args.limit,
            )
        else:
            test_dir = _resolve_test_dir(args.test_dir or args.data_dir, prefer_full_test=False)
            if not test_dir or not os.path.isdir(test_dir):
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            df_sub, stats = run_full_mode(
                weights_dir=weights_dir,
                test_dir=test_dir,
                rule_name=rule_upper,
                device_str=args.device,
                limit=args.limit,
                config_path=config_path,
                strict_weights_only=args.strict_weights,
                stage1_precision=args.stage1_precision,
            )
        df_sub.to_csv(out_path, index=False)
        generated.append((rule_upper, out_path, df_sub, stats))

    val_test_dir = _resolve_test_dir(args.test_dir or args.data_dir, prefer_full_test=True)
    val_script = os.path.join(REPO_ROOT, "scripts/validate_submission.py")
    ref_map = {"A": (1165, 1241.0), "B": (1155, 1259.0), "C": (1170, 1233.5)}

    for r_tag, out_p, df_sub, stats in generated:
        print(f"\n[Saved Submission {r_tag}] {out_p} ({len(df_sub)} rows, {stats['total_filaments']} filaments)")
        if args.validate:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_val_dir:
                cmd = [sys.executable, val_script, "--sub", out_p]
                if args.limit:
                    sub_stems = sorted(df_sub["filament_id"].astype(str).str.split("_").str[0].unique().tolist())
                    for s_val in sub_stems:
                        Path(tmp_val_dir, f"{s_val}.jpg").touch()
                    cmd.extend(["--test_dir", tmp_val_dir, "--expected_count", str(len(sub_stems))])
                elif val_test_dir and os.path.isdir(val_test_dir):
                    cmd.extend(["--test_dir", val_test_dir])
                    dir_imgs = [
                        p for p in glob.glob(os.path.join(val_test_dir, "*.*"))
                        if p.lower().endswith((".jpeg", ".jpg", ".png"))
                    ]
                    if dir_imgs:
                        cmd.extend(["--expected_count", str(len(dir_imgs))])
                print(f"[Validation Hook {r_tag}] Running: {' '.join(cmd)}")
                res = subprocess.run(cmd, capture_output=True, text=True)
                print(res.stdout)
                if res.returncode != 0:
                    print("STDERR:\n", res.stderr)
                    raise RuntimeError(f"Validation failed for {out_p}")

            n_imgs_eval = stats.get("num_images", len(df_sub["filament_id"].str.split("_").str[0].unique()))
            if n_imgs_eval == 180 and not args.limit:
                ref_filaments, ref_median = ref_map[r_tag]
                cnt_diff = abs(stats["total_filaments"] - ref_filaments) / ref_filaments * 100.0
                med_diff = abs(stats["median_area"] - ref_median) / ref_median * 100.0
                print(f"Sanity Check [{r_tag}] vs Reference:")
                print(f"  Filaments: {stats['total_filaments']} vs {ref_filaments} (diff: {cnt_diff:.2f}%)")
                print(f"  Median Area: {stats['median_area']:.1f} px vs {ref_median:.1f} px (diff: {med_diff:.2f}%)")
                if cnt_diff > 10.0 or med_diff > 10.0:
                    print(f"[WARNING] Sanity check deviation > 10% for Rule {r_tag}!")
                else:
                    print(f"[OK] Rule {r_tag} sanity check passed within ±10% boundary.")


if __name__ == "__main__":
    main()
