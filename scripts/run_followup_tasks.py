#!/usr/bin/env python3
"""
scripts/run_followup_tasks.py - Executes Follow-up Tasks 1, 2, 3, and 4:

Order of execution (fastest CPU tasks first so logs appear immediately):
  Step A: Load 4v clahe & 12v 3-family OOF pools (with per-fold precomputed fused clusters)
  Step B (Task 4): Strict 5-Fold Nested Selection across Scorer / tau_mask variants
  Step C (Task 3): Like-for-like comparison on the 296 multi-annotator images from ceiling.py
  Step D (Task 1A): Adversarial Validation with Scorer v2 (view-invariant 19 features)
                    on the exact 32-view 8-model ensemble used in current Submissions A/B vs 12-view OOF
  Step E (Task 2 & Task 1B/1C): Fast 1-pass-per-image caching of the 10 newly finished models
                    (anchors5 f1-f4, dice f1-f4, 2048 f1-f2), rebuilding Submissions A and B
                    from ALL 18 finished models (and 15-model balanced 3-family), plus Adversarial Validation.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ceiling
import pq_eval
from scripts.adversarial_validation import print_adv_report, run_adversarial_classifier
from scripts.common import EMPTY_RLE, find_dataset_paths, get_maskrcnn_model
from scripts.instance_scorer import FEATURE_NAMES, encode_crop_to_rle
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    SCORER_V2_FEATURES,
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    build_gt_rles_by_obs,
    compute_binary_crop_iou,
    drop_os_page_cache,
    evaluate_threshold_grid_nested_cv,
    extract_scorer_v2_dataset,
    load_fused_oof_pool,
    paired_bootstrap_from_per_obs,
    print_boot_line,
    weighted_mask_fusion_tagged,
)
from scripts.oof_eval import extract_4view_candidates_from_tensor
from scripts.predict import load_image


def ensure_all_finished_models_cached(cache_dir: str = "work/test_cache", device_str: str = "cuda") -> List[str]:
    """
    Caches 4-flip test candidates for all 18 currently finished models in a single pass
    over the 180 test images (loading each JPEG once per image instead of once per model!).
    Uses ~2.5 GB VRAM total across the 10 loaded models on RTX A6000.
    """
    paths = find_dataset_paths()
    test_dir = paths["test_dir"]
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    test_stems_and_paths = [(Path(p).stem, p) for p in test_files]
    all_test_stems = [s for s, _ in test_stems_and_paths]

    model_specs = [
        # clahe f0-f4
        ("f0_1536", "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt", 1536),
        ("f1_1536", "work/runs/r50_1536_all_f1/checkpoints/checkpoint_fold1_last.pt", 1536),
        ("f2_1536", "work/runs/r50_1536_all_f2/checkpoints/checkpoint_fold2_last.pt", 1536),
        ("f3_1536", "work/runs/r50_1536_all_f3/checkpoints/checkpoint_fold3_last.pt", 1536),
        ("f4_1536", "work/runs/r50_1536_all_f4/checkpoints/checkpoint_fold4_last.pt", 1536),
        # anchors5 f0-f4
        ("f0_anchors5", "work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("f1_anchors5", "work/runs/r50_1536_a5_f1/checkpoints/checkpoint_fold1_last.pt", 1536),
        ("f2_anchors5", "work/runs/r50_1536_a5_f2/checkpoints/checkpoint_fold2_last.pt", 1536),
        ("f3_anchors5", "work/runs/r50_1536_a5_f3/checkpoints/checkpoint_fold3_last.pt", 1536),
        ("f4_anchors5", "work/runs/r50_1536_a5_f4/checkpoints/checkpoint_fold4_last.pt", 1536),
        # dice f0-f4
        ("f0_dice", "work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("f1_dice", "work/runs/r50_1536_dice_f1/checkpoints/checkpoint_fold1_last.pt", 1536),
        ("f2_dice", "work/runs/r50_1536_dice_f2/checkpoints/checkpoint_fold2_last.pt", 1536),
        ("f3_dice", "work/runs/r50_1536_dice_f3/checkpoints/checkpoint_fold3_last.pt", 1536),
        ("f4_dice", "work/runs/r50_1536_dice_f4/checkpoints/checkpoint_fold4_last.pt", 1536),
        # 2048 f0-f2 (finished folds)
        ("f0_2048", "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt", 2048),
        ("f1_2048", "work/runs/r50_2048_all_f1/checkpoints/checkpoint_fold1_last.pt", 2048),
        ("f2_2048", "work/runs/r50_2048_all_f2/checkpoints/checkpoint_fold2_last.pt", 2048),
    ]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    loaded_models = []
    for name, ckpt_path, sz in model_specs:
        m_dir = os.path.join(cache_dir, name)
        os.makedirs(m_dir, exist_ok=True)
        missing = [s for s in all_test_stems if not os.path.isfile(os.path.join(m_dir, f"{s}.pt"))]
        if not missing:
            print(f"  [Cache Complete] {name} (180/180 stems)")
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {})
            a5 = cfg.get("anchors5", False) or ("anchors5" in name) or ("_a5_" in ckpt_path)
            m = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=a5)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            del ckpt
            drop_os_page_cache(ckpt_path)
            loaded_models.append((name, m, sz))
            print(f"  [Loaded GPU] {name} ({sz}px, {len(missing)} stems to cache)")

    if not loaded_models:
        return all_test_stems

    t0 = time.time()
    with torch.no_grad():
        for idx, (stem, img_path) in enumerate(test_stems_and_paths, start=1):
            base_1536 = None
            base_2048 = None
            for name, m, sz in loaded_models:
                cache_pt = os.path.join(cache_dir, name, f"{stem}.pt")
                if os.path.isfile(cache_pt):
                    continue
                if sz == 2048:
                    if base_2048 is None:
                        base_2048, _ = load_image(img_path, img_size=2048)
                    bt = base_2048
                else:
                    if base_1536 is None:
                        base_1536, _ = load_image(img_path, img_size=1536)
                    bt = base_1536
                cands = extract_4view_candidates_from_tensor(m, bt, sz, device, score_filter=0.25)
                tmp_pt = f"{cache_pt}.tmp"
                torch.save({"stem": stem, "model": name, "candidates": cands}, tmp_pt)
                os.replace(tmp_pt, cache_pt)
                drop_os_page_cache(cache_pt)
                del cands
            del base_1536, base_2048
            if idx % 30 == 0 or idx == len(test_stems_and_paths):
                print(f"  Caching progress: [{idx:03d}/{len(test_stems_and_paths):03d}] ({time.time() - t0:.1f}s)", flush=True)

    for _, m, _ in loaded_models:
        del m
    del loaded_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_test_stems


def extract_test_scorer_v2_features_and_preds(
    test_stems: List[str],
    model_list: List[Tuple[str, str, float]],
    num_views_total: int,
    fam_views: Dict[str, int],
    scorer_models: Optional[List[lgb.LGBMRegressor]] = None,
    scorer_feature_cols: Optional[List[str]] = None,
    scorer_tau: float = 0.2440,
    scorer_tau_mask: float = 0.525,
    thresh_tau_score: float = 0.875,
    thresh_tau_mask: float = 0.550,
    cache_dir: str = "work/test_cache",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Streams through all 180 test images for a given test ensemble configuration,
    extracts the exact Scorer v2 feature DataFrame (with tight binary-mask bbox),
    and optionally builds Submission A (Scorer v2) and Submission B (Threshold Rule).
    """
    all_feat_rows = []
    rows_A = []
    rows_B = []

    for stem in test_stems:
        combined_cands = []
        for m_name, fam_name, p_shift in model_list:
            pt_path = os.path.join(cache_dir, m_name, f"{stem}.pt")
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            for c in data.get("candidates", []):
                pc = c["prob_crop"]
                if p_shift != 0.0:
                    pc = np.clip(pc.astype(np.float32) + p_shift, 0.0, 1.0)
                combined_cands.append({
                    "box": c["box"],
                    "score": float(c["score"]),
                    "prob_crop": pc,
                    "family": fam_name,
                })
            del data
            drop_os_page_cache(pt_path)

        fused = weighted_mask_fusion_tagged(
            combined_cands,
            num_views=num_views_total,
            iou_thresh=0.40,
            family_views=fam_views,
        )
        del combined_cands

        # --- Submission B: Threshold Rule ---
        occupied_B = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        rles_B = []
        for c in fused:
            if c["score"] < thresh_tau_score:
                break
            box = c["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bc = (c["prob_crop"] >= thresh_tau_mask)
            oa = int(bc.sum())
            if oa == 0:
                continue
            sub_occ = occupied_B[y1:y2, x1:x2]
            clean_crop = bc & (~sub_occ)
            ca = int(clean_crop.sum())
            if ca < 10 or (ca / float(oa)) < 0.50:
                continue
            occupied_B[y1:y2, x1:x2] |= clean_crop
            rles_B.append(encode_crop_to_rle(box, clean_crop))

        if not rles_B:
            rows_B.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(rles_B, start=1):
                rows_B.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

        # --- Scorer v2 Feature Extraction & Submission A ---
        all_scores = [float(c["score"]) for c in fused]
        max_score_in_img = max(all_scores) if all_scores else 1e-6
        n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

        feat_dicts = []
        cands_A = []
        higher_rank_boxes_and_crops = []
        sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)

        for rank_0, c in enumerate(fused):
            sc = float(c["score"])
            if sc < 0.15:
                continue
            prob_crop = c["prob_crop"]
            box = c["box"]
            binary_crop = (prob_crop >= scorer_tau_mask)
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
            norm_views = float(c["cluster_size"]) * (4.0 / float(num_views_total))

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

            f_row = {
                "obs": stem,
                "fold": -1,
                "cand_idx": len(cands_A),
                "cluster_score": sc,
                "num_views": norm_views,
                "member_score_mean": member_mean,
                "member_score_std": member_std,
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
            }
            feat_dicts.append(f_row)
            all_feat_rows.append(f_row)
            cands_A.append({
                "box": box,
                "binary_crop": binary_crop,
                "orig_area": area,
                "pred_y": 0.0,
            })

        rles_A = []
        if cands_A and scorer_models is not None and scorer_feature_cols is not None:
            X_stem = pd.DataFrame(feat_dicts)[scorer_feature_cols].values
            pred_y_stem = np.mean([m.predict(X_stem) for m in scorer_models], axis=0)
            for c_item, py in zip(cands_A, pred_y_stem):
                c_item["pred_y"] = float(py)

            kept_A = [c_item for c_item in cands_A if c_item["pred_y"] > scorer_tau]
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

        if not rles_A:
            rows_A.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(rles_A, start=1):
                rows_A.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

        del fused

    return pd.DataFrame(all_feat_rows), pd.DataFrame(rows_A), pd.DataFrame(rows_B)


def run_strict_nested_scorer_selection(
    pool_12v: Dict[str, List[dict]],
    folds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    gt_by_obs: Dict[str, List[List[str]]],
    train_json_path: str,
    seed: int = 2026,
) -> Tuple[dict, Dict[str, List[str]], List[lgb.LGBMRegressor]]:
    """
    Task 4: Runs strict Nested CV where for each outer held-out fold f_val in 0..4,
    the choice among the 6 multi-family Scorer variants AND its threshold tau_y
    are selected STRICTLY inside the 4 training folds.
    """
    print("\n" + "#" * 92)
    print("  TASK 4: STRICT NESTED SELECTION OF SCORER VARIANT & TAU_MASK INSIDE TRAINING FOLDS")
    print("#" * 92, flush=True)

    feat_050, pool_050 = extract_scorer_v2_dataset(
        pool_12v, folds_df, gt_df, train_json_path, num_views_total=12, tau_mask=0.500, use_tight_bbox=True
    )
    feat_0525, pool_0525 = extract_scorer_v2_dataset(
        pool_12v, folds_df, gt_df, train_json_path, num_views_total=12, tau_mask=0.525, use_tight_bbox=True
    )

    variants = [
        ("v1_14feat_mk050", feat_050, pool_050, FEATURE_NAMES, 0.500),
        ("v2_full21_mk050", feat_050, pool_050, SCORER_V2_FEATURES, 0.500),
        ("v2_inv19_mk050", feat_050, pool_050, SCORER_V2_VIEW_INVARIANT_FEATURES, 0.500),
        ("v1_14feat_mk0525", feat_0525, pool_0525, FEATURE_NAMES, 0.525),
        ("v2_full21_mk0525", feat_0525, pool_0525, SCORER_V2_FEATURES, 0.525),
        ("v2_inv19_mk0525", feat_0525, pool_0525, SCORER_V2_VIEW_INVARIANT_FEATURES, 0.525),
    ]

    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    tau_grid = [round(0.180 + 0.005 * k, 4) for k in range(19)]

    variant_fold_tau_stats: Dict[str, Dict[int, Dict[float, dict]]] = {}
    variant_fold_tau_per_obs: Dict[str, Dict[int, Dict[float, Dict[str, dict]]]] = {}
    variant_fold_tau_rles: Dict[str, Dict[int, Dict[float, Dict[str, List[str]]]]] = {}
    v2_inv_0525_models: List[lgb.LGBMRegressor] = []

    for v_name, f_df, p_obs, f_cols, mk in variants:
        oof_preds = np.zeros(len(f_df), dtype=np.float32)
        models_v = []
        for val_f in range(5):
            tr_m = (f_df["fold"] != val_f).values
            va_m = (f_df["fold"] == val_f).values
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
            model.fit(f_df.loc[tr_m, f_cols].values, f_df.loc[tr_m, "target_y"].values)
            oof_preds[va_m] = np.clip(model.predict(f_df.loc[va_m, f_cols].values), 0.0, 1.0)
            models_v.append(model)

        if v_name == "v2_inv19_mk0525":
            v2_inv_0525_models = models_v

        for r_idx, row in enumerate(f_df[["obs", "cand_idx"]].itertuples(index=False)):
            p_obs[row.obs][int(row.cand_idx)]["pred_y"] = float(oof_preds[r_idx])

        obs_cache = {}
        for f in range(5):
            for obs in obs_by_fold[f]:
                cands = p_obs.get(obs, [])
                sorted_cands = sorted(cands, key=lambda c: c["pred_y"], reverse=True)
                occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                painted_rles = []
                painted_scores = []
                for c in sorted_cands:
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
                    painted_rles.append(encode_crop_to_rle(b, clean_crop))
                    painted_scores.append(float(c["pred_y"]))
                annot_gts = gt_by_obs.get(obs, [])
                iou_mats = [pq_eval.iou_matrix(gt_rles, painted_rles) for gt_rles in annot_gts]
                obs_cache[obs] = (np.asarray(painted_scores, dtype=np.float64), iou_mats, painted_rles)

        variant_fold_tau_stats[v_name] = {f: {} for f in range(5)}
        variant_fold_tau_per_obs[v_name] = {f: {} for f in range(5)}
        variant_fold_tau_rles[v_name] = {f: {} for f in range(5)}

        for f in range(5):
            for tau in tau_grid:
                iou_s, tp, fp, fn = 0.0, 0, 0, 0
                per_obs_f = {}
                rles_f = {}
                for obs in obs_by_fold[f]:
                    scores_arr, iou_mats, painted_rles = obs_cache[obs]
                    k_keep = int(np.sum(scores_arr > tau))
                    rles_f[obs] = painted_rles[:k_keep]
                    o_iou, o_tp, o_fp, o_fn = 0.0, 0, 0, 0
                    for m_full in iou_mats:
                        m = m_full[:, :k_keep]
                        n_gt, n_pred = m.shape
                        if n_gt == 0:
                            o_fp += n_pred
                        elif n_pred == 0:
                            o_fn += n_gt
                        else:
                            hit = m > pq_eval.IOU_THRESHOLD
                            o_iou += float(m[hit].sum())
                            o_tp += int(hit.sum())
                            o_fp += int((hit.sum(axis=0) == 0).sum())
                            o_fn += int((hit.sum(axis=1) == 0).sum())
                    per_obs_f[obs] = {"iou_sum": o_iou, "tp": o_tp, "fp": o_fp, "fn": o_fn}
                    iou_s += o_iou
                    tp += o_tp
                    fp += o_fp
                    fn += o_fn
                variant_fold_tau_stats[v_name][f][tau] = {"iou_sum": iou_s, "tp": tp, "fp": fp, "fn": fn}
                variant_fold_tau_per_obs[v_name][f][tau] = per_obs_f
                variant_fold_tau_rles[v_name][f][tau] = rles_f
        del obs_cache

    nested_per_obs: Dict[str, dict] = {}
    nested_fold_choices = {}
    tot_iou, tot_tp, tot_fp, tot_fn = 0.0, 0, 0, 0

    for val_f in range(5):
        best_v = None
        best_tau = None
        best_tr_pq = -1.0
        for v_name, _, _, _, _ in variants:
            for tau in tau_grid:
                tr_iou = sum(variant_fold_tau_stats[v_name][f][tau]["iou_sum"] for f in range(5) if f != val_f)
                tr_tp = sum(variant_fold_tau_stats[v_name][f][tau]["tp"] for f in range(5) if f != val_f)
                tr_fp = sum(variant_fold_tau_stats[v_name][f][tau]["fp"] for f in range(5) if f != val_f)
                tr_fn = sum(variant_fold_tau_stats[v_name][f][tau]["fn"] for f in range(5) if f != val_f)
                denom = tr_tp + 0.5 * tr_fp + 0.5 * tr_fn
                tr_pq = (tr_iou / denom) if denom > 0 else 0.0
                if tr_pq > best_tr_pq:
                    best_tr_pq = tr_pq
                    best_v = v_name
                    best_tau = tau

        st = variant_fold_tau_stats[best_v][val_f][best_tau]
        v_denom = st["tp"] + 0.5 * st["fp"] + 0.5 * st["fn"]
        v_pq = st["iou_sum"] / v_denom if v_denom > 0 else 0.0
        v_sq = st["iou_sum"] / st["tp"] if st["tp"] > 0 else 0.0
        v_rq = st["tp"] / v_denom if v_denom > 0 else 0.0

        nested_fold_choices[val_f] = {
            "chosen_variant": best_v,
            "chosen_tau": best_tau,
            "train_4fold_pq": best_tr_pq,
            "val_pq": v_pq,
            "val_sq": v_sq,
            "val_rq": v_rq,
        }
        print(
            f"  Outer Fold {val_f}: selected ({best_v}, tau={best_tau:.3f}) on 4 train folds "
            f"(Train PQ={best_tr_pq:.4f}) -> Held-out Val PQ={v_pq:.4f} (SQ:{v_sq:.4f}, RQ:{v_rq:.4f})",
            flush=True,
        )
        tot_iou += st["iou_sum"]
        tot_tp += st["tp"]
        tot_fp += st["fp"]
        tot_fn += st["fn"]
        nested_per_obs.update(variant_fold_tau_per_obs[best_v][val_f][best_tau])

    denom_all = tot_tp + 0.5 * tot_fp + 0.5 * tot_fn
    nested_pooled = {
        "pq": float(tot_iou / denom_all),
        "sq": float(tot_iou / tot_tp),
        "rq": float(tot_tp / denom_all),
        "tp": int(tot_tp),
        "fp": int(tot_fp),
        "fn": int(tot_fn),
    }

    v2_inv_rles_by_obs: Dict[str, List[str]] = {}
    for val_f in range(5):
        best_tau_f = max(
            tau_grid,
            key=lambda t: (
                sum(variant_fold_tau_stats["v2_inv19_mk0525"][f][t]["iou_sum"] for f in range(5) if f != val_f)
                / max(
                    1e-9,
                    sum(
                        variant_fold_tau_stats["v2_inv19_mk0525"][f][t]["tp"]
                        + 0.5 * variant_fold_tau_stats["v2_inv19_mk0525"][f][t]["fp"]
                        + 0.5 * variant_fold_tau_stats["v2_inv19_mk0525"][f][t]["fn"]
                        for f in range(5)
                        if f != val_f
                    ),
                )
            ),
        )
        v2_inv_rles_by_obs.update(variant_fold_tau_rles["v2_inv19_mk0525"][val_f][best_tau_f])

    return {
        "nested_pooled": nested_pooled,
        "nested_folds": nested_fold_choices,
        "nested_per_obs": nested_per_obs,
        "feat_0525": feat_0525,
    }, v2_inv_rles_by_obs, v2_inv_0525_models


def run_ceiling_296_comparison(
    gt_all_path: str,
    clahe_oof_csv: str,
    multifam_thresh_rles_by_obs: Dict[str, List[str]],
    scorer_v2_rles_by_obs: Dict[str, List[str]],
) -> dict:
    """
    Task 3: Evaluates our OOF models on the EXACT same 296 multi-annotator observations
    used by ceiling.py, both:
      (A) vs. ALL annotator records on those observations (standard pq_eval.py protocol)
      (B) vs. OTHERS-ONLY annotator records (gt["rank"] != r for r=0, 1, 2, exact ceiling.py protocol)
    """
    print("\n" + "#" * 92)
    print("  TASK 3: LIKE-FOR-LIKE COMPARISON ON THE 296 MULTI-ANNOTATOR IMAGES FROM CEILING.PY")
    print("#" * 92, flush=True)

    gt_multi = ceiling.load_gt(gt_all_path)
    obs_296 = sorted(gt_multi["obs"].unique().tolist())
    print(f"Loaded {len(obs_296)} multi-annotator observations from ceiling.load_gt().")

    def rles_dict_to_df(rles_by_obs: Dict[str, List[str]], obs_subset: List[str]) -> pd.DataFrame:
        rows = []
        for o in obs_subset:
            for k, rle in enumerate(rles_by_obs.get(o, []), start=1):
                rows.append({"filament_id": f"{o}_{k}", "segmentation_rle": rle})
        return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])

    df_clahe_all = pd.read_csv(clahe_oof_csv)
    df_clahe_all = df_clahe_all[df_clahe_all["segmentation_rle"] != EMPTY_RLE].copy()
    df_clahe_all["obs"] = df_clahe_all["filament_id"].str.split("_", n=1).str[0]

    ref_all_296 = gt_multi[["filament_id", "segmentation_rle"]].copy()
    pred_clahe_296 = df_clahe_all[df_clahe_all["obs"].isin(set(obs_296))][["filament_id", "segmentation_rle"]]
    pred_mf_thresh_296 = rles_dict_to_df(multifam_thresh_rles_by_obs, obs_296)
    pred_scorer_v2_296 = rles_dict_to_df(scorer_v2_rles_by_obs, obs_296)

    res_all_clahe = pq_eval.score(ref_all_296, pred_clahe_296)
    res_all_mf = pq_eval.score(ref_all_296, pred_mf_thresh_296)
    res_all_sc = pq_eval.score(ref_all_296, pred_scorer_v2_296)

    print("\n--- A. Scored vs. ALL Annotator Records on the 296 Multi-Annotator Images ---")
    print(f"  1536_clahe (4v Threshold)          : PQ = {res_all_clahe['pq']:.4f} (SQ = {res_all_clahe['sq']:.4f}, RQ = {res_all_clahe['rq']:.4f})")
    print(f"  3-Family 12v (Threshold Rule)      : PQ = {res_all_mf['pq']:.4f} (SQ = {res_all_mf['sq']:.4f}, RQ = {res_all_mf['rq']:.4f})")
    print(f"  3-Family 12v (Scorer v2 Inv)       : PQ = {res_all_sc['pq']:.4f} (SQ = {res_all_sc['sq']:.4f}, RQ = {res_all_sc['rq']:.4f})")

    print("\n--- B. Exact ceiling.py Protocol (Scored vs. OTHERS-ONLY: rank != r, and vs. ALL) ---")
    print(
        f"  {'Split / Predictor':<36} | {'Imgs':>4} | {'PQ (vs Others)':>14} | {'SQ':>6} | {'RQ':>6} | {'PQ (vs All)':>11}"
    )
    print("-" * 92)

    by_rank_out = {}
    for r in (0, 1, 2):
        ref_others, pred_human = ceiling.split(gt_multi, r)
        obs_r = sorted(gt_multi.loc[gt_multi["rank"] == r, "obs"].unique().tolist())
        ref_all_r = gt_multi[gt_multi["obs"].isin(set(obs_r))][["filament_id", "segmentation_rle"]].copy()

        res_hum_others = pq_eval.score(ref_others, pred_human)
        res_hum_all = pq_eval.score(ref_all_r, pred_human)

        pred_c_r = df_clahe_all[df_clahe_all["obs"].isin(set(obs_r))][["filament_id", "segmentation_rle"]]
        pred_m_r = rles_dict_to_df(multifam_thresh_rles_by_obs, obs_r)
        pred_s_r = rles_dict_to_df(scorer_v2_rles_by_obs, obs_r)

        res_c_others = pq_eval.score(ref_others, pred_c_r)
        res_m_others = pq_eval.score(ref_others, pred_m_r)
        res_s_others = pq_eval.score(ref_others, pred_s_r)

        res_c_all = pq_eval.score(ref_all_r, pred_c_r)
        res_m_all = pq_eval.score(ref_all_r, pred_m_r)
        res_s_all = pq_eval.score(ref_all_r, pred_s_r)

        print(
            f"  Rank r={r}: Human Annotator #{r:<9} | {len(obs_r):>4} | "
            f"{res_hum_others['pq']:>14.4f} | {res_hum_others['sq']:>6.4f} | {res_hum_others['rq']:>6.4f} | "
            f"{res_hum_all['pq']:>11.4f} (includes self!)"
        )
        print(
            f"  Rank r={r}: 1536_clahe (4v Threshold)  | {len(obs_r):>4} | "
            f"{res_c_others['pq']:>14.4f} | {res_c_others['sq']:>6.4f} | {res_c_others['rq']:>6.4f} | "
            f"{res_c_all['pq']:>11.4f}"
        )
        print(
            f"  Rank r={r}: 3-Family 12v Threshold     | {len(obs_r):>4} | "
            f"{res_m_others['pq']:>14.4f} | {res_m_others['sq']:>6.4f} | {res_m_others['rq']:>6.4f} | "
            f"{res_m_all['pq']:>11.4f}"
        )
        print(
            f"  Rank r={r}: 3-Family 12v Scorer v2     | {len(obs_r):>4} | "
            f"{res_s_others['pq']:>14.4f} | {res_s_others['sq']:>6.4f} | {res_s_others['rq']:>6.4f} | "
            f"{res_s_all['pq']:>11.4f}",
            flush=True,
        )
        print("-" * 92)

        by_rank_out[f"rank_{r}"] = {
            "n_obs": len(obs_r),
            "human_vs_others": res_hum_others,
            "human_vs_all": res_hum_all,
            "clahe_vs_others": res_c_others,
            "multifam_thresh_vs_others": res_m_others,
            "scorer_v2_vs_others": res_s_others,
        }

    return {
        "all_296_vs_all_records": {
            "clahe_4v_thresh": res_all_clahe,
            "multifam_12v_thresh": res_all_mf,
            "scorer_v2_12v": res_all_sc,
        },
        "by_rank": by_rank_out,
    }


def main():
    argparse.ArgumentParser(
        description="Execute multi-family OOF follow-up evaluation and adversarial validation checks."
    ).parse_args()
    t0 = time.time()
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    gt_df = pd.read_csv(paths["gt_all_csv"])
    gt_by_obs = build_gt_rles_by_obs(gt_df)

    print("Loading 4-view clahe OOF and 12-view 3-family OOF pools...", flush=True)
    pool_clahe = load_fused_oof_pool(folds_df, [("clahe", "work/oof", 4, 0.0)])
    score_grid = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875]
    mask_grid = [0.475, 0.500, 0.525, 0.550, 0.575]
    eval_clahe = evaluate_threshold_grid_nested_cv(pool_clahe, folds_df, gt_by_obs, score_grid, mask_grid)
    base_clahe_per_obs = eval_clahe["nested_cv_per_obs"]
    del pool_clahe
    gc.collect()

    specs_12v = [("clahe", "work/oof", 4, 0.0), ("a5", "work/oof_a5", 4, 0.0), ("dice", "work/oof_dice", 4, 0.0)]
    pool_12v = load_fused_oof_pool(folds_df, specs_12v)
    eval_12v = evaluate_threshold_grid_nested_cv(pool_12v, folds_df, gt_by_obs, score_grid, mask_grid)
    thresh_12v_per_obs = eval_12v["nested_cv_per_obs"]

    multifam_thresh_rles_by_obs: Dict[str, List[str]] = {}
    obs_by_fold = {f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist() for f in range(5)}
    for f in range(5):
        sc_f = eval_12v["nested_cv_folds"][f]["tau_score"]
        mk_f = eval_12v["nested_cv_folds"][f]["tau_mask"]
        for obs in obs_by_fold[f]:
            occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            rles = []
            for c in pool_12v.get(obs, []):
                if c["score"] < sc_f:
                    break
                box = c["box"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bc = (c["prob_crop"] >= mk_f)
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
            multifam_thresh_rles_by_obs[obs] = rles

    # -------------------------------------------------------------------------
    # TASK 4: Strict Nested Selection Across Scorer / tau_mask Variants
    # -------------------------------------------------------------------------
    nested_out, scorer_v2_rles_by_obs, v2_inv_0525_models = run_strict_nested_scorer_selection(
        pool_12v, folds_df, gt_df, gt_by_obs, paths["train_json"]
    )
    boot_nested_vs_clahe = paired_bootstrap_from_per_obs(base_clahe_per_obs, nested_out["nested_per_obs"])
    boot_nested_vs_12v_thresh = paired_bootstrap_from_per_obs(thresh_12v_per_obs, nested_out["nested_per_obs"])
    print("\n  Strict Nested-Selected Scorer (707 OOF Images + Paired Bootstrap):")
    print_boot_line("Nested-Selected Scorer vs 1536_clahe Thresh", nested_out["nested_pooled"], boot_nested_vs_clahe)
    print_boot_line("Nested-Selected Scorer vs 12v MultiFam Thresh", nested_out["nested_pooled"], boot_nested_vs_12v_thresh)

    df_oof_12v_v2 = nested_out["feat_0525"]

    # -------------------------------------------------------------------------
    # TASK 3: Ceiling 296-Image Comparison (vs All Records and vs Others-Only)
    # -------------------------------------------------------------------------
    ceiling_res = run_ceiling_296_comparison(
        gt_all_path=paths["gt_all_csv"],
        clahe_oof_csv="work/oof/oof_predictions.csv",
        multifam_thresh_rles_by_obs=multifam_thresh_rles_by_obs,
        scorer_v2_rles_by_obs=scorer_v2_rles_by_obs,
    )

    # -------------------------------------------------------------------------
    # TASK 1A: Adversarial Validation on Current 32-View 8-Model Test Ensemble vs 12-View OOF
    # -------------------------------------------------------------------------
    first_dir = "work/test_cache/f0_1536"
    test_stems = sorted([Path(p).stem for p in os.listdir(first_dir) if p.endswith(".pt")])
    models_8mod_32v = [
        ("f0_1536", "clahe", 0.0), ("f1_1536", "clahe", 0.0), ("f2_1536", "clahe", 0.0),
        ("f3_1536", "clahe", 0.0), ("f4_1536", "clahe", 0.0), ("f0_2048", "clahe", 0.0),
        ("f0_anchors5", "a5", 0.0), ("f0_dice", "dice", 0.0),
    ]
    df_test_32v, _, _ = extract_test_scorer_v2_features_and_preds(
        test_stems=test_stems,
        model_list=models_8mod_32v,
        num_views_total=32,
        fam_views={"clahe": 24, "a5": 4, "dice": 4},
        scorer_tau_mask=0.525,
    )
    adv_32v_vs_12v = run_adversarial_classifier(
        df_oof_12v_v2, df_test_32v, SCORER_V2_VIEW_INVARIANT_FEATURES, "OOF_12v", "Test_32v"
    )
    print_adv_report(
        "TASK 1A: Scorer v2 View-Invariant (19 Feats) — 12v OOF vs 32v 8-Model Test (Unbalanced 24/4/4)",
        adv_32v_vs_12v, "OOF_12v", "Test_32v", top_k=19,
    )

    if "--skip_gpu_cache" in sys.argv:
        out_summary = {
            "task1_adv_32v_vs_12v": adv_32v_vs_12v,
            "task3_ceiling_296": ceiling_res,
            "task4_nested_selection": {
                "nested_pooled": nested_out["nested_pooled"],
                "nested_folds": nested_out["nested_folds"],
                "boot_vs_clahe": boot_nested_vs_clahe,
                "boot_vs_12v_thresh": boot_nested_vs_12v_thresh,
            },
        }
        with open("work/oof/followup_tasks_summary.json", "w") as f:
            json.dump(out_summary, f, indent=2)
        return

    # -------------------------------------------------------------------------
    # TASK 2 & TASK 1B/1C: Cache 10 Newly Finished Models in 1 Pass & Rebuild Submissions A/B
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  TASK 2: CACHING ALL 18 FINISHED MODELS & REBUILDING SUBMISSIONS A & B")
    print("#" * 92, flush=True)
    ensure_all_finished_models_cached()

    models_all18_72v = [
        ("f0_1536", "clahe", 0.0), ("f1_1536", "clahe", 0.0), ("f2_1536", "clahe", 0.0),
        ("f3_1536", "clahe", 0.0), ("f4_1536", "clahe", 0.0),
        ("f0_2048", "clahe", 0.0), ("f1_2048", "clahe", 0.0), ("f2_2048", "clahe", 0.0),
        ("f0_anchors5", "a5", 0.0), ("f1_anchors5", "a5", 0.0), ("f2_anchors5", "a5", 0.0),
        ("f3_anchors5", "a5", 0.0), ("f4_anchors5", "a5", 0.0),
        ("f0_dice", "dice", 0.0), ("f1_dice", "dice", 0.0), ("f2_dice", "dice", 0.0),
        ("f3_dice", "dice", 0.0), ("f4_dice", "dice", 0.0),
    ]
    df_test_72v, df_sub_A_all18, df_sub_B_all18 = extract_test_scorer_v2_features_and_preds(
        test_stems=test_stems,
        model_list=models_all18_72v,
        num_views_total=72,
        fam_views={"clahe": 32, "a5": 20, "dice": 20},
        scorer_models=v2_inv_0525_models,
        scorer_feature_cols=SCORER_V2_VIEW_INVARIANT_FEATURES,
        scorer_tau=0.2440,
        scorer_tau_mask=0.525,
        thresh_tau_score=0.875,
        thresh_tau_mask=0.550,
    )

    models_15mod_60v = [
        ("f0_1536", "clahe", 0.0), ("f1_1536", "clahe", 0.0), ("f2_1536", "clahe", 0.0),
        ("f3_1536", "clahe", 0.0), ("f4_1536", "clahe", 0.0),
        ("f0_anchors5", "a5", 0.0), ("f1_anchors5", "a5", 0.0), ("f2_anchors5", "a5", 0.0),
        ("f3_anchors5", "a5", 0.0), ("f4_anchors5", "a5", 0.0),
        ("f0_dice", "dice", 0.0), ("f1_dice", "dice", 0.0), ("f2_dice", "dice", 0.0),
        ("f3_dice", "dice", 0.0), ("f4_dice", "dice", 0.0),
    ]
    df_test_60v, df_sub_A_15mod, df_sub_B_15mod = extract_test_scorer_v2_features_and_preds(
        test_stems=test_stems,
        model_list=models_15mod_60v,
        num_views_total=60,
        fam_views={"clahe": 20, "a5": 20, "dice": 20},
        scorer_models=v2_inv_0525_models,
        scorer_feature_cols=SCORER_V2_VIEW_INVARIANT_FEATURES,
        scorer_tau=0.2440,
        scorer_tau_mask=0.525,
        thresh_tau_score=0.875,
        thresh_tau_mask=0.550,
    )

    dl_dir = os.environ.get("SOLAR_EXPORT_DIR") or os.environ.get("DOWNLOADS_DIR")
    sub_files = [
        ("work/oof/submission_A_scorer_v2_all18_finished.csv", df_sub_A_all18),
        ("work/oof/submission_B_threshold_all18_finished.csv", df_sub_B_all18),
        ("work/oof/submission_A_scorer_v2_15model_3fam.csv", df_sub_A_15mod),
        ("work/oof/submission_B_threshold_15model_3fam.csv", df_sub_B_15mod),
    ]
    for path_csv, df_sub in sub_files:
        df_sub.to_csv(path_csv, index=False)
        n_fil = int((df_sub["segmentation_rle"] != EMPTY_RLE).sum())
        print(f"  Saved {path_csv} ({n_fil} predicted filaments across 180 test images)")
        subprocess.run([sys.executable, "scripts/validate_submission.py", "--sub", path_csv], check=True)
        if dl_dir and os.path.isdir(dl_dir):
            shutil.copy2(path_csv, os.path.join(dl_dir, os.path.basename(path_csv)))

    adv_72v_vs_12v = run_adversarial_classifier(
        df_oof_12v_v2, df_test_72v, SCORER_V2_VIEW_INVARIANT_FEATURES, "OOF_12v", "Test_72v"
    )
    print_adv_report(
        "TASK 1B: Scorer v2 View-Invariant (19 Feats) — 12v OOF vs 72v All-18-Finished Test (32/20/20)",
        adv_72v_vs_12v, "OOF_12v", "Test_72v", top_k=19,
    )

    adv_60v_vs_12v = run_adversarial_classifier(
        df_oof_12v_v2, df_test_60v, SCORER_V2_VIEW_INVARIANT_FEATURES, "OOF_12v", "Test_60v"
    )
    print_adv_report(
        "TASK 1C: Scorer v2 View-Invariant (19 Feats) — 12v OOF vs 60v Balanced 15-Model Test (20/20/20)",
        adv_60v_vs_12v, "OOF_12v", "Test_60v", top_k=19,
    )

    out_summary = {
        "task1_adv_32v_vs_12v": adv_32v_vs_12v,
        "task1_adv_72v_vs_12v": adv_72v_vs_12v,
        "task1_adv_60v_vs_12v": adv_60v_vs_12v,
        "task3_ceiling_296": ceiling_res,
        "task4_nested_selection": {
            "nested_pooled": nested_out["nested_pooled"],
            "nested_folds": nested_out["nested_folds"],
            "boot_vs_clahe": boot_nested_vs_clahe,
            "boot_vs_12v_thresh": boot_nested_vs_12v_thresh,
        },
    }
    out_json = "work/oof/followup_tasks_summary.json"
    with open(out_json, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nCompleted all follow-up tasks in {time.time() - t0:.1f}s -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
