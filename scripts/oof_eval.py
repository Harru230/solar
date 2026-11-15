#!/usr/bin/env python3
"""
scripts/oof_eval.py - Out-of-Fold (OOF) raw detection generator, 5-fold threshold tuner,
multi-model Fold-0 evaluator, and multi-checkpoint test-set ensemble generator.

Workflows:
1. --generate:
   - Loads model checkpoint (e.g. checkpoint_fold{k}_last.pt).
   - Runs 4-view Flip TTA (id, h, v, hv) on validation images for fold k.
   - Extracts candidate detections (boxes scaled to 2048, scores, soft probability crops in float16, view_idx).
   - Runs Weighted Mask Fusion (WMF) to generate fused clusters.
   - Saves:
     * Individual per-image raw detections under work/oof/fold{k}/{obs}.pt
     * Consolidated raw detections under work/oof/fold{k}/raw_detections.pt
     * Postprocessed predictions CSV under work/oof/fold{k}/pred_fold{k}_tta.csv
   - Evaluates fold PQ/SQ/RQ against ground truth with disjoint canvas check.

2. --eval:
   - Aggregates predictions across all available folds under work/oof/fold{k}/.
   - Evaluates full Out-Of-Fold Panoptic Quality (PQ, SQ, RQ, TP, FP, FN) on ground truth.
   - Saves work/oof/oof_predictions.csv and work/oof/oof_summary.json.

3. --tune_oof:
   - Loads precomputed fused clusters from work/oof/fold{0..4}/{obs}.pt on CPU.
   - Sweeps (tau_score, tau_mask) grid across all 5 folds.
   - Performs 5-fold Leave-One-Fold-Out (Nested CV) threshold selection to control selection bias.
   - Saves tuned OOF predictions and paired bootstrap comparison.

4. --eval_multimodel_fold0:
   - Evaluates multi-model WMF ensemble on Fold 0 (1536_clahe + anchors5 + dice + 2048) with paired bootstrap.

5. --predict_test_ensemble:
   - Runs multi-checkpoint 4-view Flip TTA + WMF across all 180 test images to produce:
     * work/oof/submission_5fold_1536_tta.csv (5-fold 1536 ensemble, 20 views)
     * work/oof/submission_5fold1536_plus_2048_tta.csv (5-fold 1536 + 2048 ensemble, 24 views)
     * work/oof/submission_full_8model_tta.csv (5-fold 1536 + 2048 + anchors5 + dice, 32 views)
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.bootstrap_compare import run_paired_bootstrap
from scripts.common import EMPTY_RLE, extract_state_dict, find_dataset_paths, get_maskrcnn_model
from scripts.predict import load_image
from scripts.predict_ensemble import (
    FRAME_SIZE,
    apply_flip_transform,
    generate_predictions_for_threshold,
    run_ensemble_ab_sweep,
    unflip_boxes,
    unflip_mask_probabilities,
    weighted_mask_fusion,
)


def extract_4view_candidates_from_tensor(
    model: torch.nn.Module,
    base_tensor: torch.Tensor,
    img_size: int,
    device: torch.device,
    score_filter: float = 0.25,
) -> List[dict]:
    views = [
        ("id", False, False),
        ("h", False, True),
        ("v", True, False),
        ("hv", True, True),
    ]
    scale_to_2048 = 2048.0 / img_size
    all_view_candidates = []

    for v_idx, (v_name, f_v, f_h) in enumerate(views):
        flipped_tensor = apply_flip_transform(base_tensor, flip_v=f_v, flip_h=f_h).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model([flipped_tensor])[0]

        scores = output["scores"]
        keep = scores >= score_filter
        if not keep.any():
            continue

        scores = scores[keep]
        boxes = output["boxes"][keep]
        masks = output["masks"][keep]

        boxes_unflipped = unflip_boxes(boxes, flip_v=f_v, flip_h=f_h, height=img_size, width=img_size)
        masks_unflipped = unflip_mask_probabilities(masks, flip_v=f_v, flip_h=f_h)

        boxes_2048 = (boxes_unflipped * scale_to_2048).detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

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

            crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float16)

            cand = {
                "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                "score": float(scores_np[k]),
                "prob_crop": crop_soft,
                "view_idx": v_idx,
                "view_name": v_name,
            }
            all_view_candidates.append(cand)

    return all_view_candidates


def extract_4view_candidates_for_image(
    model: torch.nn.Module,
    img_path: str,
    img_size: int,
    device: torch.device,
    score_filter: float = 0.25,
) -> List[dict]:
    base_tensor, _ = load_image(img_path, img_size=img_size)
    return extract_4view_candidates_from_tensor(model, base_tensor, img_size, device, score_filter)


def generate_fold_raw_detections(
    checkpoint_path: str,
    fold: int,
    img_size: int = 1536,
    out_dir: Optional[str] = None,
    score_filter: float = 0.25,
    tau_score: float = 0.85,
    tau_mask: float = 0.50,
    device_str: str = "cuda:0",
) -> dict:
    if out_dir is None:
        out_dir = f"work/oof/fold{fold}"
    os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print(f"   GENERATING OOF RAW DETECTIONS (4 FLIPS): FOLD {fold}          ")
    print("==================================================================")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Fold:         {fold}")
    print(f"Image size:   {img_size}")
    print(f"Output dir:   {out_dir}")
    print(f"Score filter: {score_filter}")

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

    print(f"Found {len(val_image_paths)} validation images for fold {fold}.")

    gt_df = pd.read_csv(paths["gt_all_csv"])
    val_obs_set = set(val_obs)
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]
    val_gt = gt_df[gt_img.isin(val_obs_set)].copy()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    ckpt_anchors5 = config.get("anchors5", False)
    if "img_size" in config and config["img_size"]:
        img_size = config["img_size"]

    ckpt_version = config.get("model_version", "v2")
    print(f"Model config: version={ckpt_version}, anchors5={ckpt_anchors5}, img_size={img_size}")
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

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    best_thresh_path = os.path.join(run_dir, "best_thresholds.json")
    if os.path.isfile(best_thresh_path):
        try:
            with open(best_thresh_path, "r") as f:
                thresh_info = json.load(f)
            tau_score = thresh_info.get("best_score_threshold", tau_score)
            tau_mask = thresh_info.get("best_mask_threshold", tau_mask)
            print(f"Loaded tuned thresholds from {best_thresh_path}: tau_score={tau_score}, tau_mask={tau_mask}")
        except Exception as e:
            print(f"Warning: could not read {best_thresh_path}: {e}")

    import gc
    fused_clusters_by_obs = {}

    t0 = time.time()
    print(f"\nRunning 4-view Flip TTA inference on {len(val_image_paths)} images...")

    with torch.no_grad():
        for img_idx, (obs, img_path) in enumerate(val_image_paths, start=1):
            if img_idx % 25 == 0 or img_idx == len(val_image_paths):
                print(f"  Processed {img_idx}/{len(val_image_paths)} images ({time.time() - t0:.1f}s)...")

            all_view_candidates = extract_4view_candidates_for_image(
                model=model,
                img_path=img_path,
                img_size=img_size,
                device=device,
                score_filter=score_filter,
            )

            fused = weighted_mask_fusion(all_view_candidates, num_views=4, iou_thresh=0.40)

            per_img_path = os.path.join(out_dir, f"{obs}.pt")
            torch.save({"obs": obs, "candidates": all_view_candidates, "fused": fused}, per_img_path)
            try:
                fd = os.open(per_img_path, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except Exception:
                pass

            # Only retain clusters above tau_score in RAM for immediate CSV generation
            fused_clusters_by_obs[obs] = [c for c in fused if c["score"] >= tau_score]
            del all_view_candidates, fused

            if img_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nInference & Fusion completed in {elapsed:.1f}s.")

    df_pred = generate_predictions_for_threshold(fused_clusters_by_obs, tau_score, tau_mask)
    pred_csv_path = os.path.join(out_dir, f"pred_fold{fold}_tta.csv")
    df_pred.to_csv(pred_csv_path, index=False)
    print(f"Saved postprocessed predictions to: {pred_csv_path}")

    res = pq_eval.score(val_gt, df_pred[df_pred["segmentation_rle"] != EMPTY_RLE])
    print("\n------------------------------------------------------------------")
    print(f"Fold {fold} OOF Evaluation (Score {tau_score:.2f}, Mask {tau_mask:.2f}):")
    print(f"PQ: {res['pq']:.4f} | SQ: {res['sq']:.4f} | RQ: {res['rq']:.4f}")
    print(f"TP: {res['tp']} | FP: {res['fp']} | FN: {res['fn']}")
    print(f"gt_multi_hit: {res['gt_multi_hit']} | pred_multi_hit: {res['pred_multi_hit']}")
    print("------------------------------------------------------------------")

    summary_path = os.path.join(out_dir, f"oof_fold{fold}_summary.json")
    out_dict = {
        "fold": fold,
        "checkpoint": checkpoint_path,
        "img_size": img_size,
        "tau_score": tau_score,
        "tau_mask": tau_mask,
        "metrics": res,
        "elapsed_sec": elapsed,
        "num_images": len(val_image_paths),
        "total_detections": len(df_pred),
    }
    with open(summary_path, "w") as f:
        json.dump(out_dict, f, indent=2)

    return out_dict


def evaluate_oof(oof_dir: str = "work/oof") -> dict:
    print("==================================================================")
    print("           FULL OUT-OF-FOLD (OOF) ENSEMBLE EVALUATION             ")
    print("==================================================================")
    print(f"Scanning directory: {oof_dir}")

    paths = find_dataset_paths()
    gt_df = pd.read_csv(paths["gt_all_csv"])
    folds_df = pd.read_csv(paths["folds_csv"])

    fold_dfs = []
    found_folds = []
    expected_stems = set()

    for f in range(5):
        fold_dir = os.path.join(oof_dir, f"fold{f}")
        pred_csv = os.path.join(fold_dir, f"pred_fold{f}_tta.csv")
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        if os.path.isfile(pred_csv):
            df_k = pd.read_csv(pred_csv)
            present_stems = set(df_k["filament_id"].str.split("_", n=1).str[0])
            missing_stems = [s for s in val_obs if s not in present_stems]
            if missing_stems:
                pad_rows = [{"filament_id": f"{s}_1", "segmentation_rle": EMPTY_RLE} for s in missing_stems]
                df_k = pd.concat([df_k, pd.DataFrame(pad_rows)], ignore_index=True)
                df_k.to_csv(pred_csv, index=False)
                print(f"  [Padded] Fold {f}: added {len(missing_stems)} EMPTY_RLE rows to {pred_csv}")
            fold_dfs.append(df_k)
            found_folds.append(f)
            expected_stems.update(val_obs)
            print(f"  [Found] Fold {f}: {len(df_k)} prediction rows ({len(val_obs)} stems) from {pred_csv}")
        else:
            print(f"  [Missing] Fold {f}: {pred_csv} not found.")

    if not fold_dfs:
        print(f"Error: No fold prediction CSVs found under {oof_dir}.")
        return {}

    full_oof_df = pd.concat(fold_dfs, ignore_index=True)
    out_all_csv = os.path.join(oof_dir, "oof_predictions.csv")
    full_oof_df.to_csv(out_all_csv, index=False)
    print(f"\nConcatenated {len(full_oof_df)} OOF rows across {len(found_folds)} fold(s) -> {out_all_csv}")

    oof_stems = set(full_oof_df["filament_id"].str.split("_", n=1).str[0])
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]
    gt_subset = gt_df[gt_img.isin(expected_stems)].copy()

    non_empty_df = full_oof_df[full_oof_df["segmentation_rle"] != EMPTY_RLE].copy()
    non_empty_stems = set(non_empty_df["filament_id"].str.split("_", n=1).str[0])
    res = pq_eval.score(gt_subset, non_empty_df)
    res_non_empty = pq_eval.score(gt_df[gt_img.isin(non_empty_stems)].copy(), non_empty_df)

    print("\n==================================================================")
    print(f"         FULL OOF METRICS ({len(found_folds)}/5 FOLDS, {len(expected_stems)} STEMS)       ")
    print("==================================================================")
    print(f"Panoptic Quality (PQ, all {len(expected_stems)} imgs): {res['pq']:.4f} (694 non-empty imgs: {res_non_empty['pq']:.4f})")
    print(f"Segmentation Quality (SQ):            {res['sq']:.4f}")
    print(f"Recognition Quality (RQ):             {res['rq']:.4f}")
    print(f"True Positives (TP):                  {res['tp']}")
    print(f"False Positives (FP):                 {res['fp']}")
    print(f"False Negatives (FN):                 {res['fn']} (694 non-empty imgs: {res_non_empty['fn']})")
    print(f"gt_multi_hit:                         {res['gt_multi_hit']}")
    print(f"pred_multi_hit:                       {res['pred_multi_hit']}")
    print("==================================================================")

    summary_path = os.path.join(oof_dir, "oof_summary.json")
    out_dict = {
        "folds_evaluated": found_folds,
        "num_stems": len(expected_stems),
        "num_non_empty_stems": len(non_empty_stems),
        "total_rows": len(full_oof_df),
        "metrics": res,
        "metrics_non_empty_stems_only": res_non_empty,
    }
    with open(summary_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved full OOF summary to: {summary_path}")

    return out_dict


def tune_oof_thresholds(oof_dir: str = "work/oof") -> dict:
    """
    Loads precomputed fused clusters across all 5 folds in work/oof/fold{0..4}/,
    sweeps (tau_score, tau_mask), and performs 5-fold Nested CV (leave-one-fold-out)
    plus global optimization.
    """
    print("==================================================================")
    print("       5-FOLD OOF GLOBAL & NESTED-CV THRESHOLD OPTIMIZATION       ")
    print("==================================================================")
    paths = find_dataset_paths()
    gt_df = pd.read_csv(paths["gt_all_csv"])
    folds_df = pd.read_csv(paths["folds_csv"])

    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]

    fused_by_fold: Dict[int, Dict[str, List[dict]]] = {}
    gt_by_fold: Dict[int, pd.DataFrame] = {}

    t0 = time.time()
    for f in range(5):
        fold_dir = os.path.join(oof_dir, f"fold{f}")
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        val_obs_set = set(val_obs)
        gt_by_fold[f] = gt_df[gt_img.isin(val_obs_set)].copy()

        fold_fused = {}
        for obs in val_obs:
            pt_path = os.path.join(fold_dir, f"{obs}.pt")
            if os.path.isfile(pt_path):
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                fold_fused[obs] = data["fused"]
        fused_by_fold[f] = fold_fused
        print(f"  Loaded Fold {f}: {len(fold_fused)}/{len(val_obs)} stems ({time.time() - t0:.1f}s)")

    score_grid = [0.65, 0.70, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875]
    mask_grid = [0.45, 0.50, 0.525, 0.55, 0.60]

    # Store per-fold metrics and predictions for each grid point
    # grid_stats[(sc, mk)][f] = res_dict
    grid_stats: Dict[Tuple[float, float], Dict[int, dict]] = {}
    sweep_rows = []

    print(f"\nSweeping {len(score_grid)} x {len(mask_grid)} = {len(score_grid)*len(mask_grid)} threshold pairs across 5 folds...")
    for mk in mask_grid:
        # Pre-generate binary crops at tau_mask = mk to evaluate multiple tau_score quickly
        for sc in score_grid:
            grid_stats[(sc, mk)] = {}
            tot_iou = 0.0
            tot_tp = 0
            tot_fp = 0
            tot_fn = 0
            tot_gm = 0
            tot_pm = 0

            for f in range(5):
                df_f = generate_predictions_for_threshold(fused_by_fold[f], tau_score=sc, tau_mask=mk)
                res_f = pq_eval.score(gt_by_fold[f], df_f[df_f["segmentation_rle"] != EMPTY_RLE])
                grid_stats[(sc, mk)][f] = res_f
                tot_iou += res_f["sq"] * res_f["tp"]
                tot_tp += res_f["tp"]
                tot_fp += res_f["fp"]
                tot_fn += res_f["fn"]
                tot_gm += res_f["gt_multi_hit"]
                tot_pm += res_f["pred_multi_hit"]

            denom = tot_tp + 0.5 * tot_fp + 0.5 * tot_fn
            pq_all = tot_iou / denom if denom > 0 else 0.0
            sq_all = tot_iou / tot_tp if tot_tp > 0 else 0.0
            rq_all = tot_tp / denom if denom > 0 else 0.0

            row = {
                "tau_score": sc,
                "tau_mask": mk,
                "pq_5fold": pq_all,
                "sq_5fold": sq_all,
                "rq_5fold": rq_all,
                "tp": tot_tp,
                "fp": tot_fp,
                "fn": tot_fn,
                "gt_multi_hit": tot_gm,
                "pred_multi_hit": tot_pm,
                "pq_f0": grid_stats[(sc, mk)][0]["pq"],
                "pq_f1": grid_stats[(sc, mk)][1]["pq"],
                "pq_f2": grid_stats[(sc, mk)][2]["pq"],
                "pq_f3": grid_stats[(sc, mk)][3]["pq"],
                "pq_f4": grid_stats[(sc, mk)][4]["pq"],
            }
            sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows).sort_values("pq_5fold", ascending=False)
    sweep_csv_path = os.path.join(oof_dir, "oof_threshold_sweep.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)

    best_global = sweep_df.iloc[0].to_dict()
    best_sc = float(best_global["tau_score"])
    best_mk = float(best_global["tau_mask"])

    print("\nTop 5 Global 5-Fold OOF Thresholds:")
    print(sweep_df[["tau_score", "tau_mask", "pq_5fold", "sq_5fold", "rq_5fold", "tp", "fp", "fn"]].head(5).to_string(index=False))

    # 5-Fold Nested CV (Leave-One-Fold-Out threshold selection)
    nested_cv_folds = []
    nested_dfs = []
    for held_out_f in range(5):
        best_train_pq = -1.0
        best_train_pair = (0.85, 0.50)
        for (sc, mk), f_dict in grid_stats.items():
            tr_iou = sum(f_dict[j]["sq"] * f_dict[j]["tp"] for j in range(5) if j != held_out_f)
            tr_tp = sum(f_dict[j]["tp"] for j in range(5) if j != held_out_f)
            tr_fp = sum(f_dict[j]["fp"] for j in range(5) if j != held_out_f)
            tr_fn = sum(f_dict[j]["fn"] for j in range(5) if j != held_out_f)
            tr_denom = tr_tp + 0.5 * tr_fp + 0.5 * tr_fn
            tr_pq = tr_iou / tr_denom if tr_denom > 0 else 0.0
            if tr_pq > best_train_pq:
                best_train_pq = tr_pq
                best_train_pair = (sc, mk)

        sel_sc, sel_mk = best_train_pair
        held_res = grid_stats[(sel_sc, sel_mk)][held_out_f]
        nested_cv_folds.append({
            "held_out_fold": held_out_f,
            "selected_tau_score": sel_sc,
            "selected_tau_mask": sel_mk,
            "train_4fold_pq": best_train_pq,
            "held_out_pq": held_res["pq"],
            "held_out_sq": held_res["sq"],
            "held_out_rq": held_res["rq"],
        })
        df_held = generate_predictions_for_threshold(fused_by_fold[held_out_f], tau_score=sel_sc, tau_mask=sel_mk)
        nested_dfs.append(df_held)

    nested_oof_df = pd.concat(nested_dfs, ignore_index=True)
    nested_csv = os.path.join(oof_dir, "oof_predictions_nested_cv.csv")
    nested_oof_df.to_csv(nested_csv, index=False)
    nested_res = pq_eval.score(gt_df, nested_oof_df[nested_oof_df["segmentation_rle"] != EMPTY_RLE])

    # Save global best OOF predictions
    global_dfs = [
        generate_predictions_for_threshold(fused_by_fold[f], tau_score=best_sc, tau_mask=best_mk)
        for f in range(5)
    ]
    global_oof_df = pd.concat(global_dfs, ignore_index=True)
    global_csv = os.path.join(oof_dir, "oof_predictions_tuned.csv")
    global_oof_df.to_csv(global_csv, index=False)
    global_res = pq_eval.score(gt_df, global_oof_df[global_oof_df["segmentation_rle"] != EMPTY_RLE])

    print("\n==================================================================")
    print("           5-FOLD OOF THRESHOLD TUNING SUMMARY                    ")
    print("==================================================================")
    print(f"Default OOF PQ (0.85 / 0.50):        0.4344")
    print(f"Global Best Thresholds:              tau_score = {best_sc:.3f}, tau_mask = {best_mk:.3f}")
    print(f"Global Best 5-Fold OOF PQ:           {global_res['pq']:.4f} (SQ: {global_res['sq']:.4f}, RQ: {global_res['rq']:.4f})")
    print(f"Nested-CV (Out-of-Fold Tuned) PQ:    {nested_res['pq']:.4f} (SQ: {nested_res['sq']:.4f}, RQ: {nested_res['rq']:.4f})")
    for nf in nested_cv_folds:
        print(f"  Fold {nf['held_out_fold']}: selected ({nf['selected_tau_score']:.3f}, {nf['selected_tau_mask']:.3f}) -> PQ = {nf['held_out_pq']:.4f}")
    print("==================================================================")

    summary_path = os.path.join(oof_dir, "oof_tuning_summary.json")
    out_summary = {
        "best_tau_score": best_sc,
        "best_tau_mask": best_mk,
        "global_best_metrics": global_res,
        "nested_cv_metrics": nested_res,
        "nested_cv_folds": nested_cv_folds,
    }
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"Saved OOF tuning summary to: {summary_path}")

    return out_summary


def eval_multimodel_fold0(device_str: str = "cuda:0") -> dict:
    """
    Evaluates a Multi-Model WMF Ensemble on Fold 0 combining:
    - r50_1536_clahe_e30 (best.pt)
    - r50_1536_anchors5_e30 (best.pt)
    - r50_1536_dice_e30 (best.pt)
    - r50_2048_all_e30 (best.pt)
    """
    print("==================================================================")
    print("   FOLD 0 MULTI-MODEL WMF ENSEMBLE EVALUATION (4 MODELS x 4 TTA)  ")
    print("==================================================================")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    val_obs = folds_df.loc[folds_df["fold"] == 0, "observation"].tolist()
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

    model_specs = [
        ("1536_clahe", "work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("1536_anchors5", "work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("1536_dice", "work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("2048_all", "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt", 2048),
    ]

    loaded_models = []
    for name, ckpt_path, sz in model_specs:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        a5 = cfg.get("anchors5", False)
        m = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=a5)
        m.load_state_dict(extract_state_dict(ckpt))
        m.to(device).eval()
        loaded_models.append((name, m, sz))
        print(f"  Loaded {name} ({ckpt_path}, img_size={sz}, anchors5={a5})")

    # We will evaluate two combinations on Fold 0:
    # Combo 1: 1536_clahe + 2048_all (2 models x 4 views = 8 views)
    # Combo 2: All 4 Fold-0 models (4 models x 4 views = 16 views)
    fused_2mod_by_obs = {}
    fused_4mod_by_obs = {}

    t0 = time.time()
    with torch.no_grad():
        for idx, (obs, img_path) in enumerate(val_image_paths, start=1):
            if idx % 25 == 0 or idx == len(val_image_paths):
                print(f"  Processed {idx}/{len(val_image_paths)} Fold-0 images ({time.time() - t0:.1f}s)...")

            cands_2mod = []
            cands_4mod = []
            for name, m, sz in loaded_models:
                cands = extract_4view_candidates_for_image(m, img_path, sz, device, score_filter=0.25)
                cands_4mod.extend(cands)
                if name in ("1536_clahe", "2048_all"):
                    cands_2mod.extend(cands)

            fused_2mod_by_obs[obs] = weighted_mask_fusion(cands_2mod, num_views=8, iou_thresh=0.40)
            fused_4mod_by_obs[obs] = weighted_mask_fusion(cands_4mod, num_views=16, iou_thresh=0.40)

            if idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    sweep_2mod = run_ensemble_ab_sweep(fused_2mod_by_obs, val_gt, seed=2026)
    best_2 = sweep_2mod["best_full"]
    df_2mod = generate_predictions_for_threshold(fused_2mod_by_obs, best_2["tau_score"], best_2["tau_mask"])
    csv_2mod = "work/oof/pred_fold0_ensemble_1536_2048.csv"
    df_2mod.to_csv(csv_2mod, index=False)

    sweep_4mod = run_ensemble_ab_sweep(fused_4mod_by_obs, val_gt, seed=2026)
    best_4 = sweep_4mod["best_full"]
    df_4mod = generate_predictions_for_threshold(fused_4mod_by_obs, best_4["tau_score"], best_4["tau_mask"])
    csv_4mod = "work/oof/pred_fold0_ensemble_4models.csv"
    df_4mod.to_csv(csv_4mod, index=False)

    boot_2_vs_2048 = run_paired_bootstrap(
        pred_a_path="work/runs/r50_2048_all_e30/pred_fold0_tta_wmf.csv",
        pred_b_path=csv_2mod,
        gt_path=paths["gt_all_csv"],
        folds_path=paths["folds_csv"],
        fold=0,
        n_boot=1000,
        seed=2026,
    )
    boot_4_vs_2048 = run_paired_bootstrap(
        pred_a_path="work/runs/r50_2048_all_e30/pred_fold0_tta_wmf.csv",
        pred_b_path=csv_4mod,
        gt_path=paths["gt_all_csv"],
        folds_path=paths["folds_csv"],
        fold=0,
        n_boot=1000,
        seed=2026,
    )

    print("\n==================================================================")
    print("         FOLD 0 MULTI-MODEL WMF ENSEMBLE RESULTS                  ")
    print("==================================================================")
    print(f"Single 1536 TTA+WMF Baseline PQ:          0.4377")
    print(f"Single 2048 TTA+WMF Baseline PQ:          0.4446")
    print(f"2-Model (1536 + 2048) WMF PQ:             {best_2['pq']:.4f} (sc={best_2['tau_score']:.2f}, mk={best_2['tau_mask']:.2f}, A->B={sweep_2mod['eval_A_on_B']['pq']:.4f}, B->A={sweep_2mod['eval_B_on_A']['pq']:.4f})")
    print(f"  vs 2048 TTA: Delta={boot_2_vs_2048['point_delta']:+.4f}, 95% CI=[{boot_2_vs_2048['ci_lower']:+.4f}, {boot_2_vs_2048['ci_upper']:+.4f}], P={boot_2_vs_2048['p_b_better']*100:.1f}%")
    print(f"4-Model (1536+a5+dice+2048) WMF PQ:       {best_4['pq']:.4f} (sc={best_4['tau_score']:.2f}, mk={best_4['tau_mask']:.2f}, A->B={sweep_4mod['eval_A_on_B']['pq']:.4f}, B->A={sweep_4mod['eval_B_on_A']['pq']:.4f})")
    print(f"  vs 2048 TTA: Delta={boot_4_vs_2048['point_delta']:+.4f}, 95% CI=[{boot_4_vs_2048['ci_lower']:+.4f}, {boot_4_vs_2048['ci_upper']:+.4f}], P={boot_4_vs_2048['p_b_better']*100:.1f}%")
    print("==================================================================")

    out = {
        "combo_2model_1536_2048": {
            "best_full": best_2,
            "eval_A_on_B": sweep_2mod["eval_A_on_B"],
            "eval_B_on_A": sweep_2mod["eval_B_on_A"],
            "bootstrap_vs_2048": {
                "point_delta": boot_2_vs_2048["point_delta"],
                "ci_lower": boot_2_vs_2048["ci_lower"],
                "ci_upper": boot_2_vs_2048["ci_upper"],
                "p_b_better": boot_2_vs_2048["p_b_better"],
            },
        },
        "combo_4model_all_fold0": {
            "best_full": best_4,
            "eval_A_on_B": sweep_4mod["eval_A_on_B"],
            "eval_B_on_A": sweep_4mod["eval_B_on_A"],
            "bootstrap_vs_2048": {
                "point_delta": boot_4_vs_2048["point_delta"],
                "ci_lower": boot_4_vs_2048["ci_lower"],
                "ci_upper": boot_4_vs_2048["ci_upper"],
                "p_b_better": boot_4_vs_2048["p_b_better"],
            },
        },
    }
    with open("work/oof/multimodel_fold0_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def predict_test_ensemble(
    tau_score_5fold: float = 0.80,
    tau_mask_5fold: float = 0.50,
    device_str: str = "cuda:0",
    cache_dir: str = "work/test_cache",
) -> None:
    """
    Generates test-set submissions across all 180 test images for:
    1) 5-Fold 1536 Ensemble (f0..f4, 20 views) -> work/oof/submission_5fold_1536_tta.csv
    2) 5-Fold 1536 + 2048 Ensemble (6 models, 24 views) -> work/oof/submission_5fold1536_plus_2048_tta.csv
    3) Full 8-Model Ensemble (f0..f4 + 2048 + anchors5 + dice, 32 views) -> work/oof/submission_full_8model_tta.csv
    Caches per-model 4-flip candidate detections under work/test_cache/<model_name>/<stem>.pt
    so new model combinations or thresholds can be recomputed on CPU without GPU.
    """
    print("==================================================================")
    print("   GENERATING TEST-SET MULTI-MODEL WMF ENSEMBLE SUBMISSIONS       ")
    print("==================================================================")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    paths = find_dataset_paths()
    test_dir = paths["test_dir"]

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    test_stems_and_paths = [(Path(p).stem, p) for p in test_files]
    all_test_stems = [s for s, _ in test_stems_and_paths]
    print(f"Found {len(test_stems_and_paths)} test images in {test_dir}")

    # Load tuned thresholds if oof_tuning_summary.json exists
    tuning_json = "work/oof/oof_tuning_summary.json"
    if os.path.isfile(tuning_json):
        with open(tuning_json, "r") as f:
            t_info = json.load(f)
        tau_score_5fold = float(t_info.get("best_tau_score", tau_score_5fold))
        tau_mask_5fold = float(t_info.get("best_tau_mask", tau_mask_5fold))
        print(f"Using 5-fold OOF tuned thresholds: tau_score={tau_score_5fold:.3f}, tau_mask={tau_mask_5fold:.3f}")

    # Check if multimodel_fold0_eval.json has tuned thresholds for multi-model
    tau_sc_6mod, tau_mk_6mod = tau_score_5fold, tau_mask_5fold
    tau_sc_8mod, tau_mk_8mod = tau_score_5fold, tau_mask_5fold
    mm_json = "work/oof/multimodel_fold0_eval.json"
    if os.path.isfile(mm_json):
        with open(mm_json, "r") as f:
            mm_info = json.load(f)
        b2 = mm_info.get("combo_2model_1536_2048", {}).get("best_full", {})
        b4 = mm_info.get("combo_4model_all_fold0", {}).get("best_full", {})
        if b2:
            tau_sc_6mod = float(b2.get("tau_score", tau_sc_6mod))
            tau_mk_6mod = float(b2.get("tau_mask", tau_mk_6mod))
        if b4:
            tau_sc_8mod = float(b4.get("tau_score", tau_sc_8mod))
            tau_mk_8mod = float(b4.get("tau_mask", tau_mk_8mod))

    model_specs = [
        ("f0_1536", "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt", 1536),
        ("f1_1536", "work/runs/r50_1536_all_f1/checkpoints/checkpoint_fold1_last.pt", 1536),
        ("f2_1536", "work/runs/r50_1536_all_f2/checkpoints/checkpoint_fold2_last.pt", 1536),
        ("f3_1536", "work/runs/r50_1536_all_f3/checkpoints/checkpoint_fold3_last.pt", 1536),
        ("f4_1536", "work/runs/r50_1536_all_f4/checkpoints/checkpoint_fold4_last.pt", 1536),
        ("f0_2048", "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt", 2048),
        ("f0_anchors5", "work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
        ("f0_dice", "work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt", 1536),
    ]

    loaded_models = []
    for name, ckpt_path, sz in model_specs:
        model_cache_dir = os.path.join(cache_dir, name)
        os.makedirs(model_cache_dir, exist_ok=True)
        missing_cache = [
            s for s in all_test_stems
            if not os.path.isfile(os.path.join(model_cache_dir, f"{s}.pt"))
        ]
        if not missing_cache:
            loaded_models.append((name, None, sz))
            print(f"  [Cache Complete] {name}: all {len(all_test_stems)} test stems cached in {model_cache_dir}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {})
            a5 = cfg.get("anchors5", False)
            m = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=a5)
            m.load_state_dict(extract_state_dict(ckpt))
            m.to(device).eval()
            loaded_models.append((name, m, sz))
            print(f"  [Loaded GPU] {name} ({ckpt_path}) - {len(missing_cache)} stems to compute")

    fused_5fold = {}
    fused_6mod = {}
    fused_8mod = {}

    t0 = time.time()
    with torch.no_grad():
        for idx, (stem, img_path) in enumerate(test_stems_and_paths, start=1):
            if idx % 20 == 0 or idx == len(test_stems_and_paths):
                print(f"  Test progress: [{idx:04d}/{len(test_stems_and_paths):04d}] ({time.time() - t0:.1f}s)...")

            base_1536 = None
            base_2048 = None

            cands_5fold = []
            cands_6mod = []
            cands_8mod = []

            for name, m, sz in loaded_models:
                cache_pt = os.path.join(cache_dir, name, f"{stem}.pt")
                if os.path.isfile(cache_pt):
                    cands = torch.load(cache_pt, map_location="cpu", weights_only=False)["candidates"]
                else:
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

                cands_8mod.extend(cands)
                if name in ("f0_1536", "f1_1536", "f2_1536", "f3_1536", "f4_1536"):
                    cands_5fold.extend(cands)
                    cands_6mod.extend(cands)
                elif name == "f0_2048":
                    cands_6mod.extend(cands)

            fused_5fold[stem] = weighted_mask_fusion(cands_5fold, num_views=20, iou_thresh=0.40)
            fused_6mod[stem] = weighted_mask_fusion(cands_6mod, num_views=24, iou_thresh=0.40)
            fused_8mod[stem] = weighted_mask_fusion(cands_8mod, num_views=32, iou_thresh=0.40)

            if idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    os.makedirs("work/oof", exist_ok=True)
    sub_5fold = "work/oof/submission_5fold_1536_tta.csv"
    sub_6mod = "work/oof/submission_5fold1536_plus_2048_tta.csv"
    sub_8mod = "work/oof/submission_full_8model_tta.csv"

    df_5fold = generate_predictions_for_threshold(fused_5fold, tau_score_5fold, tau_mask_5fold)
    df_5fold.to_csv(sub_5fold, index=False)
    print(f"Saved 5-Fold 1536 submission ({len(df_5fold)} rows) -> {sub_5fold}")

    df_6mod = generate_predictions_for_threshold(fused_6mod, tau_sc_6mod, tau_mk_6mod)
    df_6mod.to_csv(sub_6mod, index=False)
    print(f"Saved 5-Fold 1536 + 2048 submission ({len(df_6mod)} rows) -> {sub_6mod}")

    df_8mod = generate_predictions_for_threshold(fused_8mod, tau_sc_8mod, tau_mk_8mod)
    df_8mod.to_csv(sub_8mod, index=False)
    print(f"Saved Full 8-Model submission ({len(df_8mod)} rows) -> {sub_8mod}")


def main():
    parser = argparse.ArgumentParser(description="OOF Raw Detection Generator, Tuner, and Ensemble Evaluator")
    parser.add_argument("--generate", action="store_true", help="Generate raw detections for a fold")
    parser.add_argument("--eval", action="store_true", help="Evaluate full OOF predictions across folds")
    parser.add_argument("--tune_oof", action="store_true", help="Tune global and nested-CV thresholds on 5-fold OOF")
    parser.add_argument("--eval_multimodel_fold0", action="store_true", help="Evaluate multi-model WMF ensemble on Fold 0")
    parser.add_argument("--predict_test_ensemble", action="store_true", help="Generate multi-model ensemble test submissions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--img_size", type=int, default=1536, help="Inference image resolution")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for detections")
    parser.add_argument("--oof_dir", type=str, default="work/oof", help="OOF root directory")
    parser.add_argument("--score_filter", type=float, default=0.25, help="Proposal score filter threshold")
    parser.add_argument("--tau_score", type=float, default=0.85, help="Postprocessing score threshold")
    parser.add_argument("--tau_mask", type=float, default=0.50, help="Postprocessing mask threshold")
    args = parser.parse_args()

    ran_any = False
    if args.generate:
        if not args.checkpoint:
            print("Error: --checkpoint required for --generate.")
            sys.exit(1)
        generate_fold_raw_detections(
            checkpoint_path=args.checkpoint,
            fold=args.fold,
            img_size=args.img_size,
            out_dir=args.out_dir,
            score_filter=args.score_filter,
            tau_score=args.tau_score,
            tau_mask=args.tau_mask,
        )
        ran_any = True

    if args.eval:
        evaluate_oof(oof_dir=args.oof_dir)
        ran_any = True

    if args.tune_oof:
        tune_oof_thresholds(oof_dir=args.oof_dir)
        ran_any = True

    if args.eval_multimodel_fold0:
        eval_multimodel_fold0()
        ran_any = True

    if args.predict_test_ensemble:
        predict_test_ensemble(tau_score_5fold=args.tau_score, tau_mask_5fold=args.tau_mask)
        ran_any = True

    if not ran_any:
        parser.print_help()


if __name__ == "__main__":
    main()
