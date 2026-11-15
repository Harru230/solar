#!/usr/bin/env python3
"""
scripts/instance_scorer.py - Cross-fitted LightGBM Instance Scorer (Decision Layer)
on Out-of-Fold (OOF) WMF candidate clusters.

Design (strictly following specification):
- Candidate pool: fused 4-flip WMF clusters at LOW cluster score (S >= 0.15);
  canvas disjointness is applied ONLY AFTER selection.
- Target:
    y = (1 / K) * sum_{r=1..K} IoU_r * 1[IoU_r > 0.5]
  where K is the number of annotator records for the image, and IoU_r is the IoU
  against the best matching GT instance of annotator record r.
- Features (strictly geometry and probability only; NO observatory or filename features):
  1. cluster_score: WMF cluster score S = sum(s_k) / M
  2. num_views: number of constituent member detections (normalized to 4-view scale)
  3. member_score_mean: mean of constituent member detection scores
  4. member_score_std: std of constituent member detection scores
  5. prob_mean: mean of fused probability inside binary mask (P >= 0.50)
  6. prob_std: std of fused probability inside binary mask (P >= 0.50)
  7. prob_sharpness_core: ratio of core pixels (P >= 0.75) to mask pixels (P >= 0.50)
  8. prob_sharpness_iqr: P_q90 - P_q10 inside binary mask (P >= 0.50)
  9. log_area: log10(area + 1) of binary mask (P >= 0.50)
  10. bbox_fill: mask area / (bbox_w * bbox_h)
  11. aspect_ratio: max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
  12. rank_in_image: 1-based rank of candidate in image by cluster_score descending
  13. rel_score_to_max: cluster_score / max(cluster_score in image)
  14. n_cands_gt_05: number of candidates with S > 0.50 in the image
- Model:
  LightGBM regressor (max_depth=3, num_leaves=7, min_child_samples=50, n_estimators=300),
  cross-fitted by fold (train on 4 folds, predict the 5th).
- Selection & Disjoint Postprocessing:
  Keep candidates with predicted y_hat > tau, where tau is chosen on the 4 training
  folds only over grid PQ/2 +- 0.04; resolve overlaps in descending order of y_hat
  with canvas painting (>= 50% area retained, min_area >= 10).
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

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, find_dataset_paths
from scripts.predict_ensemble import compute_crop_mask_iou

FRAME_SIZE = 2048

FEATURE_NAMES = [
    "cluster_score",
    "num_views",
    "member_score_mean",
    "member_score_std",
    "prob_mean",
    "prob_std",
    "prob_sharpness_core",
    "prob_sharpness_iqr",
    "log_area",
    "bbox_fill",
    "aspect_ratio",
    "rank_in_image",
    "rel_score_to_max",
    "n_cands_gt_05",
]


def extract_member_scores_from_raw(
    candidate_list: List[dict],
    iou_thresh: float = 0.40,
) -> List[List[float]]:
    """
    Runs the exact same greedy clustering order as weighted_mask_fusion
    to recover the list of constituent member scores for each cluster.
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

    # Sort clusters by sum(scores) descending to match weighted_mask_fusion output order
    cluster_scores_and_members = []
    for cluster in clusters:
        scores = [float(c["score"]) for c in cluster]
        cluster_scores_and_members.append((sum(scores), scores))

    cluster_scores_and_members.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in cluster_scores_and_members]


def encode_crop_to_rle(box: np.ndarray, binary_crop: np.ndarray) -> str:
    """Encodes a binary crop within box [x1, y1, x2, y2] into a 2048x2048 COCO RLE string."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    full_mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8, order="F")
    full_mask[y1:y2, x1:x2] = binary_crop.astype(np.uint8)
    return mu.encode(full_mask)["counts"].decode("ascii")


def build_gt_lookup(gt_df: pd.DataFrame, train_json_path: str) -> Tuple[Dict[str, int], Dict[str, Dict[str, List[str]]]]:
    """
    Builds per-observation annotator count K and per-annotator-record GT RLE lists.
    Uses train.json to know the exact set of K annotator records per observation
    (including annotator records with 0 filaments), and gt_all.csv for the RLEs.
    """
    with open(train_json_path, "r") as f:
        coco = json.load(f)

    records_by_obs: Dict[str, List[str]] = defaultdict(list)
    for im in coco["images"]:
        ai = im["id"]  # "<annotator>-<obs>"
        obs = ai.split("-", 1)[1]
        records_by_obs[obs].append(ai)

    rles_by_ai: Dict[str, List[str]] = defaultdict(list)
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    for ai, group in gt_df.groupby(gt_ai, sort=False):
        rles_by_ai[ai] = group["segmentation_rle"].tolist()

    k_by_obs: Dict[str, int] = {}
    gt_records_by_obs: Dict[str, Dict[str, List[str]]] = {}
    for obs, ai_list in records_by_obs.items():
        k_by_obs[obs] = len(ai_list)
        gt_records_by_obs[obs] = {ai: rles_by_ai.get(ai, []) for ai in ai_list}

    return k_by_obs, gt_records_by_obs


def compute_candidate_targets(
    cand_rles: List[str],
    obs: str,
    k_by_obs: Dict[str, int],
    gt_records_by_obs: Dict[str, Dict[str, List[str]]],
) -> np.ndarray:
    """
    Computes target y for all candidates in an observation:
      y = (1 / K) * sum_{r=1..K} IoU_r * 1[IoU_r > 0.5]
    where IoU_r is the maximum IoU of the candidate against the GT instances
    of annotator record r.
    """
    n_cands = len(cand_rles)
    if n_cands == 0:
        return np.zeros(0, dtype=np.float32)

    K = k_by_obs.get(obs, 1)
    records = gt_records_by_obs.get(obs, {})
    if not records or K == 0:
        return np.zeros(n_cands, dtype=np.float32)

    y_sum = np.zeros(n_cands, dtype=np.float64)
    cand_rle_objs = [{"size": [FRAME_SIZE, FRAME_SIZE], "counts": r.encode("ascii")} for r in cand_rles]

    for ai, gt_rles in records.items():
        if not gt_rles:
            continue
        gt_rle_objs = [{"size": [FRAME_SIZE, FRAME_SIZE], "counts": g.encode("ascii")} for g in gt_rles]
        # iou matrix of shape (n_cands, n_gt)
        iou_mat = np.asarray(mu.iou(cand_rle_objs, gt_rle_objs, [0] * len(gt_rle_objs)), dtype=np.float64)
        best_iou = iou_mat.max(axis=1)
        y_sum += best_iou * (best_iou > 0.5)

    return (y_sum / float(K)).astype(np.float32)


def extract_features_and_pool(
    oof_dir: str,
    folds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    train_json_path: str,
    min_cluster_score: float = 0.15,
    tau_mask: float = 0.50,
    min_area: int = 10,
    num_views_total: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """
    Loads OOF detections across all 5 folds, filters clusters with S >= min_cluster_score,
    extracts geometry + probability features, and computes target y.
    """
    k_by_obs, gt_records_by_obs = build_gt_lookup(gt_df, train_json_path)

    rows = []
    pool_by_obs: Dict[str, List[dict]] = {}
    t0 = time.time()

    for f in range(5):
        fold_dir = os.path.join(oof_dir, f"fold{f}")
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()

        for obs in val_obs:
            pt_path = os.path.join(fold_dir, f"{obs}.pt")
            if not os.path.isfile(pt_path):
                pool_by_obs[obs] = []
                continue

            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            fused = data.get("fused", [])
            raw_cands = data.get("candidates", [])

            # Recover member_scores if not already stored in fused dicts
            if fused and "member_scores" not in fused[0]:
                member_scores_list = extract_member_scores_from_raw(raw_cands, iou_thresh=0.40)
            else:
                member_scores_list = [c.get("member_scores", [c["score"]]) for c in fused]

            # Image-level summary stats prior to filtering
            all_scores = [float(c["score"]) for c in fused]
            max_score_in_img = max(all_scores) if all_scores else 1e-6
            n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

            obs_candidates = []
            cand_rles_for_target = []

            for rank_0, (c, m_scores) in enumerate(zip(fused, member_scores_list)):
                sc = float(c["score"])
                if sc < min_cluster_score:
                    continue

                prob_crop = c["prob_crop"]
                box = c["box"]
                binary_crop = (prob_crop >= tau_mask)
                area = int(binary_crop.sum())
                if area < min_area:
                    continue

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                bbox_area = float(bw * bh)

                # Probability statistics inside the binary mask
                masked_probs = prob_crop[binary_crop].astype(np.float32)
                prob_mean = float(masked_probs.mean())
                prob_std = float(masked_probs.std()) if area > 1 else 0.0
                core_pixels = int((masked_probs >= 0.75).sum())
                prob_sharpness_core = float(core_pixels) / float(area)
                q90, q10 = np.percentile(masked_probs, [90, 10])
                prob_sharpness_iqr = float(q90 - q10)

                # Member view statistics
                m_arr = np.asarray(m_scores, dtype=np.float32)
                member_mean = float(m_arr.mean()) if len(m_arr) > 0 else sc
                member_std = float(m_arr.std()) if len(m_arr) > 1 else 0.0
                norm_views = float(c["cluster_size"]) * (4.0 / float(num_views_total))

                # Geometry statistics
                log_area = float(np.log10(area + 1.0))
                bbox_fill = float(area) / bbox_area
                aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))

                rank_in_image = rank_0 + 1
                rel_score_to_max = sc / max(1e-6, max_score_in_img)

                rle_str = encode_crop_to_rle(box, binary_crop)
                cand_rles_for_target.append(rle_str)

                feat_dict = {
                    "obs": obs,
                    "fold": f,
                    "cand_idx": len(obs_candidates),
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
                }
                rows.append(feat_dict)

                obs_candidates.append({
                    "box": box,
                    "binary_crop": binary_crop,
                    "orig_area": area,
                    "score": sc,
                    "pred_y": 0.0,
                })

            # Compute exact target y against the K annotators of obs
            y_targets = compute_candidate_targets(cand_rles_for_target, obs, k_by_obs, gt_records_by_obs)
            start_row_idx = len(rows) - len(obs_candidates)
            for k_i, y_val in enumerate(y_targets):
                rows[start_row_idx + k_i]["target_y"] = float(y_val)

            pool_by_obs[obs] = obs_candidates

        print(f"  Extracted Fold {f}: {len(val_obs)} images ({time.time() - t0:.1f}s)")

    df_features = pd.DataFrame(rows)
    return df_features, pool_by_obs


def generate_disjoint_predictions_from_scorer(
    pool_by_obs: Dict[str, List[dict]],
    stems: List[str],
    tau_y: float,
    min_area: int = 10,
    area_retention: float = 0.50,
    include_empty: bool = True,
) -> pd.DataFrame:
    """
    Selects candidates with pred_y > tau_y, sorts descending by pred_y,
    and resolves overlaps via canvas painting (>= 50% area retained).
    """
    rows = []
    for stem in stems:
        cands = pool_by_obs.get(stem, [])
        kept = [c for c in cands if c["pred_y"] > tau_y]
        kept.sort(key=lambda c: c["pred_y"], reverse=True)

        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        kept_rles = []

        for c in kept:
            b = c["box"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            binary_crop = c["binary_crop"]
            orig_area = c["orig_area"]

            sub_occ = occupied[y1:y2, x1:x2]
            clean_crop = binary_crop & (~sub_occ)
            clean_area = int(clean_crop.sum())

            if clean_area < min_area or (clean_area / float(orig_area)) < area_retention:
                continue

            occupied[y1:y2, x1:x2] |= clean_crop
            kept_rles.append(encode_crop_to_rle(b, clean_crop))

        if not kept_rles:
            if include_empty:
                rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
        else:
            for k, enc in enumerate(kept_rles, start=1):
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def precompute_candidate_matches_per_obs(
    pool_by_obs: Dict[str, List[dict]],
    gt_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    min_area: int = 10,
    area_retention: float = 0.50,
) -> Dict[str, dict]:
    """
    For fast threshold sweeping on training folds:
    Since canvas painting order depends only on the ordering of pred_y,
    Wait: in canvas painting, if candidate A is above tau_y, it paints on the canvas;
    if candidate A is only dropped when pred_y <= tau_y, any candidate B with pred_y > tau_y
    has pred_y(B) > tau_y >= pred_y(A), so candidate B is ALWAYS painted BEFORE candidate A!
    Think about that mathematical property:
    Because candidates are painted in strictly descending order of pred_y,
    whether a lower-ranked candidate (pred_y <= tau_y) is kept or dropped NEVER affects
    the canvas painting of higher-ranked candidates (pred_y > tau_y)!
    Therefore, the canvas painting of ALL candidates sorted by pred_y descending is
    100% INVARIANT to the threshold tau_y!
    """
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_by_obs: Dict[str, List[List[str]]] = defaultdict(list)
    for ai, group in gt_df.groupby(gt_ai, sort=False):
        obs = ai.split("-", 1)[1]
        gt_by_obs[obs].append(group["segmentation_rle"].tolist())

    obs_eval_cache: Dict[str, dict] = {}
    all_obs = folds_df["observation"].tolist()

    for obs in all_obs:
        cands = pool_by_obs.get(obs, [])
        sorted_cands = sorted(cands, key=lambda c: c["pred_y"], reverse=True)

        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        painted_rles = []
        painted_scores = []

        for c in sorted_cands:
            b = c["box"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            binary_crop = c["binary_crop"]
            orig_area = c["orig_area"]

            sub_occ = occupied[y1:y2, x1:x2]
            clean_crop = binary_crop & (~sub_occ)
            clean_area = int(clean_crop.sum())

            if clean_area < min_area or (clean_area / float(orig_area)) < area_retention:
                continue

            occupied[y1:y2, x1:x2] |= clean_crop
            painted_rles.append(encode_crop_to_rle(b, clean_crop))
            painted_scores.append(float(c["pred_y"]))

        # Precompute IoU matrices against each annotator record of obs
        annotator_gt_lists = gt_by_obs.get(obs, [])
        iou_matrices = []
        for gt_rles in annotator_gt_lists:
            iou_mat = pq_eval.iou_matrix(gt_rles, painted_rles)  # shape (n_gt, n_pred)
            iou_matrices.append(iou_mat)

        obs_eval_cache[obs] = {
            "scores": np.asarray(painted_scores, dtype=np.float64),
            "iou_matrices": iou_matrices,
            "rles": painted_rles,
        }

    return obs_eval_cache


def fast_score_from_cache(
    obs_eval_cache: Dict[str, dict],
    obs_list: List[str],
    tau_y: float,
) -> dict:
    """Evaluates exact pq_eval metrics for a subset of observations at threshold tau_y in milliseconds."""
    matrices = []
    for obs in obs_list:
        entry = obs_eval_cache[obs]
        scores = entry["scores"]
        k_keep = int(np.sum(scores > tau_y))
        for m in entry["iou_matrices"]:
            matrices.append(m[:, :k_keep])
    return pq_eval.pq_from_matrices(matrices)


def run_paired_bootstrap_707(
    gt_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    pred_a_df: pd.DataFrame,
    pred_b_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict:
    """
    Runs paired bootstrap over all 707 observations (missing/empty images count as all-FN).
    """
    all_obs = sorted(folds_df["observation"].unique().tolist())
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]

    # Filter out EMPTY_RLE so missing/empty images have n_pred = 0 (all-FN)
    pa = pred_a_df[pred_a_df["segmentation_rle"] != EMPTY_RLE].copy()
    pb = pred_b_df[pred_b_df["segmentation_rle"] != EMPTY_RLE].copy()

    pa_img = pa["filament_id"].str.split("_", n=1).str[0]
    pb_img = pb["filament_id"].str.split("_", n=1).str[0]
    pa_by_img = pa.groupby(pa_img)["segmentation_rle"].apply(list).to_dict()
    pb_by_img = pb.groupby(pb_img)["segmentation_rle"].apply(list).to_dict()

    stats_a_by_obs = defaultdict(lambda: [0.0, 0, 0, 0])  # [iou_sum, tp, fp, fn]
    stats_b_by_obs = defaultdict(lambda: [0.0, 0, 0, 0])

    for ai, rows in gt_df.groupby(gt_ai, sort=False):
        img = ai.split("-", 1)[1]
        gt_rles = rows["segmentation_rle"].tolist()

        for stats_dict, p_dict in ((stats_a_by_obs, pa_by_img), (stats_b_by_obs, pb_by_img)):
            m = pq_eval.iou_matrix(gt_rles, p_dict.get(img, []))
            n_gt, n_pred = m.shape
            if n_gt == 0:
                stats_dict[img][2] += n_pred
            elif n_pred == 0:
                stats_dict[img][3] += n_gt
            else:
                hit = m > pq_eval.IOU_THRESHOLD
                stats_dict[img][0] += float(m[hit].sum())
                stats_dict[img][1] += int(hit.sum())
                stats_dict[img][2] += int((hit.sum(axis=0) == 0).sum())
                stats_dict[img][3] += int((hit.sum(axis=1) == 0).sum())

    arr_a = np.array([stats_a_by_obs[o] for o in all_obs], dtype=np.float64)
    arr_b = np.array([stats_b_by_obs[o] for o in all_obs], dtype=np.float64)

    def calc_pq(arr_sum):
        iou_s, tp, fp, fn = arr_sum
        denom = tp + 0.5 * fp + 0.5 * fn
        return float(iou_s / denom) if denom > 0 else 0.0

    point_a = calc_pq(arr_a.sum(axis=0))
    point_b = calc_pq(arr_b.sum(axis=0))
    point_delta = point_b - point_a

    rng = np.random.default_rng(seed)
    n_obs = len(all_obs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b_i in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        pq_a_i = calc_pq(arr_a[idx].sum(axis=0))
        pq_b_i = calc_pq(arr_b[idx].sum(axis=0))
        deltas[b_i] = pq_b_i - pq_a_i

    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])
    p_b_better = float(np.mean(deltas > 0))

    return {
        "pq_a": point_a,
        "pq_b": point_b,
        "point_delta": point_delta,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_b_better": p_b_better,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Fitted LightGBM Instance Scorer on OOF Detections")
    parser.add_argument("--oof_dir", type=str, default="work/oof", help="OOF detections directory")
    parser.add_argument("--min_score", type=float, default=0.15, help="Low cluster score threshold for candidate pool")
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of LightGBM boosting trees")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="LightGBM learning rate")
    args = parser.parse_args()

    print("==================================================================")
    print("   CROSS-FITTED LIGHTGBM INSTANCE SCORER (DECISION LAYER)         ")
    print("==================================================================")
    paths = find_dataset_paths()
    gt_df = pd.read_csv(paths["gt_all_csv"])
    folds_df = pd.read_csv(paths["folds_csv"])

    print(f"Step 1: Extracting candidate pool (S >= {args.min_score:.2f}) and features across 707 images...")
    df_feat, pool_by_obs = extract_features_and_pool(
        oof_dir=args.oof_dir,
        folds_df=folds_df,
        gt_df=gt_df,
        train_json_path=paths["train_json"],
        min_cluster_score=args.min_score,
        tau_mask=0.50,
        min_area=10,
        num_views_total=4,
    )
    print(f"Total candidates in pool (S >= {args.min_score:.2f}): {len(df_feat)} across {len(pool_by_obs)} images")
    print(f"Positive target rate (y > 0): {(df_feat['target_y'] > 0).mean()*100:.2f}% | Mean y: {df_feat['target_y'].mean():.4f}")

    # Step 2: 5-Fold Cross-Fitting of LightGBM Regressor
    if lgb is None:
        raise ImportError("lightgbm is required to run scripts/instance_scorer.py (pip install lightgbm)")
    print("\nStep 2: 5-Fold Cross-Fitting LightGBM Regressor (max_depth=3, min_child_samples=50, 300 trees)...")
    oof_preds = np.zeros(len(df_feat), dtype=np.float64)
    importance_split = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    importance_gain = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    models_by_fold = {}
    for held_out_f in range(5):
        train_mask = (df_feat["fold"] != held_out_f).values
        val_mask = (df_feat["fold"] == held_out_f).values

        X_tr = df_feat.loc[train_mask, FEATURE_NAMES].values
        y_tr = df_feat.loc[train_mask, "target_y"].values
        X_va = df_feat.loc[val_mask, FEATURE_NAMES].values

        model = lgb.LGBMRegressor(
            objective="regression",
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=2026 + held_out_f,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        preds_va = model.predict(X_va)
        oof_preds[val_mask] = preds_va
        models_by_fold[held_out_f] = model

        importance_split += model.booster_.feature_importance(importance_type="split") / 5.0
        importance_gain += model.booster_.feature_importance(importance_type="gain") / 5.0

    df_feat["pred_y"] = oof_preds

    # Populate pred_y back into pool_by_obs
    for row in df_feat[["obs", "cand_idx", "pred_y"]].itertuples(index=False):
        pool_by_obs[row.obs][int(row.cand_idx)]["pred_y"] = float(row.pred_y)

    # Step 3: Precompute canvas-painted RLEs and IoU matrices per observation
    print("\nStep 3: Precomputing order-invariant disjoint canvas masks & IoU matrices across 707 images...")
    t_cache = time.time()
    obs_eval_cache = precompute_candidate_matches_per_obs(pool_by_obs, gt_df, folds_df)
    print(f"Precomputed 707 observations in {time.time() - t_cache:.1f}s.")

    # Step 4: Select threshold tau on the 4 training folds ONLY (grid around PQ/2 +- 0.04)
    # Current OOF PQ is ~0.434 -> PQ/2 = 0.217 -> grid [0.175, 0.260] with step 0.0025
    tau_grid = np.round(np.arange(0.175, 0.2601, 0.0025), 4).tolist()

    obs_by_fold = {
        f: folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        for f in range(5)
    }

    fold_results = []
    held_out_dfs = []

    print("\nStep 4: Selecting threshold tau on 4 training folds (grid PQ/2 +- 0.04 = [0.175, 0.260])...")
    for held_out_f in range(5):
        train_obs = [o for f in range(5) if f != held_out_f for o in obs_by_fold[f]]
        val_obs = obs_by_fold[held_out_f]

        best_tr_pq = -1.0
        best_tau = 0.2175
        for tau in tau_grid:
            res_tr = fast_score_from_cache(obs_eval_cache, train_obs, tau)
            if res_tr["pq"] > best_tr_pq:
                best_tr_pq = res_tr["pq"]
                best_tau = tau

        res_val = fast_score_from_cache(obs_eval_cache, val_obs, best_tau)
        fold_results.append({
            "fold": held_out_f,
            "selected_tau": best_tau,
            "train_4fold_pq": best_tr_pq,
            "val_pq": res_val["pq"],
            "val_sq": res_val["sq"],
            "val_rq": res_val["rq"],
            "tp": res_val["tp"],
            "fp": res_val["fp"],
            "fn": res_val["fn"],
            "pred_multi_hit": res_val["pred_multi_hit"],
        })

        df_fold_pred = generate_disjoint_predictions_from_scorer(
            pool_by_obs=pool_by_obs,
            stems=val_obs,
            tau_y=best_tau,
            min_area=10,
            area_retention=0.50,
            include_empty=True,
        )
        held_out_dfs.append(df_fold_pred)

    # Concatenate cross-fitted predictions across all 707 images
    scorer_oof_df = pd.concat(held_out_dfs, ignore_index=True)
    scorer_csv_path = os.path.join(args.oof_dir, "oof_predictions_instance_scorer.csv")
    scorer_oof_df.to_csv(scorer_csv_path, index=False)

    # Exact verification via pq_eval.score over all 707 images (missing/empty = all-FN)
    pooled_res = pq_eval.score(gt_df, scorer_oof_df[scorer_oof_df["segmentation_rle"] != EMPTY_RLE])

    # Also evaluate global best tau across all 707 images for reference
    all_obs_list = folds_df["observation"].tolist()
    best_global_tau = max(tau_grid, key=lambda t: fast_score_from_cache(obs_eval_cache, all_obs_list, t)["pq"])
    global_tau_res = fast_score_from_cache(obs_eval_cache, all_obs_list, best_global_tau)

    # Step 5: Paired Bootstrap vs Current Threshold Rule (Nested-CV and Default) over all 707 images
    nested_thresh_csv = os.path.join(args.oof_dir, "oof_predictions_nested_cv.csv")
    default_thresh_csv = os.path.join(args.oof_dir, "oof_predictions.csv")

    boot_vs_nested = run_paired_bootstrap_707(
        gt_df=gt_df,
        folds_df=folds_df,
        pred_a_df=pd.read_csv(nested_thresh_csv),
        pred_b_df=scorer_oof_df,
        n_boot=1000,
        seed=2026,
    )
    boot_vs_default = run_paired_bootstrap_707(
        gt_df=gt_df,
        folds_df=folds_df,
        pred_a_df=pd.read_csv(default_thresh_csv),
        pred_b_df=scorer_oof_df,
        n_boot=1000,
        seed=2026,
    )

    # Load per-fold baseline nested-CV PQ for side-by-side comparison
    with open(os.path.join(args.oof_dir, "oof_tuning_summary.json"), "r") as f:
        base_tuning = json.load(f)
    base_fold_pq = {d["held_out_fold"]: d["held_out_pq"] for d in base_tuning["nested_cv_folds"]}

    # Feature importance table
    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "gain": importance_gain,
        "gain_pct": 100.0 * importance_gain / max(1e-9, importance_gain.sum()),
        "split": importance_split,
    }).sort_values("gain", ascending=False)

    print("\n==================================================================")
    print("   CROSS-FITTED INSTANCE SCORER OOF RESULTS (ALL 707 IMAGES)      ")
    print("==================================================================")
    print(f"Baseline Default Threshold (0.85/0.50) PQ:   {boot_vs_default['pq_a']:.4f}")
    print(f"Baseline Nested-CV Threshold (0.825/0.50) PQ:{boot_vs_nested['pq_a']:.4f}")
    print(f"Instance Scorer Cross-Fitted OOF PQ:         {pooled_res['pq']:.4f} (SQ: {pooled_res['sq']:.4f}, RQ: {pooled_res['rq']:.4f})")
    print(f"  TP: {pooled_res['tp']} | FP: {pooled_res['fp']} | FN: {pooled_res['fn']} | gt_multi: {pooled_res['gt_multi_hit']} | pred_multi: {pooled_res['pred_multi_hit']}")
    print(f"Instance Scorer Global Fixed tau ({best_global_tau:.4f}) PQ: {global_tau_res['pq']:.4f}")
    print("------------------------------------------------------------------")
    print("Per-Fold PQ Comparison (Cross-Fitted Instance Scorer vs Nested-CV Threshold):")
    for fr in fold_results:
        f_idx = fr["fold"]
        b_pq = base_fold_pq.get(f_idx, 0.0)
        diff = fr["val_pq"] - b_pq
        print(
            f"  Fold {f_idx}: tau={fr['selected_tau']:.4f} | Scorer PQ={fr['val_pq']:.4f} "
            f"(SQ={fr['val_sq']:.4f}, RQ={fr['val_rq']:.4f}) vs Thresh PQ={b_pq:.4f} (Delta={diff:+.4f})"
        )
    print("------------------------------------------------------------------")
    print("Paired Bootstrap (1000 iter, seed=2026, all 707 images):")
    print(
        f"  vs Nested-CV Threshold ({boot_vs_nested['pq_a']:.4f} -> {boot_vs_nested['pq_b']:.4f}): "
        f"Delta = {boot_vs_nested['point_delta']:+.4f}, 95% CI = [{boot_vs_nested['ci_lower']:+.4f}, {boot_vs_nested['ci_upper']:+.4f}], "
        f"P(Scorer > Thresh) = {boot_vs_nested['p_b_better']*100:.1f}%"
    )
    print(
        f"  vs Default Threshold   ({boot_vs_default['pq_a']:.4f} -> {boot_vs_default['pq_b']:.4f}): "
        f"Delta = {boot_vs_default['point_delta']:+.4f}, 95% CI = [{boot_vs_default['ci_lower']:+.4f}, {boot_vs_default['ci_upper']:+.4f}], "
        f"P(Scorer > Default) = {boot_vs_default['p_b_better']*100:.1f}%"
    )
    print("------------------------------------------------------------------")
    print("Feature Importances (sorted by Gain %):")
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("==================================================================")

    summary_out = {
        "pooled_oof_metrics": pooled_res,
        "global_best_tau": best_global_tau,
        "global_best_metrics": global_tau_res,
        "per_fold": fold_results,
        "bootstrap_vs_nested_cv_threshold": boot_vs_nested,
        "bootstrap_vs_default_threshold": boot_vs_default,
        "feature_importances": imp_df.to_dict(orient="records"),
    }
    summary_json_path = os.path.join(args.oof_dir, "instance_scorer_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved summary to {summary_json_path}")


if __name__ == "__main__":
    main()
