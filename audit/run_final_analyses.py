#!/usr/bin/env python3
"""
audit/run_final_analyses.py - Release Verification & Statistical Analyses (CPU-Only).

Computes all supplementary quantitative analyses required for final release documentation:
1. Per-image PQ, Semantic Union Dice/IoU, and Instance Matched-TP IoU/Dice distributions
   (histograms and quantiles) across all 707 OOF observations for Rule A and Rule B.
2. One-to-many and many-to-one matching analysis across 1,154 (annotator, image) records
   and 296 multi-annotator images at IoU > 0.50 and IoU > 0.0 (pixel overlap > 0).
3. Compact 6-model Fold-0 OOF evaluation on the 142 Fold-0 observations for Rule A and Rule B,
   plus live CPU FP32-compute inference comparison (FP16 repo weights/ vs FP32 originals)
   on sample_images/.
4. Generates 4 clean figures in figures/:
   - figures/oof_pq_dice_iou_distributions.png
   - figures/morphology_gt_vs_pred_distributions.png
   - figures/morphology_stratified_pq.png
   - figures/matching_and_error_decomposition.png

Outputs:
  - audit/final_release_analyses.json
  - figures/*.png
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, extract_state_dict, get_maskrcnn_model, preprocess_solar_image
from scripts.instance_scorer import encode_crop_to_rle
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    build_gt_rles_by_obs,
    weighted_mask_fusion_tagged,
)
from scripts.predict_ensemble import (
    apply_flip_transform,
    unflip_boxes,
    unflip_mask_probabilities,
)

FAMILY_SUBDIRS = {
    "clahe": ("oof", 4, 1536),
    "anchors5": ("oof_a5", 4, 1536),
    "dice": ("oof_dice", 4, 1536),
    "2048": ("oof_2048", 4, 2048),
    "a5dice": ("oof_a5dice", 4, 1536),
    "1792_clahe": ("oof_1792", 4, 1792),
}

QUANTILE_PROBS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
QUANTILE_NAMES = ["min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max"]


def summarize_array(vals: List[float]) -> Dict[str, float]:
    """Compute mean, std, and summary quantiles for a list of scalar values."""
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    qs = np.quantile(arr, QUANTILE_PROBS)
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "zero_count": int((arr == 0.0).sum()),
    }
    for name, qv in zip(QUANTILE_NAMES, qs):
        out[name] = float(qv)
    return out


def load_oof_raw_candidates(
    work_dir: Path,
    folds_df: pd.DataFrame,
    fold_filter: Optional[int] = None,
) -> Tuple[Dict[str, List[dict]], int]:
    """Load cached 4-flip OOF candidate proposals across all 6 families."""
    obs_to_cands: Dict[str, List[dict]] = {}
    obs_to_fold = dict(zip(folds_df["observation"], folds_df["fold"]))
    for obs in obs_to_fold:
        obs_to_cands[obs] = []

    loaded_files = 0
    for fam, (sub, _, _) in FAMILY_SUBDIRS.items():
        base_dir = work_dir / sub
        if not base_dir.is_dir():
            raise FileNotFoundError(
                f"Missing OOF cache directory for family '{fam}': {base_dir}. Pass --work_dir to full 6-family OOF cache."
            )
        for fold in range(5):
            if fold_filter is not None and fold != fold_filter:
                continue
            fold_dir = base_dir / f"fold{fold}"
            if not fold_dir.is_dir():
                continue
            for obs, f in obs_to_fold.items():
                if f != fold:
                    continue
                pt_path = fold_dir / f"{obs}.pt"
                if not pt_path.is_file():
                    continue
                loaded_files += 1
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                for c in d.get("candidates", []):
                    obs_to_cands[obs].append({
                        "box": c["box"],
                        "score": float(c["score"]),
                        "prob_crop": (
                            c["prob_crop"].astype(np.float16)
                            if isinstance(c["prob_crop"], np.ndarray)
                            else c["prob_crop"]
                        ),
                        "family": fam,
                    })
    return obs_to_cands, loaded_files


def fuse_with_rule(raw_cands: List[dict], family_weights: Dict[str, float], gamma: float) -> List[dict]:
    """Run Weighted Mask Fusion with family weighting and score exponentiation."""
    effective_family_views = {
        fam: n_views * family_weights.get(fam, 1.0)
        for fam, (_, n_views, _) in FAMILY_SUBDIRS.items()
    }
    weighted_cands = [
        {
            "box": c["box"],
            "score": (float(c["score"]) ** gamma) * family_weights.get(c["family"], 1.0),
            "prob_crop": c["prob_crop"],
            "family": c["family"],
        }
        for c in raw_cands
    ]
    eff_total_views = sum(effective_family_views.values())
    return weighted_mask_fusion_tagged(
        weighted_cands,
        num_views=eff_total_views,
        iou_thresh=0.40,
        family_views=effective_family_views,
    )


def paint_disjoint_rles(fused_clusters: List[dict], tau_score: float, tau_mask: float) -> List[str]:
    """Apply greedy score-descending disjoint canvas painting and RLE encoding."""
    occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    rles: List[str] = []
    for c in fused_clusters:
        if float(c["score"]) < tau_score:
            break
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bc = c["prob_crop"] >= tau_mask
        oa = int(bc.sum())
        if oa < 10:
            continue
        clean_crop = bc & (~occupied[y1:y2, x1:x2])
        ca = int(clean_crop.sum())
        if ca < 10 or (ca / float(oa)) < 0.50:
            continue
        occupied[y1:y2, x1:x2] |= clean_crop
        rles.append(encode_crop_to_rle(box, clean_crop))
    return rles


def analyze_rule_oof(
    rule_name: str,
    clusters_by_obs: Dict[str, List[dict]],
    obs_list: List[str],
    gt_by_obs: Dict[str, List[List[str]]],
    tau_score: float,
    tau_mask: float,
) -> dict:
    """Compute pooled PQ, per-image distributions, and matching diagnostics for a rule."""
    per_image_pq: List[float] = []
    per_image_union_dice: List[float] = []
    per_image_union_iou: List[float] = []
    per_image_mean_tp_iou: List[float] = []
    per_image_mean_tp_dice: List[float] = []
    tp_instance_ious: List[float] = []
    tp_instance_dices: List[float] = []

    all_gt_counts: List[int] = []
    all_pred_counts: List[int] = []

    # Matching & overlap counters (1,154 annotator-image pairs)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_ann_records = 0
    total_gt_instances_records = 0
    total_pred_instance_evals = 0  # sum of n_pred per (ann, img) record

    # Within-annotator IoU > 0.50
    gt_hits_05_hist: Dict[int, int] = defaultdict(int)
    pred_hits_05_hist: Dict[int, int] = defaultdict(int)
    # Within-annotator IoU > 0.0 (any pixel overlap)
    gt_hits_any_hist: Dict[int, int] = defaultdict(int)
    pred_hits_any_hist: Dict[int, int] = defaultdict(int)

    # Cross-annotator analysis on multi-annotator observations (>= 2 annotators)
    multi_ann_obs_count = 0
    total_unique_preds_on_multi = 0
    pred_cross_ann_05_hist: Dict[int, int] = defaultdict(int)
    pred_unique_records_matched_05: Dict[int, int] = defaultdict(int)
    pred_same_ann_multi_05 = 0
    pred_cross_ann_only_multi_05 = 0

    for obs in obs_list:
        pred_rles = paint_disjoint_rles(clusters_by_obs.get(obs, []), tau_score, tau_mask)
        ann_gt_lists = gt_by_obs[obs]
        n_pred = len(pred_rles)
        all_pred_counts.append(n_pred)

        # Build predicted union RLE once per observation
        if n_pred > 0:
            pred_rle_objs = [{"size": [FRAME_SIZE, FRAME_SIZE], "counts": r.encode("ascii")} for r in pred_rles]
            pred_union_rle = mu.merge(pred_rle_objs, intersect=False)
            pred_union_area = int(mu.area(pred_union_rle))
        else:
            pred_rle_objs = []
            pred_union_rle = None
            pred_union_area = 0

        obs_tp = 0
        obs_fp = 0
        obs_fn = 0
        obs_iou_sum = 0.0
        obs_union_dices: List[float] = []
        obs_union_ious: List[float] = []
        obs_tp_ious_this_img: List[float] = []

        # Collect per-prediction hit matrix across all annotator records for this image
        pred_hits_per_record = np.zeros((n_pred, len(ann_gt_lists)), dtype=np.int32)

        for r_idx, gt_list in enumerate(ann_gt_lists):
            total_ann_records += 1
            n_gt = len(gt_list)
            total_gt_instances_records += n_gt
            total_pred_instance_evals += n_pred
            all_gt_counts.append(n_gt)

            m = pq_eval.iou_matrix(gt_list, pred_rles)  # shape (n_gt, n_pred)
            hit05 = m > 0.5
            hit_any = m > 0.0

            rec_tp_ious = m[hit05].tolist()
            tp_r = len(rec_tp_ious)
            fp_r = int((hit05.sum(axis=0) == 0).sum()) if n_pred > 0 else 0
            fn_r = int((hit05.sum(axis=1) == 0).sum()) if n_gt > 0 else 0

            obs_tp += tp_r
            obs_fp += fp_r
            obs_fn += fn_r
            obs_iou_sum += float(sum(rec_tp_ious))

            for v in rec_tp_ious:
                fv = float(v)
                tp_instance_ious.append(fv)
                tp_instance_dices.append((2.0 * fv) / (1.0 + fv))
                obs_tp_ious_this_img.append(fv)

            if n_gt > 0:
                for c_cnt in hit05.sum(axis=1):
                    gt_hits_05_hist[int(c_cnt)] += 1
                for c_cnt in hit_any.sum(axis=1):
                    gt_hits_any_hist[int(c_cnt)] += 1
            if n_pred > 0:
                p_hits_05 = hit05.sum(axis=0)
                pred_hits_per_record[:, r_idx] = p_hits_05
                for c_cnt in p_hits_05:
                    pred_hits_05_hist[int(c_cnt)] += 1
                for c_cnt in hit_any.sum(axis=0):
                    pred_hits_any_hist[int(c_cnt)] += 1

            # Semantic union Dice & IoU for this annotator record
            if n_gt > 0:
                gt_rle_objs = [{"size": [FRAME_SIZE, FRAME_SIZE], "counts": g.encode("ascii")} for g in gt_list]
                gt_union_rle = mu.merge(gt_rle_objs, intersect=False)
                gt_union_area = int(mu.area(gt_union_rle))
            else:
                gt_union_rle = None
                gt_union_area = 0

            if pred_union_area == 0 and gt_union_area == 0:
                u_iou, u_dice = 1.0, 1.0
            elif pred_union_area == 0 or gt_union_area == 0:
                u_iou, u_dice = 0.0, 0.0
            else:
                inter_rle = mu.merge([pred_union_rle, gt_union_rle], intersect=True)
                inter_area = int(mu.area(inter_rle))
                union_area = pred_union_area + gt_union_area - inter_area
                u_iou = inter_area / float(union_area) if union_area > 0 else 0.0
                u_dice = (2.0 * inter_area) / float(pred_union_area + gt_union_area)
            obs_union_ious.append(float(u_iou))
            obs_union_dices.append(float(u_dice))

        if len(ann_gt_lists) >= 2 and n_pred > 0:
            multi_ann_obs_count += 1
            total_unique_preds_on_multi += n_pred
            for p_idx in range(n_pred):
                tot_hits = int(pred_hits_per_record[p_idx].sum())
                rec_hits = int((pred_hits_per_record[p_idx] > 0).sum())
                max_in_single_rec = int(pred_hits_per_record[p_idx].max())
                pred_cross_ann_05_hist[tot_hits] += 1
                pred_unique_records_matched_05[rec_hits] += 1
                if max_in_single_rec >= 2:
                    pred_same_ann_multi_05 += 1
                elif tot_hits >= 2 and max_in_single_rec <= 1:
                    pred_cross_ann_only_multi_05 += 1

        total_tp += obs_tp
        total_fp += obs_fp
        total_fn += obs_fn
        total_iou_sum += obs_iou_sum

        denom_i = obs_tp + 0.5 * obs_fp + 0.5 * obs_fn
        pq_i = obs_iou_sum / denom_i if denom_i > 0 else 0.0
        per_image_pq.append(float(pq_i))
        per_image_union_iou.append(float(np.mean(obs_union_ious)))
        per_image_union_dice.append(float(np.mean(obs_union_dices)))
        if obs_tp_ious_this_img:
            per_image_mean_tp_iou.append(float(np.mean(obs_tp_ious_this_img)))
            per_image_mean_tp_dice.append(
                float(np.mean([(2.0 * v) / (1.0 + v) for v in obs_tp_ious_this_img]))
            )
        else:
            per_image_mean_tp_iou.append(0.0)
            per_image_mean_tp_dice.append(0.0)

    denom = total_tp + 0.5 * total_fp + 0.5 * total_fn
    pooled_pq = total_iou_sum / denom if denom > 0 else 0.0
    pooled_sq = total_iou_sum / total_tp if total_tp > 0 else 0.0
    pooled_rq = total_tp / denom if denom > 0 else 0.0

    gt_multi_05 = sum(v for k, v in gt_hits_05_hist.items() if k >= 2)
    pred_multi_05 = sum(v for k, v in pred_hits_05_hist.items() if k >= 2)
    gt_multi_any = sum(v for k, v in gt_hits_any_hist.items() if k >= 2)
    pred_multi_any = sum(v for k, v in pred_hits_any_hist.items() if k >= 2)

    return {
        "rule": rule_name,
        "tau_score": tau_score,
        "tau_mask": tau_mask,
        "n_observations": len(obs_list),
        "n_annotator_records": total_ann_records,
        "pooled_metrics": {
            "pq": float(pooled_pq),
            "sq": float(pooled_sq),
            "rq": float(pooled_rq),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "pred_instances_unique": int(sum(all_pred_counts)),
            "zero_pred_images": int(sum(1 for x in all_pred_counts if x == 0)),
        },
        "distributions": {
            "per_image_pq": summarize_array(per_image_pq),
            "per_image_semantic_union_dice": summarize_array(per_image_union_dice),
            "per_image_semantic_union_iou": summarize_array(per_image_union_iou),
            "per_image_mean_tp_iou_incl_zero": summarize_array(per_image_mean_tp_iou),
            "per_image_mean_tp_dice_incl_zero": summarize_array(per_image_mean_tp_dice),
            "matched_tp_instance_iou": summarize_array(tp_instance_ious),
            "matched_tp_instance_dice": summarize_array(tp_instance_dices),
        },
        "raw_arrays": {
            "per_image_pq": per_image_pq,
            "per_image_union_dice": per_image_union_dice,
            "per_image_union_iou": per_image_union_iou,
        },
        "matching_analysis": {
            "within_annotator_iou_gt_05": {
                "total_gt_instance_evals": int(total_gt_instances_records),
                "total_pred_instance_evals": int(total_pred_instance_evals),
                "gt_multi_hit_count": int(gt_multi_05),
                "gt_multi_hit_pct": float(100.0 * gt_multi_05 / max(1, total_gt_instances_records)),
                "pred_multi_hit_count": int(pred_multi_05),
                "pred_multi_hit_pct": float(100.0 * pred_multi_05 / max(1, total_pred_instance_evals)),
                "gt_hits_distribution": {str(k): int(v) for k, v in sorted(gt_hits_05_hist.items())},
                "pred_hits_distribution": {str(k): int(v) for k, v in sorted(pred_hits_05_hist.items())},
            },
            "within_annotator_any_overlap_iou_gt_0": {
                "gt_multi_overlap_count": int(gt_multi_any),
                "gt_multi_overlap_pct": float(100.0 * gt_multi_any / max(1, total_gt_instances_records)),
                "pred_multi_overlap_count": int(pred_multi_any),
                "pred_multi_overlap_pct": float(100.0 * pred_multi_any / max(1, total_pred_instance_evals)),
                "gt_overlap_distribution": {str(k): int(v) for k, v in sorted(gt_hits_any_hist.items())},
                "pred_overlap_distribution": {str(k): int(v) for k, v in sorted(pred_hits_any_hist.items())},
            },
            "multi_annotator_cross_record_iou_gt_05": {
                "multi_ann_images_with_preds": int(multi_ann_obs_count),
                "total_unique_preds_on_multi_ann_images": int(total_unique_preds_on_multi),
                "pred_matched_across_ge2_total_gts": int(
                    sum(v for k, v in pred_cross_ann_05_hist.items() if k >= 2)
                ),
                "pred_matched_ge2_same_annotator": int(pred_same_ann_multi_05),
                "pred_matched_ge2_cross_annotators_only": int(pred_cross_ann_only_multi_05),
                "pred_total_hits_hist": {str(k): int(v) for k, v in sorted(pred_cross_ann_05_hist.items())},
                "pred_distinct_annotator_records_hit_hist": {
                    str(k): int(v) for k, v in sorted(pred_unique_records_matched_05.items())
                },
            },
        },
    }


def run_sample_fp16_vs_fp32_check(
    repo_weights_dir: Path,
    fp32_dir: Optional[Path],
    sample_dir: Path,
    cfg: dict,
) -> dict:
    """Run live CPU FP32-compute inference on sample_images comparing FP16 weights/ vs FP32 checkpoints."""
    sample_paths = sorted(sample_dir.glob("*.jpeg"))
    f0_models = [m for m in cfg["models"] if int(m["fold"]) == 0]
    views = [("id", False, False), ("h", False, True), ("v", True, False), ("hv", True, True)]

    def _infer_set(ckpt_dir: Path, label: str) -> dict:
        t0_all = time.time()
        cands_by_stem: Dict[str, List[dict]] = {p.stem: [] for p in sample_paths}
        max_weight_diff = 0.0
        for m_info in f0_models:
            fname = m_info["pruned_name"]
            sz = int(m_info["img_size"])
            is_a5 = bool(m_info["anchors5"])
            fam = m_info["family"]
            ckpt_path = ckpt_dir / fname
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = extract_state_dict(ckpt)
            model = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=is_a5)
            model.load_state_dict(sd)
            model.eval()

            for sp in sample_paths:
                img_gray = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
                prep = preprocess_solar_image(img_gray, target_size=sz)
                base_t = torch.from_numpy(prep).float()
                scale = float(FRAME_SIZE) / float(sz)
                with torch.no_grad():
                    for _, fv, fh in views:
                        ft = apply_flip_transform(base_t, flip_v=fv, flip_h=fh)
                        out = model([ft])[0]
                        scs = out["scores"].detach().cpu()
                        keep = scs >= 0.05
                        if not keep.any():
                            continue
                        scs_k = scs[keep].numpy()
                        boxes_k = out["boxes"][keep]
                        masks_k = out["masks"][keep]
                        unf_boxes = (unflip_boxes(boxes_k, fv, fh, sz, sz) * scale).cpu().numpy()
                        unf_masks = unflip_mask_probabilities(masks_k, fv, fh)
                        masks_up = (
                            unf_masks.squeeze(1).cpu().numpy()
                            if sz == FRAME_SIZE
                            else F.interpolate(
                                unf_masks.float(), size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
                            ).squeeze(1).cpu().numpy()
                        )
                        for k in range(len(scs_k)):
                            b = unf_boxes[k]
                            x1, y1 = max(0, int(np.floor(b[0]))), max(0, int(np.floor(b[1])))
                            x2, y2 = min(FRAME_SIZE, int(np.ceil(b[2])) + 1), min(FRAME_SIZE, int(np.ceil(b[3])) + 1)
                            if (x2 - x1) < 2 or (y2 - y1) < 2:
                                continue
                            cands_by_stem[sp.stem].append({
                                "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                                "score": float(scs_k[k]),
                                "prob_crop": masks_up[k, y1:y2, x1:x2].astype(np.float32),
                                "family": fam,
                            })
        elapsed = time.time() - t0_all
        rules_out = {}
        for rname, fw, g, ts, tm in [
            ("rule_A", {f: 1.0 for f in FAMILY_SUBDIRS}, 1.0, 0.85, 0.55),
            ("rule_B", {"clahe": 1.0, "anchors5": 1.0, "dice": 1.0, "a5dice": 1.0, "2048": 2.0, "1792_clahe": 2.0}, 1.5, 0.75, 0.55),
        ]:
            per_img = {}
            total_fils = 0
            for sp in sample_paths:
                fused = fuse_with_rule(cands_by_stem[sp.stem], fw, g)
                rles = paint_disjoint_rles(fused, ts, tm)
                per_img[sp.stem] = rles
                total_fils += len(rles)
            rules_out[rname] = {"total_filaments": total_fils, "rles_by_stem": per_img}
        return {
            "label": label,
            "elapsed_sec": float(elapsed),
            "sec_per_image": float(elapsed / max(1, len(sample_paths))),
            "rules": rules_out,
        }

    fp16_res = _infer_set(repo_weights_dir, "fp16_weights")
    out = {
        "n_sample_images": len(sample_paths),
        "sample_stems": [p.stem for p in sample_paths],
        "fp16_compact_6model_cpu": {
            "elapsed_sec": fp16_res["elapsed_sec"],
            "sec_per_image": fp16_res["sec_per_image"],
            "rule_A_filaments": fp16_res["rules"]["rule_A"]["total_filaments"],
            "rule_B_filaments": fp16_res["rules"]["rule_B"]["total_filaments"],
        },
    }

    if fp32_dir and fp32_dir.is_dir() and all((fp32_dir / m["pruned_name"]).is_file() for m in f0_models):
        fp32_res = _infer_set(fp32_dir, "fp32_originals")
        parity = {}
        for rname in ("rule_A", "rule_B"):
            a_rles = fp16_res["rules"][rname]["rles_by_stem"]
            b_rles = fp32_res["rules"][rname]["rles_by_stem"]
            # Compute semantic union IoU & exact RLE count difference across 3 sample images
            ious = []
            exact_rle_matches = 0
            for stem in out["sample_stems"]:
                ra, rb = a_rles[stem], b_rles[stem]
                if ra == rb:
                    exact_rle_matches += 1
                ma = mu.merge([{"size": [FRAME_SIZE, FRAME_SIZE], "counts": r.encode("ascii")} for r in ra], intersect=False) if ra else None
                mb = mu.merge([{"size": [FRAME_SIZE, FRAME_SIZE], "counts": r.encode("ascii")} for r in rb], intersect=False) if rb else None
                if ma is None and mb is None:
                    ious.append(1.0)
                elif ma is None or mb is None:
                    ious.append(0.0)
                else:
                    inter = int(mu.area(mu.merge([ma, mb], intersect=True)))
                    union = int(mu.area(ma)) + int(mu.area(mb)) - inter
                    ious.append(inter / float(union) if union > 0 else 1.0)
            parity[rname] = {
                "fp16_filaments": fp16_res["rules"][rname]["total_filaments"],
                "fp32_filaments": fp32_res["rules"][rname]["total_filaments"],
                "exact_stem_matches": int(exact_rle_matches),
                "mean_union_iou_fp16_vs_fp32": float(np.mean(ious)),
                "min_union_iou_fp16_vs_fp32": float(np.min(ious)),
            }
        out["fp32_compact_6model_cpu"] = {
            "elapsed_sec": fp32_res["elapsed_sec"],
            "sec_per_image": fp32_res["sec_per_image"],
            "rule_A_filaments": fp32_res["rules"]["rule_A"]["total_filaments"],
            "rule_B_filaments": fp32_res["rules"]["rule_B"]["total_filaments"],
        }
        out["fp16_vs_fp32_sample_parity"] = parity

    return out


def generate_figures(
    res_A: dict,
    res_B: dict,
    morph_data: dict,
    fig_dir: Path,
) -> List[str]:
    """Generate 4 clean engineering figures into figures/."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    generated: List[str] = []

    # 1. Per-image PQ, Semantic Union Dice, and Semantic Union IoU Distributions (N=707)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=160)
    panels = [
        ("Per-Image PQ (N=707)", "per_image_pq", "per_image_pq"),
        ("Per-Image Union Dice (N=707)", "per_image_union_dice", "per_image_semantic_union_dice"),
        ("Per-Image Union IoU (N=707)", "per_image_union_iou", "per_image_semantic_union_iou"),
    ]
    bins = np.linspace(0.0, 1.0, 26)
    for ax, (title, raw_key, stat_key) in zip(axes, panels):
        va = np.asarray(res_A["raw_arrays"][raw_key])
        vb = np.asarray(res_B["raw_arrays"][raw_key])
        sa = res_A["distributions"][stat_key]
        sb = res_B["distributions"][stat_key]
        ax.hist(va, bins=bins, alpha=0.45, color="#1f77b4", edgecolor="white", label=f"Rule A (med={sa['p50']:.3f})")
        ax.hist(vb, bins=bins, alpha=0.45, color="#d62728", edgecolor="white", label=f"Rule B (med={sb['p50']:.3f})")
        ax.axvline(sa["p50"], color="#1f77b4", linestyle="--", linewidth=1.4)
        ax.axvline(sb["p50"], color="#d62728", linestyle="-", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Images")
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("OOF Per-Image Score Distributions: Rule A (Uniform) vs Rule B (Weighted), [in-sample] Global Thresholds", y=1.02)
    fig.tight_layout()
    p1 = fig_dir / "oof_pq_dice_iou_distributions.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p1))

    # 2. Morphology Summary Comparison (Median & IQR Bars: GT vs OOF Rule B vs Test Rule B)
    gt_s = morph_data["ground_truth"]["all_gt"]["features"]
    oof_s = morph_data["oof_rule_B"]["morphology"]["features"]
    test_s = morph_data["test_rule_B"]["morphology"]["features"]
    gt_n = morph_data["ground_truth"]["all_gt"]["count"]
    oof_n = morph_data["oof_rule_B"]["morphology"]["count"]
    test_n = morph_data["test_rule_B"]["morphology"]["count"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), dpi=160)
    morph_metrics = [
        ("Area (px^2)", "area_px", True),
        ("Skeleton Length (px)", "skeleton_length_px", False),
        ("Mean Width (px)", "width_mean_px", False),
        ("Solidity (Area / Hull)", "solidity", False),
    ]
    labels = [f"GT\n(n={gt_n:,})", f"OOF B\n(n={oof_n:,})", f"Test B\n(n={test_n:,})"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    for ax, (title, mkey, use_log) in zip(axes, morph_metrics):
        meds = [gt_s[mkey]["median"], oof_s[mkey]["median"], test_s[mkey]["median"]]
        p25s = [gt_s[mkey]["p25"], oof_s[mkey]["p25"], test_s[mkey]["p25"]]
        p75s = [gt_s[mkey]["p75"], oof_s[mkey]["p75"], test_s[mkey]["p75"]]
        yerr_low = [m - lo for m, lo in zip(meds, p25s)]
        yerr_high = [hi - m for m, hi in zip(meds, p75s)]
        xs = np.arange(3)
        bars = ax.bar(xs, meds, yerr=[yerr_low, yerr_high], capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title)
        if use_log:
            ax.set_yscale("log")
        for b, m in zip(bars, meds):
            val_str = f"{m:.0f}" if m >= 50 else f"{m:.2f}"
            ax.text(b.get_x() + b.get_width() / 2, m * 1.03, val_str, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.suptitle("Solar Filament Morphology Comparison: Ground Truth vs OOF Rule B vs Test Rule B (Median & IQR)", y=1.02)
    fig.tight_layout()
    p2 = fig_dir / "morphology_gt_vs_pred_distributions.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p2))

    # 3. Morphology-Stratified PQ / SQ / RQ (OOF Rule B)
    strat = morph_data["oof_rule_B"]["stratified_pq"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), dpi=160)
    strat_specs = [
        ("Area Bin (px^2)", "area_bin", ["<300", "300-1000", "1000-3000", ">=3000"]),
        ("Mean Width Bin (px)", "width_bin", ["<5", "5-10", "10-18", ">=18"]),
        ("Elongation Bin (PCA)", "elong_bin", ["<3", "3-7", "7-15", ">=15"]),
    ]
    w = 0.25
    for ax, (title, skey, order) in zip(axes, strat_specs):
        dmap = strat[skey]
        x = np.arange(len(order))
        pqs = [dmap[b]["pq"] for b in order]
        sqs = [dmap[b]["sq"] for b in order]
        rqs = [dmap[b]["rq"] for b in order]
        ax.bar(x - w, pqs, width=w, label="PQ", color="#1f77b4")
        ax.bar(x, sqs, width=w, label="SQ", color="#2ca02c")
        ax.bar(x + w, rqs, width=w, label="RQ", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        ax.set_ylim(0.0, 0.88)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8.5)
    oof_b_pq = morph_data["oof_rule_B"]["global_pq"]["pq"]
    fig.suptitle(f"OOF Rule B Performance Stratified by Filament Morphology ([in-sample] PQ = {oof_b_pq:.4f})", y=1.02)
    fig.tight_layout()
    p3 = fig_dir / "morphology_stratified_pq.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p3))

    # 4. Matching Multi-Hit Analysis & TP vs FP vs FN Morphological Signatures (Rule B)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), dpi=160)
    # Left: 1-to-many / many-to-1 within-annotator & cross-annotator (Rule B)
    m_b = res_B["matching_analysis"]
    cats = [
        "GT hit >=2x\n(IoU>0.5)",
        "Pred hit >=2\n(Same Ann,\nIoU>0.5)",
        "Pred hit >=2\n(Cross-Ann,\nIoU>0.5)",
        "GT overlap >=2x\n(Any px >0)",
        "Pred overlap >=2\n(Any px >0)",
    ]
    vals = [
        m_b["within_annotator_iou_gt_05"]["gt_multi_hit_count"],
        m_b["within_annotator_iou_gt_05"]["pred_multi_hit_count"],
        m_b["multi_annotator_cross_record_iou_gt_05"]["pred_matched_ge2_cross_annotators_only"],
        m_b["within_annotator_any_overlap_iou_gt_0"]["gt_multi_overlap_count"],
        m_b["within_annotator_any_overlap_iou_gt_0"]["pred_multi_overlap_count"],
    ]
    bars = axes[0].bar(np.arange(len(cats)), vals, color=["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b"], edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(np.arange(len(cats)))
    axes[0].set_xticklabels(cats, fontsize=8.5)
    axes[0].set_ylabel("Event Count (OOF Rule B)")
    axes[0].set_title("One-to-Many & Many-to-One Matching Counts")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 30, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: TP vs FP vs FN relative shift vs Overall GT medians
    tp_m = morph_data["oof_rule_B"]["tp_pred_morphology"]["features"]
    fp_m = morph_data["oof_rule_B"]["fp_pred_morphology"]["features"]
    fn_m = morph_data["oof_rule_B"]["fn_gt_morphology"]["features"]
    tp_n = morph_data["oof_rule_B"]["tp_pred_morphology"]["count"]
    fp_n = morph_data["oof_rule_B"]["fp_pred_morphology"]["count"]
    fn_n = morph_data["oof_rule_B"]["fn_gt_morphology"]["count"]
    feat_keys = ["area_px", "skeleton_length_px", "width_mean_px", "solidity"]
    feat_labels = ["Area", "Skel Length", "Mean Width", "Solidity"]
    gt_ref = [gt_s[k]["median"] for k in feat_keys]
    tp_rel = [100.0 * (tp_m[k]["median"] / ref - 1.0) for k, ref in zip(feat_keys, gt_ref)]
    fp_rel = [100.0 * (fp_m[k]["median"] / ref - 1.0) for k, ref in zip(feat_keys, gt_ref)]
    fn_rel = [100.0 * (fn_m[k]["median"] / ref - 1.0) for k, ref in zip(feat_keys, gt_ref)]
    x2 = np.arange(len(feat_keys))
    axes[1].bar(x2 - 0.25, tp_rel, width=0.25, label=f"TP Preds (n={tp_n:,})", color="#2ca02c")
    axes[1].bar(x2, fp_rel, width=0.25, label=f"FP Preds (n={fp_n:,})", color="#d62728")
    axes[1].bar(x2 + 0.25, fn_rel, width=0.25, label=f"FN GT (n={fn_n:,})", color="#1f77b4")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(feat_labels, fontsize=9)
    axes[1].set_ylabel("% Difference vs Overall GT Median")
    axes[1].set_title("Morphological Signature Bias: TP vs FP vs FN")
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    p4 = fig_dir / "matching_and_error_decomposition.png"
    fig.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p4))

    return generated


def main():
    parser = argparse.ArgumentParser(description="Compute OOF score distributions, matching analysis, compact Fold-0 PQ, and morphology figures.")
    parser.add_argument("--work_dir", type=str, default=str(REPO_ROOT / "work"), help="Directory containing folds.csv, gt_all.csv, and OOF caches")
    parser.add_argument("--weights_dir", type=str, default=str(REPO_ROOT / "weights"), help="Repository compact FP16 weights directory")
    parser.add_argument("--fp32_dir", type=str, default="", help="Optional directory containing FP32 checkpoints for sample-image parity comparison")
    parser.add_argument("--sample_dir", type=str, default=str(REPO_ROOT / "sample_images"), help="Sample images directory for CPU inference check")
    parser.add_argument("--morph_json", type=str, default=str(REPO_ROOT / "audit" / "morphology_analysis.json"), help="Existing morphology analysis JSON")
    parser.add_argument("--out_json", type=str, default=str(REPO_ROOT / "audit" / "final_release_analyses.json"), help="Output JSON summary file")
    parser.add_argument("--fig_dir", type=str, default=str(REPO_ROOT / "figures"), help="Directory to write generated PNG figures")
    parser.add_argument("--skip_sample_cpu", action="store_true", help="Skip live CPU inference check on sample_images/")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    t_start = time.time()

    gt_df = pd.read_csv(work_dir / "gt_all.csv")
    folds_df = pd.read_csv(work_dir / "folds.csv")
    gt_by_obs = build_gt_rles_by_obs(gt_df)
    all_obs = sorted(gt_by_obs.keys())
    f0_obs = sorted(folds_df[folds_df["fold"] == 0]["observation"].tolist())

    with open(REPO_ROOT / "configs" / "final_ensemble.json") as f:
        ens_cfg = json.load(f)

    print(f"[1/4] Loading cached 6-family OOF candidates from {work_dir}...", flush=True)
    raw_cands_all, n_pt_files = load_oof_raw_candidates(work_dir, folds_df)
    n_with_cands = sum(1 for v in raw_cands_all.values() if len(v) > 0)
    print(f"Loaded {len(raw_cands_all)} observations across {n_pt_files} family-cache files ({n_with_cands} with >=1 raw candidate, {len(raw_cands_all) - n_with_cands} empty).", flush=True)

    rule_a_weights = {fam: 1.0 for fam in FAMILY_SUBDIRS}
    rule_b_weights = {"clahe": 1.0, "anchors5": 1.0, "dice": 1.0, "a5dice": 1.0, "2048": 2.0, "1792_clahe": 2.0}

    print("[2/4] Fusing Rule A & Rule B and analyzing full 707 OOF + compact Fold-0...", flush=True)
    clusters_A = {obs: fuse_with_rule(raw_cands_all.get(obs, []), rule_a_weights, gamma=1.0) for obs in all_obs}
    clusters_B = {obs: fuse_with_rule(raw_cands_all.get(obs, []), rule_b_weights, gamma=1.5) for obs in all_obs}

    oof_707_A = analyze_rule_oof("Rule A (Uniform 30-model OOF, [in-sample])", clusters_A, all_obs, gt_by_obs, tau_score=0.85, tau_mask=0.55)
    oof_707_B = analyze_rule_oof("Rule B (Weighted 30-model OOF, [in-sample])", clusters_B, all_obs, gt_by_obs, tau_score=0.75, tau_mask=0.55)

    # Compact 6-model Fold-0 OOF evaluation (same 142 Fold-0 images, exact 6 Fold-0 models)
    f0_6m_A = analyze_rule_oof("Compact 6-model Fold-0 (Rule A, Fold-0 142 imgs, [in-sample])", clusters_A, f0_obs, gt_by_obs, tau_score=0.85, tau_mask=0.55)
    f0_6m_B = analyze_rule_oof("Compact 6-model Fold-0 (Rule B, Fold-0 142 imgs, [in-sample])", clusters_B, f0_obs, gt_by_obs, tau_score=0.75, tau_mask=0.55)

    print("[3/4] Generating 4 morphology and distribution figures...", flush=True)
    with open(args.morph_json) as f:
        morph_data = json.load(f)
    fig_paths = generate_figures(oof_707_A, oof_707_B, morph_data, Path(args.fig_dir))

    sample_check = {}
    if not args.skip_sample_cpu:
        print("[4/4] Running live CPU FP32-compute sample inference parity check...", flush=True)
        fp32_path = Path(args.fp32_dir).resolve() if args.fp32_dir else None
        sample_check = run_sample_fp16_vs_fp32_check(
            Path(args.weights_dir).resolve(),
            fp32_path,
            Path(args.sample_dir).resolve(),
            ens_cfg,
        )

    # Strip raw per-image arrays from JSON output to keep JSON compact
    for res in (oof_707_A, oof_707_B, f0_6m_A, f0_6m_B):
        res.pop("raw_arrays", None)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.time() - t_start),
        "matching_convention": "Official competition pq_eval convention: each (annotator, image) record scored independently against image predictions; IoU > 0.50 counts as TP (no bipartite matching; pixel-disjoint canvas prevents >1 pred hitting same GT above 0.50). Per-image PQ pools TP/FP/FN across annotator records of that image.",
        "oof_707_rule_A": oof_707_A,
        "oof_707_rule_B": oof_707_B,
        "compact_6model_fold0_142_rule_A": f0_6m_A,
        "compact_6model_fold0_142_rule_B": f0_6m_B,
        "sample_images_cpu_check": sample_check,
        "generated_figures": [os.path.relpath(p, REPO_ROOT) for p in fig_paths],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Saved {out_path} in {summary['elapsed_sec']:.1f}s")
    print(f"Rule A 707 OOF [in-sample]: PQ={oof_707_A['pooled_metrics']['pq']:.6f} | Per-img PQ med={oof_707_A['distributions']['per_image_pq']['p50']:.4f} | Union Dice med={oof_707_A['distributions']['per_image_semantic_union_dice']['p50']:.4f} | gt_multi_05={oof_707_A['matching_analysis']['within_annotator_iou_gt_05']['gt_multi_hit_count']} | pred_multi_05={oof_707_A['matching_analysis']['within_annotator_iou_gt_05']['pred_multi_hit_count']}")
    print(f"Rule B 707 OOF [in-sample]: PQ={oof_707_B['pooled_metrics']['pq']:.6f} | Per-img PQ med={oof_707_B['distributions']['per_image_pq']['p50']:.4f} | Union Dice med={oof_707_B['distributions']['per_image_semantic_union_dice']['p50']:.4f} | gt_multi_05={oof_707_B['matching_analysis']['within_annotator_iou_gt_05']['gt_multi_hit_count']} | pred_multi_05={oof_707_B['matching_analysis']['within_annotator_iou_gt_05']['pred_multi_hit_count']}")
    print(f"Compact 6-model Fold-0 (142 imgs): Rule A PQ={f0_6m_A['pooled_metrics']['pq']:.6f}, Rule B PQ={f0_6m_B['pooled_metrics']['pq']:.6f}")


if __name__ == "__main__":
    main()
