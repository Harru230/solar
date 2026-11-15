#!/usr/bin/env python3
"""
scripts/generate_r50_1536_fold0_preds.py - Generate fold-0 raw and thresholded predictions
for r50_1536_clahe_e30 (best checkpoint).

Workflow:
1. Loads work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt onto GPU.
2. Extracts raw detections with score >= 0.30 for all 142 fold-0 validation images.
   Stores boxes, scores, and soft mask probability crops in bbox crops.
   Saves under work/runs/r50_1536_clahe_e30/raw_detections_fold0.pt.
3. Generates the final prediction CSV at score=0.85, mask=0.55.
   Saves to work/runs/r50_1536_clahe_e30/pred_fold0_score0.85_mask0.55.csv (and work/pred_fold0.csv).
4. Evaluates against ground truth and asserts reproduction of PQ = 0.4248.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import (
    EMPTY_RLE,
    apply_mask_nms,
    find_dataset_paths,
    get_maskrcnn_model,
    get_run_dir,
    make_masks_disjoint,
    preprocess_solar_image,
    rle_encode_mask,
)
from scripts.predict import load_image


def generate_fold0_predictions():
    ckpt_path = "work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt"
    run_dir = "work/runs/r50_1536_clahe_e30"
    raw_out_path = os.path.join(run_dir, "raw_detections_fold0.pt")
    csv_out_path = os.path.join(run_dir, "pred_fold0_score0.85_mask0.55.csv")
    work_pred_path = "work/pred_fold0.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {ckpt_path} onto {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    img_size = config.get("img_size", 1536)

    model = get_maskrcnn_model(pretrained=False, min_size=img_size, max_size=img_size, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])
    val_obs = folds_df.loc[folds_df["fold"] == 0, "observation"].tolist()
    print(f"Validation images in fold 0: {len(val_obs)}")

    train_dir = paths["train_dir"]
    val_image_paths = []
    for o in val_obs:
        p = os.path.join(train_dir, f"{o}.jpeg")
        if not os.path.exists(p):
            p = os.path.join(train_dir, f"{o}.jpg")
        val_image_paths.append((o, p))

    # Pre-index GT
    gt_df = pd.read_csv(paths["gt_all_csv"])
    val_obs_set = set(val_obs)
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_img = gt_ai.str.split("-", n=1).str[1]
    val_gt = gt_df[gt_img.isin(val_obs_set)].copy()

    raw_detections_by_obs = {}
    csv_rows = []

    scale_to_2048 = 2048.0 / img_size
    batch_size = 2

    t0 = time.time()
    print(f"Running inference across {len(val_image_paths)} images (batch size {batch_size})...")

    with torch.no_grad():
        for b_idx in range(0, len(val_image_paths), batch_size):
            batch = val_image_paths[b_idx : b_idx + batch_size]
            tensors = []
            obs_list = []
            for obs, p in batch:
                t, _ = load_image(p, img_size=img_size)
                tensors.append(t.to(device))
                obs_list.append(obs)

            outputs = model(tensors)

            for i, output in enumerate(outputs):
                obs = obs_list[i]
                scores = output["scores"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy() * scale_to_2048  # scale to 2048
                masks = output["masks"]  # [N, 1, img_size, img_size]

                # Filter at raw score >= 0.30
                keep_030 = scores >= 0.30
                scores_030 = scores[keep_030]
                boxes_030 = boxes[keep_030]
                masks_030 = masks[keep_030]

                raw_detections = []
                if len(scores_030) > 0:
                    # Upsample masks to 2048x2048
                    masks_up = F.interpolate(
                        masks_030.float(), size=(2048, 2048), mode="bilinear", align_corners=False
                    ).squeeze(1).cpu().numpy()

                    for k in range(len(scores_030)):
                        b = boxes_030[k]
                        x1 = max(0, int(np.floor(b[0])))
                        y1 = max(0, int(np.floor(b[1])))
                        x2 = min(2048, int(np.ceil(b[2])) + 1)
                        y2 = min(2048, int(np.ceil(b[3])) + 1)

                        crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float16)

                        raw_detections.append({
                            "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                            "score": float(scores_030[k]),
                            "crop": crop_soft,
                        })

                raw_detections_by_obs[obs] = raw_detections

                # ─────────────────────────────────────────────────────────────
                # Standard threshold tuning pipeline for score=0.85, mask=0.55
                # ─────────────────────────────────────────────────────────────
                keep_base = scores >= 0.05
                if np.any(keep_base):
                    s_kept = scores[keep_base]
                    m_kept = masks[keep_base]
                    K_kept = len(s_kept)

                    m_512 = F.interpolate(m_kept.float(), size=(512, 512), mode="nearest").squeeze(1)
                    bin_512 = (m_512 >= 0.55).float()
                    areas_512 = bin_512.sum(dim=(-2, -1))
                    valid_512 = areas_512 > 0

                    if valid_512.any():
                        flat = bin_512.view(K_kept, -1)
                        inter = torch.mm(flat, flat.t())
                        union = areas_512.unsqueeze(1) + areas_512.unsqueeze(0) - inter
                        iou_gpu = torch.where(union > 0, inter / union, torch.zeros_like(inter)).cpu().numpy()

                        order = np.argsort(s_kept)[::-1]
                        keep_nms = []
                        for oi in range(K_kept):
                            if not valid_512[order[oi]]:
                                continue
                            orig_i = order[oi]
                            overlap = False
                            for kn in keep_nms:
                                if iou_gpu[orig_i, order[kn]] > 0.5:
                                    overlap = True
                                    break
                            if not overlap:
                                keep_nms.append(oi)

                        kept_indices = [int(order[oi]) for oi in keep_nms]

                        if kept_indices:
                            # 2048 upsampling
                            m_up = F.interpolate(
                                m_kept[kept_indices].float(), size=(2048, 2048), mode="bilinear", align_corners=False
                            ).squeeze(1)
                            bin_2048 = (m_up >= 0.55).byte().cpu().numpy()
                            s_nms = [float(s_kept[idx]) for idx in kept_indices]

                            # Disjoint enforcement
                            clean_m, clean_s = make_masks_disjoint(list(bin_2048), s_nms)

                            # Filter at score >= 0.85
                            kept_rows = []
                            for cm, cs in zip(clean_m, clean_s):
                                if cs >= 0.85 and cm.sum() > 0:
                                    rle = rle_encode_mask(cm)
                                    kept_rows.append(rle)

                            if kept_rows:
                                for kid, rle in enumerate(kept_rows, start=1):
                                    csv_rows.append({"filament_id": f"{obs}_{kid}", "segmentation_rle": rle})
                            else:
                                csv_rows.append({"filament_id": f"{obs}_1", "segmentation_rle": EMPTY_RLE})
                        else:
                            csv_rows.append({"filament_id": f"{obs}_1", "segmentation_rle": EMPTY_RLE})
                    else:
                        csv_rows.append({"filament_id": f"{obs}_1", "segmentation_rle": EMPTY_RLE})
                else:
                    csv_rows.append({"filament_id": f"{obs}_1", "segmentation_rle": EMPTY_RLE})

    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s.")

    # Save raw detections
    print(f"Saving raw detections to {raw_out_path}...")
    torch.save(raw_detections_by_obs, raw_out_path)
    total_raw = sum(len(v) for v in raw_detections_by_obs.values())
    print(f"Saved {total_raw} raw detections (score >= 0.30) across {len(raw_detections_by_obs)} observations.")

    # Save final CSV
    pred_df = pd.DataFrame(csv_rows)
    pred_df.to_csv(csv_out_path, index=False)
    pred_df.to_csv(work_pred_path, index=False)
    print(f"Saved {len(pred_df)} rows to {csv_out_path} and updated {work_pred_path}.")

    # Evaluate against GT
    res = pq_eval.score(val_gt, pred_df)
    print("\n==================================================================")
    print("           REPRODUCIBILITY EVALUATION ON FOLD 0                   ")
    print("==================================================================")
    print(f"PQ:             {res['pq']:.4f}  (Expected: 0.4248)")
    print(f"SQ:             {res['sq']:.4f}  (Expected: 0.6626)")
    print(f"RQ:             {res['rq']:.4f}  (Expected: 0.6412)")
    print(f"TP:             {res['tp']}      (Expected: 1158)")
    print(f"FP:             {res['fp']}       (Expected: 552)")
    print(f"FN:             {res['fn']}       (Expected: 744)")
    print(f"gt_multi_hit:   {res['gt_multi_hit']}")
    print(f"pred_multi_hit: {res['pred_multi_hit']}")
    print("==================================================================")

    diff = abs(res['pq'] - 0.42484885)
    if diff < 1e-3:
        print(f"[SUCCESS] Exactly reproduced PQ = {res['pq']:.4f}!")
    else:
        print(f"[WARNING] Discrepancy detected: got {res['pq']:.4f}, expected 0.4248")

    return res, pred_df, raw_detections_by_obs


def main() -> None:
    """CLI wrapper for generate_fold0_predictions."""
    parser = argparse.ArgumentParser(description="Generate fold-0 raw and thresholded predictions for r50_1536_clahe_e30.")
    parser.parse_args()
    generate_fold0_predictions()


if __name__ == "__main__":
    main()
