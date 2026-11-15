#!/usr/bin/env python3
"""
scripts/postprocess.py - Disjoint canvas post-processing for Solar Filament Segmentation.

Enforces pixel-wise disjoint predictions per image:
1. Sorts instances by score descending (or existing file order if score is omitted).
2. Maintains an occupied binary canvas (2048x2048).
3. For each candidate mask:
   - Subtracts already occupied pixels: clean_mask = mask & (~occupied)
   - Drops the mask if clean_mask area < 50% of original area or < min_area (default 10 px)
   - Otherwise keeps clean_mask and marks canvas: occupied |= clean_mask
4. Ensures images with zero remaining masks have a placeholder empty row ('<stem>_1', 'PPPP4').
5. Verifies pred_multi_hit == 0.
6. Evaluates and reports PQ change against ground truth if provided.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import EMPTY_RLE, find_dataset_paths

FRAME_SIZE = 2048


def postprocess_predictions(
    pred_df: pd.DataFrame,
    min_area: int = 10,
    area_retention_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Applies disjoint canvas post-processing to a predictions DataFrame.
    """
    has_score = "score" in pred_df.columns

    # Group by image stem
    img_to_rows = defaultdict(list)
    for idx, row in pred_df.iterrows():
        fid = str(row["filament_id"])
        stem = fid.split("_")[0]
        score_val = float(row["score"]) if has_score else 0.0
        img_to_rows[stem].append({
            "idx": idx,
            "filament_id": fid,
            "segmentation_rle": str(row["segmentation_rle"]),
            "score": score_val,
        })

    out_rows = []
    total_dropped_ratio = 0
    total_dropped_min_area = 0
    total_kept = 0

    for stem, items in img_to_rows.items():
        if has_score:
            items = sorted(items, key=lambda x: x["score"], reverse=True)

        occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        kept_masks = []

        for item in items:
            rle_str = item["segmentation_rle"]
            if rle_str == EMPTY_RLE:
                continue

            rle_obj = {"size": [FRAME_SIZE, FRAME_SIZE], "counts": rle_str.encode("ascii")}
            m = mu.decode(rle_obj).astype(bool)
            orig_area = int(m.sum())
            if orig_area == 0:
                continue

            # Subtract occupied pixels
            clean_m = m & (~occupied)
            clean_area = int(clean_m.sum())

            if clean_area < min_area:
                total_dropped_min_area += 1
                continue
            if (clean_area / orig_area) < area_retention_ratio:
                total_dropped_ratio += 1
                continue

            # Keep clean mask and update canvas
            occupied |= clean_m
            fortran_arr = np.asfortranarray(clean_m.astype(np.uint8))
            enc_rle = mu.encode(fortran_arr)["counts"].decode("ascii")
            kept_masks.append({
                "clean_rle": enc_rle,
                "score": item["score"],
                "area": clean_area,
            })
            total_kept += 1

        if not kept_masks:
            out_row = {"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE}
            if has_score:
                out_row["score"] = 0.0
            out_rows.append(out_row)
        else:
            for k, km in enumerate(kept_masks, start=1):
                out_row = {"filament_id": f"{stem}_{k}", "segmentation_rle": km["clean_rle"]}
                if has_score:
                    out_row["score"] = km["score"]
                out_rows.append(out_row)

    res_df = pd.DataFrame(out_rows)
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Disjoint canvas post-processing for filament masks")
    parser.add_argument("--pred", type=str, default="work/pred_fold0.csv", help="Input prediction CSV")
    parser.add_argument("--out", type=str, default=None, help="Output post-processed prediction CSV")
    parser.add_argument("--gt", type=str, default="work/gt_all.csv", help="Ground truth CSV for evaluation")
    parser.add_argument("--folds", type=str, default="work/folds.csv", help="Folds CSV for evaluation")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to filter GT evaluation (default: 0)")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum pixel area for kept masks")
    parser.add_argument("--area_retention", type=float, default=0.5, help="Minimum area retention ratio (default 0.5)")
    args = parser.parse_args()

    if not os.path.isfile(args.pred):
        print(f"Error: {args.pred} does not exist.")
        sys.exit(1)

    print("==================================================================")
    print("           DISJOINT CANVAS POST-PROCESSING                        ")
    print("==================================================================")
    print(f"Input predictions:   {args.pred}")
    print(f"Min area threshold:  {args.min_area} px")
    print(f"Area retention:      {args.area_retention * 100:.0f}%")

    pred_df = pd.read_csv(args.pred)
    print(f"Loaded {len(pred_df)} rows from input CSV.")

    # Apply post-processing
    post_df = postprocess_predictions(
        pred_df=pred_df,
        min_area=args.min_area,
        area_retention_ratio=args.area_retention,
    )
    print(f"Postprocessed predictions: {len(post_df)} rows.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        post_df.to_csv(args.out, index=False)
        print(f"Saved postprocessed predictions to: {args.out}")

    # Evaluate official counting metrics if GT is available
    if args.gt and os.path.isfile(args.gt):
        gt_df = pd.read_csv(args.gt)
        if args.folds and os.path.isfile(args.folds) and args.fold is not None:
            folds_df = pd.read_csv(args.folds)
            val_obs = set(folds_df.loc[folds_df["fold"] == args.fold, "observation"])
            gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
            gt_img = gt_ai.str.split("-", n=1).str[1]
            gt_df = gt_df[gt_img.isin(val_obs)].copy()

        print("\nEvaluating Official PQ Metrics (Before vs After)...")
        res_before = pq_eval.score(gt_df, pred_df)
        res_after = pq_eval.score(gt_df, post_df)

        print("------------------------------------------------------------------")
        print(f"Metric          Before          After           Change")
        print("------------------------------------------------------------------")
        for k in ["pq", "sq", "rq"]:
            diff = res_after[k] - res_before[k]
            print(f"{k.upper():12s}    {res_before[k]:.4f}          {res_after[k]:.4f}          {diff:+.4f}")
        for k in ["tp", "fp", "fn", "gt_multi_hit", "pred_multi_hit"]:
            diff = res_after[k] - res_before[k]
            print(f"{k:12s}    {res_before[k]:<14d}  {res_after[k]:<14d}  {diff:+d}")
        print("------------------------------------------------------------------")

        # Check pred_multi_hit == 0
        assert res_after["pred_multi_hit"] == 0, f"Error: pred_multi_hit={res_after['pred_multi_hit']} (expected 0)!"
        print("Verification PASS: pred_multi_hit == 0 confirmed!")
    print("==================================================================")


if __name__ == "__main__":
    main()
