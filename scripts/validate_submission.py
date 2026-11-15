#!/usr/bin/env python3
"""
scripts/validate_submission.py - Submission Validator for Solar Filament Segmentation 2026.

Validates:
1. Columns: exactly ['filament_id', 'segmentation_rle'].
2. Unique IDs: matches '<stem>_<k>' pattern with non-negative integer k, all IDs unique.
3. Test Coverage: all 180 test stems from test_images are present.
4. Valid 2048x2048 RLE: every RLE decodes successfully to (2048, 2048) with counts sum = 2048^2 = 4194304.
5. Placeholder Empty Rows: empty masks (area = 0, e.g. 'PPPP4') appear ONLY as single placeholders for zero-prediction images.
6. Area Distribution / Masks > 1% of frame: counts and reports instances with area > 41,943 pixels (1% of 2048x2048).
7. Instance Overlaps: checks pixel-level overlaps (intersection > 0) between distinct instances of the same image.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import EMPTY_RLE, find_dataset_paths

FRAME_SIZE = 2048
TOTAL_PIXELS = FRAME_SIZE * FRAME_SIZE  # 4194304
ONE_PERCENT_PIXELS = 0.01 * TOTAL_PIXELS  # 41943.04


def validate_submission(csv_path: str, test_images_dir: str, expected_count: int = 180) -> dict:
    results = {
        "csv_path": csv_path,
        "total_rows": 0,
        "unique_images": 0,
        "columns_valid": False,
        "unique_ids_valid": False,
        "test_coverage_valid": False,
        "rle_valid": False,
        "empty_rows_valid": False,
        "masks_gt_1pct_count": 0,
        "masks_gt_1pct_list": [],
        "overlap_images_count": 0,
        "overlap_details": [],
        "all_passed": False,
        "errors": [],
        "warnings": [],
    }

    if not os.path.isfile(csv_path):
        results["errors"].append(f"File not found: {csv_path}")
        return results

    # 1. Read CSV and check columns
    df = pd.read_csv(csv_path)
    results["total_rows"] = len(df)
    expected_cols = ["filament_id", "segmentation_rle"]
    if list(df.columns) == expected_cols:
        results["columns_valid"] = True
    else:
        results["errors"].append(f"Columns mismatch: got {list(df.columns)}, expected {expected_cols}")
        if "filament_id" not in df.columns or "segmentation_rle" not in df.columns:
            return results

    # 2. Check unique IDs and '<stem>_<k>' format
    ids = df["filament_id"].tolist()
    if len(ids) == len(set(ids)):
        results["unique_ids_valid"] = True
    else:
        dup_count = len(ids) - len(set(ids))
        results["errors"].append(f"Duplicate filament_ids found: {dup_count} duplicates")

    pattern = re.compile(r"^(.+)_(\d+)$")
    invalid_id_format = []
    stem_to_rows = defaultdict(list)

    for idx, fid in enumerate(ids):
        match = pattern.match(str(fid))
        if not match:
            invalid_id_format.append(fid)
        else:
            stem, k = match.groups()
            stem_to_rows[stem].append((idx, fid, k, df.at[idx, "segmentation_rle"]))

    if invalid_id_format:
        results["unique_ids_valid"] = False
        results["errors"].append(f"Invalid filament_id pattern for {len(invalid_id_format)} ids (e.g. {invalid_id_format[:3]})")

    results["unique_images"] = len(stem_to_rows)

    # 3. Check all expected test stems present
    test_files = glob.glob(os.path.join(test_images_dir, "*.*"))
    expected_stems = {Path(p).stem for p in test_files if p.lower().endswith((".jpeg", ".jpg", ".png"))}
    results["expected_test_stems_count"] = len(expected_stems)

    missing_stems = expected_stems - set(stem_to_rows.keys())
    extra_stems = set(stem_to_rows.keys()) - expected_stems

    if len(missing_stems) == 0 and len(extra_stems) == 0 and len(expected_stems) == expected_count:
        results["test_coverage_valid"] = True
    else:
        if missing_stems:
            results["errors"].append(f"Missing test stems ({len(missing_stems)}): {list(missing_stems)[:5]}")
        if extra_stems:
            results["errors"].append(f"Extra test stems not in test set ({len(extra_stems)}): {list(extra_stems)[:5]}")
        if len(expected_stems) != expected_count:
            results["warnings"].append(
                f"Expected {expected_count} test stems, but found {len(expected_stems)} in {test_images_dir}"
            )

    # 4. Validate RLE strings, empty rows, area distribution, and overlaps
    rle_errors = 0
    empty_row_errors = []
    masks_gt_1pct = []
    overlap_images = []

    for stem, rows in stem_to_rows.items():
        image_masks = []
        image_areas = []
        is_empty_image = False

        for row_idx, fid, k, rle_str in rows:
            if not isinstance(rle_str, str) or len(rle_str.strip()) == 0:
                rle_errors += 1
                results["errors"].append(f"Row {row_idx} ({fid}) has empty/null RLE string")
                continue

            # Decode RLE
            try:
                rle_obj = {"size": [FRAME_SIZE, FRAME_SIZE], "counts": rle_str.encode("ascii")}
                mask = mu.decode(rle_obj)
                area = int(mask.sum())
                image_masks.append((fid, mask))
                image_areas.append((fid, area))

                if area == 0:
                    is_empty_image = True

                if area > ONE_PERCENT_PIXELS:
                    pct = (area / TOTAL_PIXELS) * 100
                    masks_gt_1pct.append((fid, area, pct))

            except Exception as e:
                rle_errors += 1
                results["errors"].append(f"RLE decode error for {fid}: {e}")

        # 5. Check placeholder empty rows
        # If any row is empty, there must be EXACTLY 1 row for this stem, and area must be 0
        if is_empty_image:
            if len(rows) > 1:
                empty_row_errors.append(f"Stem '{stem}' has {len(rows)} rows, but contains empty mask rows!")
            elif image_areas[0][1] != 0:
                empty_row_errors.append(f"Stem '{stem}' marked empty but has non-zero mask area {image_areas[0][1]}")

        # 7. Check instance overlaps
        # Only check if image has >= 2 non-empty masks
        non_empty_masks = [m for fid, m in image_masks if m.sum() > 0]
        if len(non_empty_masks) >= 2:
            # Check overlap via pairwise bitwise AND or sum
            mask_stack = np.stack(non_empty_masks, axis=0)
            pixel_overlap = mask_stack.sum(axis=0)
            max_overlap_count = int(pixel_overlap.max())
            if max_overlap_count > 1:
                # Count overlapping pixels
                overlapping_pixels = int((pixel_overlap > 1).sum())
                overlap_images.append({
                    "stem": stem,
                    "num_instances": len(non_empty_masks),
                    "overlapping_pixels": overlapping_pixels,
                    "max_instances_per_pixel": max_overlap_count,
                })

    results["rle_valid"] = (rle_errors == 0)
    results["empty_rows_valid"] = (len(empty_row_errors) == 0)
    if empty_row_errors:
        results["errors"].extend(empty_row_errors[:5])

    results["masks_gt_1pct_count"] = len(masks_gt_1pct)
    results["masks_gt_1pct_list"] = masks_gt_1pct

    results["overlap_images_count"] = len(overlap_images)
    results["overlap_details"] = overlap_images

    # Overall status
    results["all_passed"] = (
        results["columns_valid"]
        and results["unique_ids_valid"]
        and results["test_coverage_valid"]
        and results["rle_valid"]
        and results["empty_rows_valid"]
        and results["overlap_images_count"] == 0
    )

    return results


def print_report(res: dict):
    print("==================================================================")
    print("              SUBMISSION VALIDATION REPORT                        ")
    print("==================================================================")
    print(f"File:               {res['csv_path']}")
    print(f"Total Rows:         {res['total_rows']}")
    print(f"Unique Test Stems:  {res['unique_images']} (Expected: {res.get('expected_test_stems_count', 180)})")
    print("------------------------------------------------------------------")
    print(f"1. Column Names:       {'[PASS]' if res['columns_valid'] else '[FAIL]'}")
    print(f"2. Unique IDs (<s_k>): {'[PASS]' if res['unique_ids_valid'] else '[FAIL]'}")
    print(f"3. 180 Test Coverage:  {'[PASS]' if res['test_coverage_valid'] else '[FAIL]'}")
    print(f"4. 2048x2048 RLE Valid:{'[PASS]' if res['rle_valid'] else '[FAIL]'}")
    print(f"5. Empty Row Format:   {'[PASS]' if res['empty_rows_valid'] else '[FAIL]'}")
    print(f"6. Instance Overlaps:  {'[PASS] (0 overlapping images)' if res['overlap_images_count'] == 0 else f'[FAIL] ({res['overlap_images_count']} images with overlaps!)'}")
    print("------------------------------------------------------------------")
    print(f"Mask Area > 1% frame:  {res['masks_gt_1pct_count']} instances (> 41,943 px)")
    if res['masks_gt_1pct_count'] > 0:
        for fid, area, pct in res['masks_gt_1pct_list'][:5]:
            print(f"   - {fid}: {area} px ({pct:.2f}% of frame)")
        if res['masks_gt_1pct_count'] > 5:
            print(f"   ... and {res['masks_gt_1pct_count'] - 5} more")
    print("------------------------------------------------------------------")
    if res["overlap_images_count"] > 0:
        print(f"OVERLAP WARNING: {res['overlap_images_count']} images contain overlapping instances!")
        for item in res["overlap_details"][:5]:
            print(f"   - {item['stem']}: {item['overlapping_pixels']} overlapping pixels across {item['num_instances']} instances (max stack depth {item['max_instances_per_pixel']})")
        if res["overlap_images_count"] > 5:
            print(f"   ... and {res['overlap_images_count'] - 5} more")
    else:
        print("Instances are 100% pixel-disjoint (no multi-hit risk).")
    print("------------------------------------------------------------------")
    if res["errors"]:
        print(f"ERRORS ({len(res['errors'])}):")
        for err in res["errors"]:
            print(f"  [!] {err}")
    if res["warnings"]:
        print(f"WARNINGS ({len(res['warnings'])}):")
        for warn in res["warnings"]:
            print(f"  [*] {warn}")
    print("==================================================================")
    print(f"OVERALL STATUS: {'[PASSED ALL CHECKS]' if res['all_passed'] else '[VALIDATION FAILED]'}")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Validate Kaggle submission CSV for Solar Filament Challenge")
    parser.add_argument("--sub", type=str, default="work/submission_r50_1536.csv",
                        help="Path to submission CSV to validate")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test images directory (default: auto-detected)")
    parser.add_argument("--expected_count", type=int, default=180,
                        help="Expected number of image stems in test_dir (default: 180)")
    args = parser.parse_args()

    paths = find_dataset_paths()
    test_dir = args.test_dir or paths["test_dir"]

    if not test_dir or not os.path.exists(test_dir):
        print(f"Error: test_dir not found at {test_dir}")
        sys.exit(1)

    # Fallback paths for submission
    sub_path = args.sub
    if not os.path.exists(sub_path):
        candidates = [
            "work/submission.csv",
            "work/runs/r50_1536_clahe_e30/submission.csv",
        ]
        for c in candidates:
            cand_p = c if os.path.isabs(c) else os.path.join(paths["repo_root"], c)
            if os.path.exists(cand_p):
                sub_path = cand_p
                break

    report = validate_submission(sub_path, test_dir, expected_count=args.expected_count)
    print_report(report)
    if not report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
