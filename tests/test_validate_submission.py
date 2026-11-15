"""
tests/test_validate_submission.py - Comprehensive Unit Tests for Submission Validator.

Validates all 7 competition submission rules:
1. Columns: exact match ['filament_id', 'segmentation_rle'].
2. Unique IDs: '<stem>_<k>' pattern with non-negative integer k, all IDs unique.
3. Test Coverage: all 180 test stems present, no extra, no missing.
4. Valid 2048x2048 RLE: every string decodes to (2048, 2048) with valid COCO counts.
5. Empty row format: '<stem>_1, PPPP4' appears ONLY as single row for zero-prediction images.
6. Area Distribution / Masks > 1% of frame: counts instances > 41,943 pixels.
7. Canvas disjointness: zero overlapping pixels between distinct instances of same image.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.common import EMPTY_RLE
from scripts.validate_submission import validate_submission


def _make_small_test_dir(tmp_path: Path, stems: list) -> Path:
    """Helper to create a small directory of dummy JPEG images for isolated rule tests."""
    td = tmp_path / "small_test_images"
    td.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (td / f"{stem}.jpeg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return td


def test_validate_submission_fully_valid(tmp_path: Path, dummy_test_dir: Path, valid_submission_df: pd.DataFrame):
    """Verifies that a fully compliant submission passes all 7 validation rules on 180 stems."""
    csv_path = tmp_path / "valid_submission.csv"
    valid_submission_df.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(dummy_test_dir))

    assert report["columns_valid"] is True
    assert report["unique_ids_valid"] is True
    assert report["test_coverage_valid"] is True
    assert report["rle_valid"] is True
    assert report["empty_rows_valid"] is True
    assert report["overlap_images_count"] == 0
    assert report["all_passed"] is True
    assert len(report["errors"]) == 0


def test_validate_submission_missing_columns(tmp_path: Path):
    """Verifies failure when required columns are missing."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_missing = pd.DataFrame([{"id": "test_000_1", "rle": EMPTY_RLE}])
    csv_path = tmp_path / "missing_cols.csv"
    df_missing.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["columns_valid"] is False
    assert report["all_passed"] is False
    assert any("Columns mismatch" in err for err in report["errors"])


def test_validate_submission_extra_columns(tmp_path: Path):
    """Verifies failure when extra columns (e.g. 'score') are present."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_extra = pd.DataFrame([{"filament_id": "test_000_1", "segmentation_rle": EMPTY_RLE, "score": 0.95}])
    csv_path = tmp_path / "extra_cols.csv"
    df_extra.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["columns_valid"] is False
    assert report["all_passed"] is False


def test_validate_submission_duplicate_filament_ids(tmp_path: Path):
    """Verifies failure when duplicate filament_ids are present."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_dup = pd.DataFrame([
        {"filament_id": "test_000_1", "segmentation_rle": EMPTY_RLE},
        {"filament_id": "test_000_1", "segmentation_rle": EMPTY_RLE},
    ])
    csv_path = tmp_path / "dup_ids.csv"
    df_dup.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["unique_ids_valid"] is False
    assert report["all_passed"] is False
    assert any("Duplicate filament_ids found" in err for err in report["errors"])


def test_validate_submission_malformed_filament_ids(tmp_path: Path):
    """Verifies failure when filament_ids do not match '<stem>_<k>' pattern."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_bad_id = pd.DataFrame([{"filament_id": "bad-id-without-underscore", "segmentation_rle": EMPTY_RLE}])
    csv_path = tmp_path / "bad_id.csv"
    df_bad_id.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["unique_ids_valid"] is False
    assert report["all_passed"] is False
    assert any("Invalid filament_id pattern" in err for err in report["errors"])


def test_validate_submission_missing_test_stems(tmp_path: Path, dummy_test_dir: Path):
    """Verifies failure when some test stems (e.g. 1 out of 180) are present and 179 are missing."""
    df_missing_stem = pd.DataFrame([{"filament_id": "test_000_1", "segmentation_rle": EMPTY_RLE}])
    csv_path = tmp_path / "missing_stem.csv"
    df_missing_stem.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(dummy_test_dir))
    assert report["test_coverage_valid"] is False
    assert report["all_passed"] is False
    assert any("Missing test stems" in err for err in report["errors"])


def test_validate_submission_extra_test_stems(tmp_path: Path):
    """
    Verifies error reporting when an unrecognized stem outside the test set is present.
    (Note: Also documents the edge condition in validate_submission:105 where extra stems
    are logged when test coverage fails).
    """
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_extra_stem = pd.DataFrame([
        {"filament_id": "test_000_1", "segmentation_rle": EMPTY_RLE},
        {"filament_id": "unknown_stem_1", "segmentation_rle": EMPTY_RLE},
    ])
    csv_path = tmp_path / "extra_stem.csv"
    df_extra_stem.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["test_coverage_valid"] is False
    assert report["all_passed"] is False
    assert any("Extra test stems" in err for err in report["errors"])


def test_validate_submission_corrupt_rle(tmp_path: Path):
    """Verifies failure when an RLE string cannot be decoded by pycocotools."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_corrupt = pd.DataFrame([{"filament_id": "test_000_1", "segmentation_rle": "NOT_A_VALID_COCO_RLE"}])
    csv_path = tmp_path / "corrupt_rle.csv"
    df_corrupt.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["rle_valid"] is False
    assert report["all_passed"] is False
    assert any("RLE decode error" in err for err in report["errors"])


def test_validate_submission_empty_rle_string(tmp_path: Path):
    """Verifies failure when an RLE string is empty or whitespace."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    df_empty_str = pd.DataFrame([{"filament_id": "test_000_1", "segmentation_rle": "   "}])
    csv_path = tmp_path / "empty_str_rle.csv"
    df_empty_str.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["rle_valid"] is False
    assert report["all_passed"] is False
    assert any("empty/null RLE string" in err for err in report["errors"])


def test_validate_submission_invalid_empty_row_with_multiple_instances(
    tmp_path: Path, helper_make_rect, helper_encode_rle
):
    """
    Verifies rule 5: an empty mask row is ONLY permitted as the solitary row
    for an image. Having multiple rows when one is empty is invalid.
    """
    td = _make_small_test_dir(tmp_path, ["test_000"])
    m1 = helper_encode_rle(helper_make_rect(10, 10, 50, 50))
    df_bad_empty = pd.DataFrame([
        {"filament_id": "test_000_1", "segmentation_rle": m1},
        {"filament_id": "test_000_2", "segmentation_rle": EMPTY_RLE},
    ])
    csv_path = tmp_path / "bad_empty.csv"
    df_bad_empty.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["empty_rows_valid"] is False
    assert report["all_passed"] is False
    assert any("contains empty mask rows" in err for err in report["errors"])


def test_validate_submission_overlapping_instances_fails(
    tmp_path: Path, helper_make_rect, helper_encode_rle
):
    """
    Verifies rule 7: pixel-level overlaps between masks of the same image fail validation.
    """
    td = _make_small_test_dir(tmp_path, ["test_000"])
    m1 = helper_encode_rle(helper_make_rect(100, 100, 200, 200))
    m2 = helper_encode_rle(helper_make_rect(150, 150, 250, 250))
    df_overlap = pd.DataFrame([
        {"filament_id": "test_000_1", "segmentation_rle": m1},
        {"filament_id": "test_000_2", "segmentation_rle": m2},
    ])
    csv_path = tmp_path / "overlap.csv"
    df_overlap.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["overlap_images_count"] > 0
    assert report["all_passed"] is False
    assert report["overlap_details"][0]["stem"] == "test_000"
    assert report["overlap_details"][0]["overlapping_pixels"] == 50 * 50  # 2500 px


def test_validate_submission_outlier_masks_gt_1pct(
    tmp_path: Path, helper_make_rect, helper_encode_rle
):
    """
    Verifies rule 6: instances with area > 41,943 px (> 1% of 2048x2048)
    are detected and reported in masks_gt_1pct_list.
    """
    td = _make_small_test_dir(tmp_path, ["test_000"])
    large_mask = helper_encode_rle(helper_make_rect(0, 0, 300, 200))  # 60,000 px
    df_outlier = pd.DataFrame([{"filament_id": "test_000_1", "segmentation_rle": large_mask}])
    csv_path = tmp_path / "outlier.csv"
    df_outlier.to_csv(csv_path, index=False)

    report = validate_submission(str(csv_path), str(td))
    assert report["masks_gt_1pct_count"] == 1
    assert report["masks_gt_1pct_list"][0][0] == "test_000_1"
    assert report["masks_gt_1pct_list"][0][1] == 60000
    assert report["rle_valid"] is True
    assert report["overlap_images_count"] == 0


def test_validate_submission_missing_file(tmp_path: Path):
    """Verifies graceful error handling when CSV file does not exist."""
    td = _make_small_test_dir(tmp_path, ["test_000"])
    non_existent = tmp_path / "does_not_exist.csv"
    report = validate_submission(str(non_existent), str(td))
    assert report["all_passed"] is False
    assert any("File not found" in err for err in report["errors"])
