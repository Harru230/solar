"""
tests/test_canvas_disjointness.py - Unit Tests for Canvas Disjointness Enforcement.

Tests:
1. make_masks_disjoint and postprocess_predictions.
2. Isolated non-overlapping masks remain completely intact.
3. Completely overlapping duplicates are suppressed (highest score kept).
4. Partial overlap resolution favoring higher confidence scores.
5. 50% area retention cutoff: masks losing >= 50% of original area are discarded.
6. 10 px minimum area cutoff: clean masks with < 10 pixels are discarded.
7. Nested concentric shapes (inner higher score vs outer higher score).
8. Empty inputs and all-occluded inputs fallback to '<stem>_1, PPPP4'.
9. Mathematical invariant assertion: max(sum(M_i)) <= 1.
"""
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mu

from scripts.common import EMPTY_RLE, make_masks_disjoint, rle_encode_mask
from scripts.postprocess import postprocess_predictions


def test_make_masks_disjoint_no_overlap(helper_make_rect):
    """Verifies that non-overlapping masks are completely retained without alteration."""
    m1 = helper_make_rect(10, 10, 50, 50)     # 40x40 = 1600 px
    m2 = helper_make_rect(100, 100, 150, 150) # 50x50 = 2500 px
    m3 = helper_make_rect(200, 200, 220, 220) # 20x20 = 400 px

    masks = [m1, m2, m3]
    scores = [0.90, 0.85, 0.70]

    clean_masks, clean_scores = make_masks_disjoint(masks, scores)

    assert len(clean_masks) == 3
    assert clean_scores == [0.90, 0.85, 0.70]
    assert int(clean_masks[0].sum()) == 1600
    assert int(clean_masks[1].sum()) == 2500
    assert int(clean_masks[2].sum()) == 400

    stack = np.stack(clean_masks, axis=0)
    assert int(stack.sum(axis=0).max()) <= 1


def test_make_masks_disjoint_duplicate_suppression(helper_make_rect):
    """Verifies that exact duplicates are suppressed, keeping only the highest scoring mask."""
    m1 = helper_make_rect(50, 50, 100, 100) # 50x50 = 2500 px
    m2 = helper_make_rect(50, 50, 100, 100) # identical
    m3 = helper_make_rect(50, 50, 100, 100) # identical

    masks = [m1, m2, m3]
    scores = [0.60, 0.95, 0.80]

    clean_masks, clean_scores = make_masks_disjoint(masks, scores)

    assert len(clean_masks) == 1
    assert clean_scores == [0.95]
    assert int(clean_masks[0].sum()) == 2500


def test_make_masks_disjoint_partial_overlap_score_preference(helper_make_rect):
    """
    Verifies that overlap resolution strictly favors the higher scoring mask:
    - High-score mask retains 100% of its area.
    - Low-score mask has overlapping region subtracted.
    """
    # m1: rows 10..50, cols 10..50 (1600 px)
    # m2: rows 30..70, cols 10..50 (1600 px)
    # Overlap: rows 30..50, cols 10..50 (800 px)
    m1 = helper_make_rect(10, 10, 50, 50)
    m2 = helper_make_rect(30, 10, 70, 50)

    # Case A: m1 has higher score (0.90 vs 0.70)
    clean_masks, clean_scores = make_masks_disjoint([m1, m2], [0.90, 0.70])
    assert len(clean_masks) == 2
    assert clean_scores == [0.90, 0.70]
    # m1 retained completely
    assert int(clean_masks[0].sum()) == 1600
    # m2 retains rows 50..70 (800 px = 50% retention >= 50%)
    assert int(clean_masks[1].sum()) == 800
    # Overlap between clean masks is zero
    assert int((clean_masks[0] & clean_masks[1]).sum()) == 0

    # Case B: m2 has higher score (0.70 vs 0.95)
    clean_masks_b, clean_scores_b = make_masks_disjoint([m1, m2], [0.70, 0.95])
    assert len(clean_masks_b) == 2
    assert clean_scores_b == [0.95, 0.70]
    # m2 retained completely
    assert int(clean_masks_b[0].sum()) == 1600
    # m1 retains rows 10..30 (800 px = 50% retention >= 50%)
    assert int(clean_masks_b[1].sum()) == 800
    assert int((clean_masks_b[0] & clean_masks_b[1]).sum()) == 0


def test_make_masks_disjoint_retention_cutoff(helper_make_rect):
    """
    Verifies the 50% retention cutoff rule:
    Candidates retaining < 50% of original area after subtracting
    occupied canvas are discarded entirely.
    """
    # m1: rows 0..100, cols 0..100 (10,000 px, score 0.9)
    m1 = helper_make_rect(0, 0, 100, 100)
    # m2: rows 40..100, cols 0..100 (6000 px, score 0.7)
    # Overlap with m1: rows 40..100, cols 0..100 (6000 px -> 100% overlap)
    # Let m2 extend to rows 40..140 (100 rows * 100 cols = 10,000 px)
    # Overlap = 60 rows * 100 cols = 6,000 px (60% overlap, 40% clean retention < 50%)
    m2 = helper_make_rect(40, 0, 140, 100)

    clean_masks, clean_scores = make_masks_disjoint([m1, m2], [0.90, 0.70])
    assert len(clean_masks) == 1
    assert clean_scores == [0.90]
    assert int(clean_masks[0].sum()) == 10000


def test_make_masks_disjoint_min_area_cutoff(helper_make_rect):
    """
    Verifies the 10 px minimum area rule:
    Candidates with clean area < 10 px are discarded even if retention >= 50%.
    """
    # m1: large anchor (score 0.9)
    m1 = helper_make_rect(0, 0, 50, 50)
    # m2: tiny mask with 16 px total (rows 46..50, cols 0..4 = 16 px)
    # Overlap with m1: 8 px. Clean area: 8 px (50% retention, but 8 < 10 px)
    m2 = helper_make_rect(48, 0, 52, 4)  # rows 48..52 (4 rows) x cols 0..4 (4 cols) = 16 px
    # Overlap: rows 48..50 (2 rows * 4 cols = 8 px)
    # Clean: rows 50..52 (2 rows * 4 cols = 8 px) -> 8 px < 10 px cutoff!

    clean_masks, clean_scores = make_masks_disjoint([m1, m2], [0.90, 0.70])
    assert len(clean_masks) == 1
    assert clean_scores == [0.90]


def test_make_masks_disjoint_nested_concentric_inner_higher(helper_make_rect):
    """
    Verifies concentric shapes when inner mask has higher score:
    - Inner mask preserved intact.
    - Outer mask has center punched out (donut), retained if clean area >= 50%.
    """
    # Outer: rows 20..80, cols 20..80 (60x60 = 3600 px)
    outer = helper_make_rect(20, 20, 80, 80)
    # Inner: rows 40..60, cols 40..60 (20x20 = 400 px)
    inner = helper_make_rect(40, 40, 60, 60)

    clean_masks, clean_scores = make_masks_disjoint([outer, inner], [0.60, 0.95])
    assert len(clean_masks) == 2
    assert clean_scores == [0.95, 0.60]

    # Inner is first (score 0.95)
    assert int(clean_masks[0].sum()) == 400
    # Outer is second: clean area = 3600 - 400 = 3200 px (3200 / 3600 = 88.9% retention)
    assert int(clean_masks[1].sum()) == 3200

    # Ensure hole is punched: inner area in outer mask must be all False (0)
    donut = clean_masks[1]
    assert np.all(donut[40:60, 40:60] == 0)

    # Zero overlap invariant
    assert int((clean_masks[0] & clean_masks[1]).sum()) == 0


def test_make_masks_disjoint_nested_concentric_outer_higher(helper_make_rect):
    """
    Verifies concentric shapes when outer mask has higher score:
    - Outer mask preserved intact (3600 px).
    - Inner mask is completely occluded (clean area = 0) and discarded.
    """
    outer = helper_make_rect(20, 20, 80, 80) # 3600 px
    inner = helper_make_rect(40, 40, 60, 60) # 400 px

    clean_masks, clean_scores = make_masks_disjoint([outer, inner], [0.95, 0.60])
    assert len(clean_masks) == 1
    assert clean_scores == [0.95]
    assert int(clean_masks[0].sum()) == 3600


def test_make_masks_disjoint_empty_inputs():
    """Verifies that empty mask lists or zero-area masks return empty lists cleanly."""
    clean_masks, clean_scores = make_masks_disjoint([], [])
    assert clean_masks == []
    assert clean_scores == []

    all_zero_mask = np.zeros((2048, 2048), dtype=bool)
    clean_masks_z, clean_scores_z = make_masks_disjoint([all_zero_mask], [0.9])
    assert clean_masks_z == []
    assert clean_scores_z == []


def test_postprocess_predictions_csv(helper_make_rect):
    """
    Verifies end-to-end postprocess_predictions on a DataFrame:
    - Resolves overlapping masks into disjoint masks for stemA.
    - Drops sub-threshold masks (< 10 px) for stemB and replaces with '<stemB>_1, PPPP4'.
    """
    # stemA: 2 overlapping masks (score 0.90 and 0.80)
    mA1 = helper_make_rect(10, 10, 50, 50) # 1600 px
    mA2 = helper_make_rect(30, 10, 70, 50) # 1600 px (800 clean)
    rle_A1 = rle_encode_mask(mA1)
    rle_A2 = rle_encode_mask(mA2)

    # stemB: 1 mask with area 5 px (< 10 px min_area)
    mB = np.zeros((2048, 2048), dtype=np.uint8)
    mB[10:15, 10] = 1 # 5 px
    rle_B = rle_encode_mask(mB)

    in_df = pd.DataFrame([
        {"filament_id": "stemA_1", "segmentation_rle": rle_A1, "score": 0.90},
        {"filament_id": "stemA_2", "segmentation_rle": rle_A2, "score": 0.80},
        {"filament_id": "stemB_1", "segmentation_rle": rle_B, "score": 0.75},
    ])

    out_df = postprocess_predictions(in_df, min_area=10, area_retention_ratio=0.5)

    # stemA should have 2 kept rows
    rows_A = out_df[out_df["filament_id"].str.startswith("stemA_")]
    assert len(rows_A) == 2
    assert list(rows_A["filament_id"]) == ["stemA_1", "stemA_2"]

    # Decode stemA masks and verify zero overlap
    rle_obj1 = {"size": [2048, 2048], "counts": rows_A.iloc[0]["segmentation_rle"].encode("ascii")}
    rle_obj2 = {"size": [2048, 2048], "counts": rows_A.iloc[1]["segmentation_rle"].encode("ascii")}
    clean_mA1 = mu.decode(rle_obj1)
    clean_mA2 = mu.decode(rle_obj2)
    assert int((clean_mA1 & clean_mA2).sum()) == 0

    # stemB should have exactly 1 empty placeholder row
    rows_B = out_df[out_df["filament_id"].str.startswith("stemB_")]
    assert len(rows_B) == 1
    assert rows_B.iloc[0]["filament_id"] == "stemB_1"
    assert rows_B.iloc[0]["segmentation_rle"] == EMPTY_RLE


def test_zero_overlap_invariant_assertion(helper_make_rect):
    """
    Verifies that the strict mathematical invariant max(sum(M_i)) <= 1
    is maintained across 10 randomly overlapping geometric shapes.
    """
    np.random.seed(42)
    masks = []
    scores = []
    for _ in range(5):
        y1 = np.random.randint(100, 500)
        x1 = np.random.randint(100, 500)
        y2 = y1 + np.random.randint(20, 80)
        x2 = x1 + np.random.randint(20, 80)
        masks.append(helper_make_rect(y1, x1, y2, x2))
        scores.append(float(np.random.uniform(0.5, 0.99)))

    clean_masks, clean_scores = make_masks_disjoint(masks, scores)
    assert len(clean_masks) > 0

    stack = np.stack(clean_masks, axis=0)
    pixel_counts = stack.sum(axis=0)
    assert int(pixel_counts.max()) <= 1, f"Overlap invariant violated! max overlap = {pixel_counts.max()}"
