"""
tests/conftest.py - Pytest fixtures and synthetic data generators
for the Solar Filament Segmentation test suite.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mu

FRAME_SIZE = 2048
EMPTY_RLE = "PPPP4"


@pytest.fixture
def canvas_size() -> Tuple[int, int]:
    """Returns standard canvas dimensions (2048, 2048)."""
    return (FRAME_SIZE, FRAME_SIZE)


def make_rect_mask(
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    h: int = FRAME_SIZE,
    w: int = FRAME_SIZE,
) -> np.ndarray:
    """Helper to generate a rectangular boolean mask on a canvas of shape (h, w)."""
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def encode_rle(mask: np.ndarray) -> str:
    """Helper to encode a binary numpy mask into a COCO compressed RLE ASCII string."""
    fortran_arr = np.asfortranarray(mask.astype(np.uint8))
    encoded = mu.encode(fortran_arr)
    return encoded["counts"].decode("ascii")


def decode_rle(rle_str: str, shape: Tuple[int, int] = (FRAME_SIZE, FRAME_SIZE)) -> np.ndarray:
    """Helper to decode a COCO compressed RLE ASCII string into a binary numpy mask."""
    rle_obj = {"size": list(shape), "counts": rle_str.encode("ascii")}
    return mu.decode(rle_obj)


@pytest.fixture
def helper_make_rect():
    """Provides make_rect_mask as a fixture."""
    return make_rect_mask


@pytest.fixture
def helper_encode_rle():
    """Provides encode_rle as a fixture."""
    return encode_rle


@pytest.fixture
def helper_decode_rle():
    """Provides decode_rle as a fixture."""
    return decode_rle


@pytest.fixture(scope="session")
def dummy_stems_180() -> List[str]:
    """Provides exactly 180 unique image stem names."""
    return [f"test_{i:03d}" for i in range(180)]


@pytest.fixture(scope="session")
def dummy_test_dir(tmp_path_factory, dummy_stems_180: List[str]) -> Path:
    """
    Creates a hermetic session-scoped temporary test directory containing 180 dummy JPEG files
    matching the expected competition test stems.
    """
    test_dir = tmp_path_factory.mktemp("test_images")
    for stem in dummy_stems_180:
        (test_dir / f"{stem}.jpeg").touch()
    return test_dir


@pytest.fixture(scope="session")
def synthetic_dataset_pair() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the analytical synthetic ground-truth and prediction DataFrames
    matching the exact specification from survey_eval_tests.md:
      - 2 observations: imgA and imgB
      - 2 annotators: ann1 and ann2
      - 3 GT filaments across imgA and imgB
      - 2 predicted filaments for imgA, 0 for imgB
      - Exact metrics:
          TP = 2
          FP = 2
          FN = 2
          Total IoU = 1.6000
          Denominator = 4.0000
          SQ = 0.8000
          RQ = 0.5000
          PQ = 0.4000
          gt_multi_hit = 0
          pred_multi_hit = 0
          annotator_images = 3
          images_without_prediction = 1
    """
    # GT filaments
    # ann1-imgA: 2 filaments
    gt_ann1_imgA_1 = encode_rle(make_rect_mask(100, 100, 200, 200))  # 10,000 px
    gt_ann1_imgA_2 = encode_rle(make_rect_mask(500, 500, 600, 600))  # 10,000 px
    # ann2-imgA: 1 filament identical to GT1
    gt_ann2_imgA_1 = encode_rle(make_rect_mask(100, 100, 200, 200))  # 10,000 px
    # ann1-imgB: 1 filament
    gt_ann1_imgB_1 = encode_rle(make_rect_mask(300, 300, 400, 400))  # 10,000 px

    gt_df = pd.DataFrame([
        {"filament_id": "ann1-imgA_1", "segmentation_rle": gt_ann1_imgA_1},
        {"filament_id": "ann1-imgA_2", "segmentation_rle": gt_ann1_imgA_2},
        {"filament_id": "ann2-imgA_1", "segmentation_rle": gt_ann2_imgA_1},
        {"filament_id": "ann1-imgB_1", "segmentation_rle": gt_ann1_imgB_1},
    ])

    # Predictions
    # imgA: Pred1 overlaps GT1 with IoU = 0.80 (8,000 / 10,000)
    pred_imgA_1 = encode_rle(make_rect_mask(100, 100, 200, 180))     # 8,000 px
    # imgA: Pred2 overlaps nothing (pure FP)
    pred_imgA_2 = encode_rle(make_rect_mask(1000, 1000, 1100, 1100)) # 10,000 px

    pred_df = pd.DataFrame([
        {"filament_id": "imgA_1", "segmentation_rle": pred_imgA_1},
        {"filament_id": "imgA_2", "segmentation_rle": pred_imgA_2},
    ])

    return gt_df, pred_df


@pytest.fixture(scope="session")
def valid_submission_df(dummy_stems_180: List[str]) -> pd.DataFrame:
    """
    Creates a fully compliant submission DataFrame covering all 180 stems:
      - Stem 0 has 2 disjoint filaments.
      - Stem 1 has 1 filament.
      - Stems 2..179 have 1 empty placeholder row ('<stem>_1', 'PPPP4').
    """
    m1 = encode_rle(make_rect_mask(50, 50, 100, 100))
    m2 = encode_rle(make_rect_mask(200, 200, 300, 300))
    m3 = encode_rle(make_rect_mask(150, 150, 250, 250))

    rows = [
        {"filament_id": f"{dummy_stems_180[0]}_1", "segmentation_rle": m1},
        {"filament_id": f"{dummy_stems_180[0]}_2", "segmentation_rle": m2},
        {"filament_id": f"{dummy_stems_180[1]}_1", "segmentation_rle": m3},
    ]
    for stem in dummy_stems_180[2:]:
        rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})

    return pd.DataFrame(rows)
