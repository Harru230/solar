"""
tests/test_pq.py - Comprehensive Unit Tests for Panoptic Quality (PQ) Metric Engine.

Covers:
1. Analytical synthetic equivalence test (TP=2, FP=2, FN=2, SQ=0.8, RQ=0.5, PQ=0.4).
2. Bit-for-bit equivalence against pq_eval.score and pq_eval.pq_from_matrices.
3. Edge cases: empty predictions, empty GT, both empty, perfect matches (PQ=1.0), zero matches (PQ=0.0).
4. Multi-annotator pooling and independent accumulation.
5. Strict mathematical identity: PQ = SQ * RQ.
6. Threshold boundary conditions at IoU = 0.50.
7. Multi-hit diagnostics (gt_multi_hit, pred_multi_hit).
"""
import numpy as np
import pandas as pd
import pytest

import pq_eval
from pq_eval import iou_matrix, pq_from_matrices, score


def test_pq_score_synthetic_analytical_equivalence(synthetic_dataset_pair):
    """
    Verifies that pq_eval.score produces exact analytical values on the synthetic dataset:
      - 2 observations (imgA with 2 annotators, imgB with 1 annotator)
      - 3 GT filaments, 2 predicted filaments
      - Analytical expectation:
          TP=2, FP=2, FN=2, Total IoU=1.6000, Denom=4.0000
          SQ=0.8000, RQ=0.5000, PQ=0.4000
          gt_multi_hit=0, pred_multi_hit=0, annotator_images=3, images_without_prediction=1
    """
    gt_df, pred_df = synthetic_dataset_pair
    result = score(gt_df, pred_df)

    assert result["tp"] == 2, f"Expected TP=2, got {result['tp']}"
    assert result["fp"] == 2, f"Expected FP=2, got {result['fp']}"
    assert result["fn"] == 2, f"Expected FN=2, got {result['fn']}"
    assert np.isclose(result["sq"], 0.8000, atol=1e-6), f"Expected SQ=0.8000, got {result['sq']}"
    assert np.isclose(result["rq"], 0.5000, atol=1e-6), f"Expected RQ=0.5000, got {result['rq']}"
    assert np.isclose(result["pq"], 0.4000, atol=1e-6), f"Expected PQ=0.4000, got {result['pq']}"
    assert result["gt_multi_hit"] == 0, f"Expected gt_multi_hit=0, got {result['gt_multi_hit']}"
    assert result["pred_multi_hit"] == 0, f"Expected pred_multi_hit=0, got {result['pred_multi_hit']}"
    assert result["annotator_images"] == 3, f"Expected 3 annotator images, got {result['annotator_images']}"
    assert result["images_without_prediction"] == 1, f"Expected 1 image without prediction, got {result['images_without_prediction']}"


def test_bit_for_bit_score_vs_matrices(synthetic_dataset_pair):
    """
    Verifies bit-for-bit equivalence between pq_eval.score and manual pq_from_matrices.
    """
    gt_df, pred_df = synthetic_dataset_pair

    # Manual evaluation matching score() logic
    m1 = iou_matrix(
        gt_df[gt_df["filament_id"].str.startswith("ann1-imgA")]["segmentation_rle"].tolist(),
        pred_df[pred_df["filament_id"].str.startswith("imgA")]["segmentation_rle"].tolist(),
    )
    m2 = iou_matrix(
        gt_df[gt_df["filament_id"].str.startswith("ann2-imgA")]["segmentation_rle"].tolist(),
        pred_df[pred_df["filament_id"].str.startswith("imgA")]["segmentation_rle"].tolist(),
    )
    m3 = iou_matrix(
        gt_df[gt_df["filament_id"].str.startswith("ann1-imgB")]["segmentation_rle"].tolist(),
        pred_df[pred_df["filament_id"].str.startswith("imgB")]["segmentation_rle"].tolist(),
    )

    matrices_res = pq_from_matrices([m1, m2, m3])
    score_res = score(gt_df, pred_df)

    for metric in ["pq", "sq", "rq", "tp", "fp", "fn", "gt_multi_hit", "pred_multi_hit"]:
        assert matrices_res[metric] == score_res[metric], (
            f"Bit-for-bit mismatch on {metric}: matrices={matrices_res[metric]} vs score={score_res[metric]}"
        )


def test_empty_predictions():
    """Test with GT present but no predictions (FN == N_gt, TP == 0, FP == 0, PQ == 0.0)."""
    m = np.zeros((3, 0))
    res = pq_from_matrices([m])
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 3
    assert res["pq"] == 0.0
    assert res["sq"] == 0.0
    assert res["rq"] == 0.0


def test_empty_ground_truth():
    """Test with predictions present but no GT (FP == N_pred, TP == 0, FN == 0, PQ == 0.0)."""
    m = np.zeros((0, 4))
    res = pq_from_matrices([m])
    assert res["tp"] == 0
    assert res["fp"] == 4
    assert res["fn"] == 0
    assert res["pq"] == 0.0
    assert res["sq"] == 0.0
    assert res["rq"] == 0.0


def test_both_empty():
    """Test with zero GT and zero predictions (all zeros)."""
    m = np.zeros((0, 0))
    res = pq_from_matrices([m])
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["pq"] == 0.0
    assert res["sq"] == 0.0
    assert res["rq"] == 0.0


def test_perfect_matches():
    """Test perfect matches where all predictions match GT with IoU = 1.0 (PQ == 1.0)."""
    m = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    res = pq_from_matrices([m])
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert np.isclose(res["sq"], 1.0)
    assert np.isclose(res["rq"], 1.0)
    assert np.isclose(res["pq"], 1.0)


def test_zero_matches():
    """Test non-empty GT and predictions where all pairwise IoU <= 0.50 (PQ == 0.0)."""
    m = np.array([
        [0.45, 0.20],
        [0.10, 0.50],  # Exactly 0.50 should NOT count as hit (strict > 0.50 threshold)
    ])
    res = pq_from_matrices([m])
    assert res["tp"] == 0
    assert res["fp"] == 2
    assert res["fn"] == 2
    assert res["sq"] == 0.0
    assert res["rq"] == 0.0
    assert res["pq"] == 0.0


def test_strict_threshold_boundary():
    """
    Test strict inequality > 0.50 threshold in pq_eval:
      - IoU = 0.500000 is NOT a hit.
      - IoU = 0.500001 IS a hit.
    """
    # 0.500000 -> No hit
    m_no_hit = np.array([[0.500000]])
    res_no_hit = pq_from_matrices([m_no_hit])
    assert res_no_hit["tp"] == 0
    assert res_no_hit["fp"] == 1
    assert res_no_hit["fn"] == 1

    # 0.500001 -> Hit
    m_hit = np.array([[0.500001]])
    res_hit = pq_from_matrices([m_hit])
    assert res_hit["tp"] == 1
    assert res_hit["fp"] == 0
    assert res_hit["fn"] == 0
    assert np.isclose(res_hit["pq"], 0.500001)


def test_pq_decomposition_identity():
    """
    Verifies that the fundamental mathematical identity PQ = SQ * RQ holds
    across various random and non-trivial IoU matrices whenever TP > 0.
    """
    test_matrices = [
        np.array([[0.8, 0.1], [0.2, 0.3]]),
        np.array([[0.6, 0.7]]),
        np.array([[0.9, 0.1, 0.0], [0.1, 0.85, 0.2], [0.0, 0.0, 0.4]]),
        np.array([[0.75, 0.0], [0.0, 0.65], [0.0, 0.0]]),
    ]
    for m in test_matrices:
        res = pq_from_matrices([m])
        if res["tp"] > 0:
            expected_pq = res["sq"] * res["rq"]
            assert np.isclose(res["pq"], expected_pq, atol=1e-12), (
                f"PQ != SQ * RQ failed for matrix:\n{m}\nres={res}"
            )


def test_multi_annotator_independent_pooling():
    """
    Verifies that multiple annotators on the same image are evaluated independently
    and pooled into the global denominator and numerator.
    """
    # Observation 1: 2 annotators against the same prediction
    # Ann 1 gives IoU = 0.8 (TP)
    # Ann 2 gives IoU = 0.4 (FN, FP)
    m_ann1 = np.array([[0.8]])
    m_ann2 = np.array([[0.4]])

    res = pq_from_matrices([m_ann1, m_ann2])
    # Ann 1: TP=1 (iou=0.8), FP=0, FN=0
    # Ann 2: TP=0, FP=1, FN=1
    # Global: TP=1, FP=1, FN=1, Denom = 1 + 0.5*1 + 0.5*1 = 2.0
    # SQ = 0.8 / 1 = 0.8
    # RQ = 1 / 2.0 = 0.5
    # PQ = 0.8 / 2.0 = 0.4
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert np.isclose(res["pq"], 0.4)
    assert np.isclose(res["sq"], 0.8)
    assert np.isclose(res["rq"], 0.5)


def test_duplicate_hits_diagnostics():
    """
    Verifies gt_multi_hit and pred_multi_hit counters:
      - 2 predictions hitting 1 GT with IoU > 0.5 -> gt_multi_hit = 1
      - 1 prediction hitting 2 GT with IoU > 0.5 -> pred_multi_hit = 1
    """
    # gt_multi_hit: 1 GT, 2 predictions both > 0.5
    m_gt_multi = np.array([[0.6, 0.7]])
    res1 = pq_from_matrices([m_gt_multi])
    assert res1["gt_multi_hit"] == 1
    assert res1["pred_multi_hit"] == 0
    assert res1["tp"] == 2

    # pred_multi_hit: 2 GT, 1 prediction > 0.5 for both
    m_pred_multi = np.array([[0.6], [0.7]])
    res2 = pq_from_matrices([m_pred_multi])
    assert res2["gt_multi_hit"] == 0
    assert res2["pred_multi_hit"] == 1
    assert res2["tp"] == 2
