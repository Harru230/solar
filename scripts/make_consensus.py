#!/usr/bin/env python3
"""
scripts/make_consensus.py - Multi-annotator mask consensus via Signed Distance Transforms.

Algorithm:
1. Clusters masks of different annotators of the same observation by IoU > 0.25.
2. For each cluster of m masks across K total annotators for that observation:
   - Support weight: w = m / K.
   - Filter: keeps clusters with w >= 0.5 (discards low-agreement clusters).
3. For kept clusters:
   - If m == 1: returns the single mask.
   - If m >= 2:
     - Computes bounding box of the union + 16 px margin.
     - Computes Signed Distance Transform (EDT on crop + 16 px margin):
         SDT = EDT(mask) - EDT(~mask)
     - Averages SDTs across all m masks in the cluster: SDT_avg = mean(SDTs).
     - Thresholds at SDT_avg > 0 to obtain the consensus boundary.
     - If consensus area < 0.7 * min(input_areas): fallback to the medoid mask
       (the input mask with the highest sum of IoUs with other cluster members).
4. Reports counts per support level and medoid fallback frequency.
5. Provides unit tests and integration with train_maskrcnn.py (--target consensus).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import find_dataset_paths

FRAME_SIZE = 2048


def rasterize_polygon(poly_list: List, h: int = FRAME_SIZE, w: int = FRAME_SIZE) -> np.ndarray:
    """Rasterizes a list of polygon coordinate lists into a binary uint8 mask of shape (h, w)."""
    m = np.zeros((h, w), dtype=np.uint8)
    for poly in poly_list:
        if isinstance(poly, list) and len(poly) >= 6:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def compute_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Computes Intersection over Union between two binary masks."""
    inter = int(np.logical_and(m1, m2).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(m1, m2).sum())
    return inter / union if union > 0 else 0.0


def compute_medoid_mask(masks: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Finds the medoid mask among a list of masks (the mask maximizing sum of IoUs to all others).
    Returns (medoid_mask, medoid_index).
    """
    m = len(masks)
    if m == 1:
        return masks[0], 0
    if m == 2:
        # Both have same IoU to each other; return the one with larger area
        areas = [int(mask.sum()) for mask in masks]
        best_idx = 0 if areas[0] >= areas[1] else 1
        return masks[best_idx], best_idx

    iou_sums = np.zeros(m, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            score = compute_mask_iou(masks[i], masks[j])
            iou_sums[i] += score
            iou_sums[j] += score

    best_idx = int(np.argmax(iou_sums))
    return masks[best_idx], best_idx


def fuse_cluster_sdt(
    masks: List[np.ndarray],
    margin: int = 16,
    area_ratio_thresh: float = 0.7,
) -> Tuple[np.ndarray, bool, float]:
    """
    Fuses m >= 2 binary masks using Signed Distance Transform (EDT) within union bbox + margin.
    If fused area < area_ratio_thresh * min(input_areas), falls back to the medoid mask.

    Returns:
      (fused_mask, fallback_fired, area_ratio)
    """
    m = len(masks)
    if m == 1:
        return masks[0].copy(), False, 1.0

    input_areas = [int(mask.sum()) for mask in masks]
    min_input_area = min(input_areas)
    h, w = masks[0].shape

    # Union mask bounding box
    union_mask = np.logical_or.reduce(masks)
    y_idx, x_idx = np.where(union_mask)
    if len(y_idx) == 0:
        return np.zeros((h, w), dtype=np.uint8), False, 0.0

    y0 = max(0, int(y_idx.min()) - margin)
    y1 = min(h, int(y_idx.max()) + margin + 1)
    x0 = max(0, int(x_idx.min()) - margin)
    x1 = min(w, int(x_idx.max()) + margin + 1)

    crops = [mask[y0:y1, x0:x1] for mask in masks]

    # Compute Signed Distance Transform on crops
    # Inside: pos > 0, Outside: -neg < 0
    sdts = []
    for crop in crops:
        b = crop.astype(bool)
        pos = distance_transform_edt(b)
        neg = distance_transform_edt(~b)
        sdts.append(pos - neg)

    sdt_avg = np.mean(sdts, axis=0)
    crop_fused = (sdt_avg > 0).astype(np.uint8)
    fused_area = int(crop_fused.sum())

    area_ratio = fused_area / max(1, min_input_area)

    # Check area fallback threshold
    if fused_area < area_ratio_thresh * min_input_area:
        medoid, _ = compute_medoid_mask(masks)
        return medoid.copy(), True, area_ratio

    # Paste fused crop onto full-size canvas
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = crop_fused
    return full_mask, False, area_ratio


def cluster_and_fuse_observation(
    masks: List[np.ndarray],
    annotator_ids: List[str],
    total_annotators_K: int,
    iou_thresh: float = 0.25,
    min_support: float = 0.5,
    margin: int = 16,
    area_ratio_thresh: float = 0.7,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Performs clustering and consensus fusion for all masks of one observation.

    Returns:
      (kept_consensus_masks, cluster_metadata_list)
    """
    n = len(masks)
    if n == 0:
        return [], []

    K = max(1, total_annotators_K)

    # If only 1 mask total
    if n == 1:
        w = 1.0 / K
        meta = {
            "m": 1,
            "K": K,
            "support": w,
            "kept": w >= min_support,
            "fallback": False,
            "area_ratio": 1.0,
            "cluster_size": 1,
        }
        if w >= min_support:
            return [masks[0].copy()], [meta]
        else:
            return [], [meta]

    # Build adjacency matrix between masks of DIFFERENT annotators with IoU > iou_thresh
    areas = [int(m.sum()) for m in masks]
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        if areas[i] == 0:
            continue
        for j in range(i + 1, n):
            if areas[j] == 0:
                continue
            if annotator_ids[i] == annotator_ids[j]:
                continue  # Only connect masks from different annotators
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            if inter > 0:
                union = areas[i] + areas[j] - inter
                if union > 0 and (inter / union) > iou_thresh:
                    adj[i, j] = adj[j, i] = True

    n_comp, labels = connected_components(adj)

    kept_masks = []
    meta_list = []

    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        c_masks = [masks[i] for i in idx if areas[i] > 0]
        if not c_masks:
            continue

        c_owners = {annotator_ids[i] for i in idx}
        m = len(c_owners)
        w = m / K
        kept = (w >= min_support)

        meta = {
            "cluster_id": c,
            "m": m,
            "K": K,
            "support": round(w, 4),
            "kept": kept,
            "cluster_size": len(c_masks),
            "fallback": False,
            "area_ratio": 1.0,
        }

        if kept:
            if len(c_masks) == 1:
                kept_masks.append(c_masks[0].copy())
            else:
                fused_mask, fallback, ratio = fuse_cluster_sdt(
                    masks=c_masks,
                    margin=margin,
                    area_ratio_thresh=area_ratio_thresh,
                )
                meta["fallback"] = fallback
                meta["area_ratio"] = round(ratio, 4)
                kept_masks.append(fused_mask)

        meta_list.append(meta)

    return kept_masks, meta_list


def run_unit_tests():
    """Unit tests for consensus clustering and SDT fusion."""
    print("\n==================================================================")
    print("           RUNNING UNIT TESTS FOR CONSENSUS FUSION                ")
    print("==================================================================")

    # Test 1: Identical masks from 2 annotators (K=2)
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[30:70, 30:70] = 1
    m2 = m1.copy()
    fused, fallback, ratio = fuse_cluster_sdt([m1, m2], margin=16, area_ratio_thresh=0.7)
    assert not fallback, "Unit test 1 failed: fallback fired on identical masks!"
    assert np.array_equal(fused, m1), "Unit test 1 failed: fused mask not equal to input!"
    print("[PASS] Unit Test 1: Identical masks fusion (ratio=1.0, fallback=False)")

    # Test 2: Overlapping masks with IoU > 0.5 (K=2)
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[35:75, 35:75] = 1
    fused, fallback, ratio = fuse_cluster_sdt([m1, m2], margin=16, area_ratio_thresh=0.7)
    assert not fallback, f"Unit test 2 failed: fallback fired unexpectedly (ratio={ratio})!"
    assert fused.sum() > 0.7 * min(m1.sum(), m2.sum()), "Unit test 2 failed: area too small!"
    print(f"[PASS] Unit Test 2: Smooth consensus on offset masks (ratio={ratio:.3f}, fallback=False)")

    # Test 3: Synthetic thin disjoint strips triggering area erosion fallback
    m_thin1 = np.zeros((100, 100), dtype=np.uint8)
    m_thin1[50:52, 20:80] = 1  # 2x60 strip
    m_thin2 = np.zeros((100, 100), dtype=np.uint8)
    m_thin2[51:53, 20:80] = 1  # 2x60 strip offset by 1 px
    m_thin3 = np.zeros((100, 100), dtype=np.uint8)
    m_thin3[53:55, 20:80] = 1  # separated strip
    fused, fallback, ratio = fuse_cluster_sdt([m_thin1, m_thin2, m_thin3], margin=16, area_ratio_thresh=0.7)
    assert fallback, "Unit test 3 failed: fallback should have fired on severely eroded strip!"
    print(f"[PASS] Unit Test 3: Medoid fallback triggered correctly when area < 0.7*min (ratio={ratio:.3f}, fallback=True)")

    # Test 4: Support thresholding with K=3
    # 3 annotators: ann1, ann2, ann3
    # ann1 has mask A, ann2 has mask B (IoU > 0.25) -> cluster 1 (m=2, w=2/3 >= 0.5 -> kept)
    # ann3 has mask C far away -> cluster 2 (m=1, w=1/3 < 0.5 -> dropped)
    mA = np.zeros((100, 100), dtype=np.uint8); mA[20:40, 20:40] = 1
    mB = np.zeros((100, 100), dtype=np.uint8); mB[22:42, 22:42] = 1
    mC = np.zeros((100, 100), dtype=np.uint8); mC[70:90, 70:90] = 1

    kept, metas = cluster_and_fuse_observation(
        masks=[mA, mB, mC],
        annotator_ids=["ann1", "ann2", "ann3"],
        total_annotators_K=3,
        iou_thresh=0.25,
        min_support=0.5,
    )
    assert len(kept) == 1, f"Unit test 4 failed: expected 1 kept mask, got {len(kept)}"
    assert metas[0]["kept"] and metas[0]["support"] == round(2/3, 4), "Cluster 1 should be kept with w=0.6667"
    assert not metas[1]["kept"] and metas[1]["support"] == round(1/3, 4), "Cluster 2 should be dropped with w=0.3333"
    print(f"[PASS] Unit Test 4: Support filtering (w=2/3 kept, w=1/3 dropped)")

    print("All unit tests passed successfully!")
    print("==================================================================\n")


def compute_dataset_consensus_stats(json_path: str):
    """Computes consensus statistics over the entire dataset."""
    print("==================================================================")
    print("           COMPUTING CONSENSUS DATASET STATISTICS                 ")
    print(f"JSON: {json_path}")
    print("==================================================================")

    with open(json_path) as f:
        coco = json.load(f)

    obs_annotators = defaultdict(set)
    for im in coco["images"]:
        ann_id, obs = im["id"].split("-", 1)
        obs_annotators[obs].add(ann_id)

    obs_to_anns = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        ann_id, obs = ann["image_id"].split("-", 1)
        obs_to_anns[obs][ann_id].append(ann)

    support_counter = Counter()
    kept_support_counter = Counter()
    dropped_support_counter = Counter()
    total_clusters = 0
    total_fused_m_ge_2 = 0
    fallback_fired_count = 0

    t0 = time.time()
    num_obs = len(obs_annotators)

    for obs, annotators in obs_annotators.items():
        K = len(annotators)
        all_masks = []
        mask_owners = []

        for ann in annotators:
            anns_list = obs_to_anns[obs].get(ann, [])
            for a in anns_list:
                m = rasterize_polygon(a["segmentation"])
                if m.sum() > 0:
                    all_masks.append(m)
                    mask_owners.append(ann)

        if not all_masks:
            continue

        if K == 1:
            for _ in all_masks:
                support_counter[1.0] += 1
                kept_support_counter[1.0] += 1
                total_clusters += 1
            continue

        kept_masks, metas = cluster_and_fuse_observation(
            masks=all_masks,
            annotator_ids=mask_owners,
            total_annotators_K=K,
            iou_thresh=0.25,
            min_support=0.5,
            margin=16,
            area_ratio_thresh=0.7,
        )

        for meta in metas:
            total_clusters += 1
            w = meta["support"]
            support_counter[w] += 1
            if meta["kept"]:
                kept_support_counter[w] += 1
                if meta["cluster_size"] >= 2:
                    total_fused_m_ge_2 += 1
                    if meta["fallback"]:
                        fallback_fired_count += 1
            else:
                dropped_support_counter[w] += 1

    elapsed = time.time() - t0
    print(f"Processed {num_obs} observations in {elapsed:.1f}s.")
    print("------------------------------------------------------------------")
    print(f"Total Clusters:           {total_clusters}")
    print(f"Kept Clusters (w >= 0.5): {sum(kept_support_counter.values())}")
    print(f"Dropped Clusters (w < 0.5):{sum(dropped_support_counter.values())}")
    print("------------------------------------------------------------------")
    print("Counts per Support Level w = m/K:")
    for w in sorted(support_counter.keys()):
        status = "KEPT" if w >= 0.5 else "DROPPED"
        print(f"  w = {w:<6.4f}: {support_counter[w]:>5d} clusters  ({status})")
    print("------------------------------------------------------------------")
    print(f"Clusters fused with m >= 2:     {total_fused_m_ge_2}")
    fallback_pct = (fallback_fired_count / max(1, total_fused_m_ge_2)) * 100
    print(f"Medoid Fallback Fired:          {fallback_fired_count} / {total_fused_m_ge_2} ({fallback_pct:.2f}%)")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Make consensus filament masks")
    parser.add_argument("--json", type=str, default=None, help="Path to train JSON annotations")
    parser.add_argument("--unit_test", action="store_true", help="Run unit tests")
    parser.add_argument("--stats", action="store_true", help="Compute consensus statistics on dataset")
    args = parser.parse_args()

    if args.unit_test:
        run_unit_tests()

    paths = find_dataset_paths()
    json_path = args.json or paths["train_json"]

    if args.stats or (not args.unit_test and not args.stats):
        if not json_path or not os.path.exists(json_path):
            print(f"Error: JSON not found at {json_path}")
            sys.exit(1)
        compute_dataset_consensus_stats(json_path)


if __name__ == "__main__":
    main()
