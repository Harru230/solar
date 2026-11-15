#!/usr/bin/env python3
"""
audit/run_error_decomposition.py - CPU-only error decomposition of the 5-family OOF WMF ensemble
(clahe, a5, dice, a5dice, 2048; 20 views) at (tau_score, tau_mask) = (0.85, 0.55), using the same
canvas painting rule as multifamily_pipeline.evaluate_threshold_grid_nested_cv (min_area=10,
area_retention=0.50) and the same GT (build_gt_rles_by_obs, per annotator record).

For every annotator record of every image:
  FN categories (first match wins):
    lost_in_canvas : a cluster with score >= tau_score has independent IoU > 0.5 with the GT,
                     but after disjoint-canvas painting nothing matches it
    below_tau      : a cluster with IoU > 0.5 exists, but its score < tau_score (ranking / threshold)
    seg_near_miss  : best candidate IoU in (0.3, 0.5]
    partial        : best candidate IoU in (0, 0.3]
    no_candidate   : no candidate overlaps the GT at all (score >= MIN_CLUSTER_SCORE)
  FP categories (first match wins):
    other_annotator: matches (IoU > 0.5) a GT of a *different* annotator record of the same image
    near_miss      : best IoU to a GT of this record in (0.3, 0.5]
    partial        : best IoU in (0, 0.3]
    background     : no overlap with any GT of this record
Also: TP IoU distribution and pred/GT area ratio on TPs, FN support by other annotators,
GT-area bins, and PQ sensitivities (dPQ per FP removed / FN converted / mean-IoU change).
Processes one fold at a time to bound RAM. Writes audit/error_decomposition_5fam.json.
"""
import argparse
import sys
import json
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pycocotools import mask as mu

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLAR_ROOT = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))).resolve()
sys.path.insert(0, str(REPO_ROOT))
import pq_eval  # noqa: E402
from scripts.common import find_dataset_paths  # noqa: E402
from scripts.multifamily_pipeline import build_gt_rles_by_obs, load_fused_oof_pool, FRAME_SIZE  # noqa: E402
from scripts.instance_scorer import encode_crop_to_rle  # noqa: E402

TAU_SCORE, TAU_MASK = 0.85, 0.55
MIN_AREA, AREA_RET = 10, 0.50
MIN_CLUSTER_SCORE = 0.30
W = str(SOLAR_ROOT / "work")
SPECS = [
    ("clahe", f"{W}/oof", 4, 0.0),
    ("a5", f"{W}/oof_a5", 4, 0.0),
    ("dice", f"{W}/oof_dice", 4, 0.0),
    ("a5dice", f"{W}/oof_a5dice", 4, 0.0),
    ("2048", f"{W}/oof_2048", 4, 0.0),
]
AREA_BINS = [(0, 300), (300, 1000), (1000, 3000), (3000, 10**9)]


def rle_obj(s):
    return {"size": [FRAME_SIZE, FRAME_SIZE], "counts": s.encode("ascii")}


def area_of(s):
    return int(mu.area(rle_obj(s)))


def bin_name(a):
    for lo, hi in AREA_BINS:
        if lo <= a < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "?"


def match(m):
    """Greedy-free PQ matching: IoU > 0.5 is unique. Returns dict gt_idx -> pred_idx."""
    out = {}
    if m.size == 0:
        return out
    gi, pj = np.where(m > pq_eval.IOU_THRESHOLD)
    for g, p in zip(gi, pj):
        out[int(g)] = int(p)
    return out


def main():
    argparse.ArgumentParser(
        description="CPU error decomposition of the 5-family OOF WMF ensemble."
    ).parse_args()
    t0 = time.time()
    paths = find_dataset_paths()
    gt_df = pd.read_csv(paths["gt_all_csv"])
    folds_df = pd.read_csv(paths["folds_csv"])
    gt_by_obs = build_gt_rles_by_obs(gt_df)

    tot = Counter()
    fn_cat = Counter()
    fp_cat = Counter()
    fn_cat_by_fold = defaultdict(Counter)
    fp_cat_by_fold = defaultdict(Counter)
    fn_below_tau_scores = []
    fn_support = Counter()          # on multi-record images: supported by >=1 other annotator or not
    fn_bins = Counter()
    gt_bins = Counter()
    tp_ious, tp_area_ratio = [], []
    iou_sum = 0.0

    for f in range(5):
        fdf = folds_df[folds_df["fold"] == f]
        pool = load_fused_oof_pool(fdf, SPECS, min_cluster_score=MIN_CLUSTER_SCORE)
        for obs in fdf["observation"].tolist():
            clusters = pool.get(obs, [])
            occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            painted, cand, cand_scores = [], [], []
            for c in clusters:
                sc = float(c["score"])
                x1, y1, x2, y2 = (int(v) for v in c["box"])
                bc = c["prob_crop"] >= TAU_MASK
                oa = int(bc.sum())
                if oa < MIN_AREA:
                    continue
                cand.append(encode_crop_to_rle(c["box"], bc))
                cand_scores.append(sc)
                if sc >= TAU_SCORE:
                    clean = bc & (~occupied[y1:y2, x1:x2])
                    ca = int(clean.sum())
                    if ca < MIN_AREA or ca / float(oa) < AREA_RET:
                        continue
                    occupied[y1:y2, x1:x2] |= clean
                    painted.append(encode_crop_to_rle(c["box"], clean))
            cand_scores = np.asarray(cand_scores)
            records = gt_by_obs.get(obs, [])
            K = len(records)
            for r, gts in enumerate(records):
                m_p = pq_eval.iou_matrix(gts, painted)
                m_c = pq_eval.iou_matrix(gts, cand)
                mt = match(m_p)
                matched_preds = set(mt.values())
                tot["tp"] += len(mt)
                tot["fn"] += len(gts) - len(mt)
                tot["fp"] += len(painted) - len(matched_preds)
                for g, p in mt.items():
                    q = float(m_p[g, p])
                    iou_sum += q
                    tp_ious.append(q)
                    tp_area_ratio.append(area_of(painted[p]) / max(1, area_of(gts[g])))
                other_gts = [s for rr, gl in enumerate(records) if rr != r for s in gl]
                # ---- FN ----
                for g in range(len(gts)):
                    a = area_of(gts[g])
                    gt_bins[bin_name(a)] += 1
                    if g in mt:
                        continue
                    fn_bins[bin_name(a)] += 1
                    row = m_c[g] if m_c.size else np.zeros(0)
                    hit = row > pq_eval.IOU_THRESHOLD if row.size else np.zeros(0, bool)
                    if row.size and np.any(hit & (cand_scores >= TAU_SCORE)):
                        cat = "lost_in_canvas"
                    elif row.size and np.any(hit):
                        cat = "below_tau"
                        fn_below_tau_scores.append(float(cand_scores[hit].max()))
                    elif row.size and row.max() > 0.3:
                        cat = "seg_near_miss"
                    elif row.size and row.max() > 0.0:
                        cat = "partial"
                    else:
                        cat = "no_candidate"
                    fn_cat[cat] += 1
                    fn_cat_by_fold[f][cat] += 1
                    if K >= 2:
                        sup = pq_eval.iou_matrix([gts[g]], other_gts) if other_gts else np.zeros((1, 0))
                        n_sup = int((sup > pq_eval.IOU_THRESHOLD).sum()) if sup.size else 0
                        fn_support["supported_by_other_annotator" if n_sup > 0 else "only_this_annotator"] += 1
                    else:
                        fn_support["single_record_image"] += 1
                # ---- FP ----
                m_o = pq_eval.iou_matrix(other_gts, painted) if other_gts else np.zeros((0, len(painted)))
                for p in range(len(painted)):
                    if p in matched_preds:
                        continue
                    if m_o.size and (m_o[:, p] > pq_eval.IOU_THRESHOLD).any():
                        cat = "other_annotator"
                    else:
                        best = float(m_p[:, p].max()) if m_p.size else 0.0
                        cat = "near_miss" if best > 0.3 else ("partial" if best > 0.0 else "background")
                    fp_cat[cat] += 1
                    fp_cat_by_fold[f][cat] += 1
        del pool
        print(f"fold {f} done ({time.time() - t0:.0f}s): tp={tot['tp']} fp={tot['fp']} fn={tot['fn']}", flush=True)

    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = iou_sum / denom
    sq = iou_sum / tp
    sens = {
        "dPQ_per_FP_removed": pq * 0.5 / denom,
        "dPQ_per_FN_converted_to_TP_at_mean_SQ": (sq - 0.5 * pq) / denom,
        "dPQ_per_+0.01_mean_TP_IoU": 0.01 * tp / denom,
        "FP_removed_for_+0.001": 0.001 / (pq * 0.5 / denom),
        "FN_converted_for_+0.001": 0.001 / ((sq - 0.5 * pq) / denom),
        "mean_TP_IoU_gain_for_+0.001": 0.001 / (tp / denom),
    }
    q = np.asarray(tp_ious)
    ar = np.asarray(tp_area_ratio)
    out = {
        "config": {"tau_score": TAU_SCORE, "tau_mask": TAU_MASK, "min_area": MIN_AREA,
                   "area_retention": AREA_RET, "min_cluster_score_for_candidates": MIN_CLUSTER_SCORE,
                   "families": [s[0] for s in SPECS]},
        "totals": {"tp": tp, "fp": fp, "fn": fn, "pq": pq, "sq": sq, "rq": tp / denom},
        "fn_categories": dict(fn_cat), "fp_categories": dict(fp_cat),
        "fn_categories_by_fold": {k: dict(v) for k, v in fn_cat_by_fold.items()},
        "fp_categories_by_fold": {k: dict(v) for k, v in fp_cat_by_fold.items()},
        "fn_support": dict(fn_support),
        "fn_below_tau_best_score_quantiles": (np.quantile(fn_below_tau_scores, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
                                              if fn_below_tau_scores else []),
        "gt_area_bins": dict(gt_bins), "fn_area_bins": dict(fn_bins),
        "tp_iou_quantiles": np.quantile(q, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "tp_iou_frac_in_0.5_0.6": float(((q > 0.5) & (q <= 0.6)).mean()),
        "tp_pred_over_gt_area_quantiles": np.quantile(ar, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "pq_sensitivities": sens,
        "elapsed_sec": time.time() - t0,
    }
    out_path = REPO_ROOT / "audit" / "error_decomposition_5fam.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
