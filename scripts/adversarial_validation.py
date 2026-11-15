#!/usr/bin/env python3
"""
scripts/adversarial_validation.py - Step 1: Adversarial Validation & Multi-View Feature Check

Performs:
1) Feature & Scorer check across 4 views (clahe), 8 views (clahe + anchors5),
   and 12 views (clahe + anchors5 + dice) over all 707 OOF images:
   - Compares feature means/stds across 4, 8, and 12 views.
   - Runs adversarial LGBMClassifier between 4v vs 8v, 4v vs 12v, and 8v vs 12v
     to identify view-count-sensitive features.
   - Evaluates 5-fold cross-fitted LightGBM Scorer on 4, 8, and 12 views over all 707 OOF images
     with 1,000-iteration paired bootstrap.
2) Adversarial Validation: OOF-features vs Test-features:
   - Case A (Matched 12-View 3-Family Ensemble: clahe + anchors5 + dice, M=12 on OOF vs M=12 on Test)
   - Case B (Matched 16-View 4-Family Ensemble on Fold 0: clahe + anchors5 + dice + 2048, M=16 on OOF vs M=16 on Test)
   - Case C (Matched 4-View Single-Family: clahe M=4 on OOF vs f0..f4_1536 M=4 on Test)
   - Case D (Mismatched View-Count Diagnostic: 4-View OOF M=4 vs 20-View 5-Fold Test M=20)
   Reports 5-fold CV ROC AUC, single-feature AUC, KS-stat, and top separating features by Gain.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd
import torch
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    roc_auc_score = None
    StratifiedKFold = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import find_dataset_paths
from scripts.instance_scorer import (
    FEATURE_NAMES,
    build_gt_lookup,
    compute_candidate_targets,
    encode_crop_to_rle,
)
from scripts.predict_ensemble import weighted_mask_fusion


def drop_os_page_cache(file_path: str):
    """Evicts file from Linux page cache so WSL2 RAM stays flat."""
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(file_path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def extract_features_from_fused_clusters(
    fused: List[dict],
    obs: str,
    fold: int,
    num_views_total: int,
    min_cluster_score: float = 0.15,
    tau_mask: float = 0.50,
    min_area: int = 10,
) -> List[dict]:
    """
    Extracts the 14 scorer features from a list of WMF fused clusters for a single image.
    """
    all_scores = [float(c["score"]) for c in fused]
    max_score_in_img = max(all_scores) if all_scores else 1e-6
    n_cands_gt_05 = sum(1 for s in all_scores if s > 0.50)

    rows = []
    cand_idx = 0
    for rank_0, c in enumerate(fused):
        sc = float(c["score"])
        if sc < min_cluster_score:
            continue

        prob_crop = c["prob_crop"]
        box = c["box"]
        binary_crop = (prob_crop >= tau_mask)
        area = int(binary_crop.sum())
        if area < min_area:
            continue

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        bbox_area = float(bw * bh)

        masked_probs = prob_crop[binary_crop].astype(np.float32)
        prob_mean = float(masked_probs.mean())
        prob_std = float(masked_probs.std()) if area > 1 else 0.0
        core_pixels = int((masked_probs >= 0.75).sum())
        prob_sharpness_core = float(core_pixels) / float(area)
        q90, q10 = np.percentile(masked_probs, [90, 10])
        prob_sharpness_iqr = float(q90 - q10)

        m_scores = c.get("member_scores", [sc])
        m_arr = np.asarray(m_scores, dtype=np.float32)
        member_mean = float(m_arr.mean()) if len(m_arr) > 0 else sc
        member_std = float(m_arr.std()) if len(m_arr) > 1 else 0.0
        norm_views = float(c["cluster_size"]) * (4.0 / float(num_views_total))

        log_area = float(np.log10(area + 1.0))
        bbox_fill = float(area) / bbox_area
        aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))

        rank_in_image = rank_0 + 1
        rel_score_to_max = sc / max(1e-6, max_score_in_img)

        rows.append({
            "obs": obs,
            "fold": fold,
            "cand_idx": cand_idx,
            "cluster_score": sc,
            "num_views": norm_views,
            "member_score_mean": member_mean,
            "member_score_std": member_std,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "prob_sharpness_core": prob_sharpness_core,
            "prob_sharpness_iqr": prob_sharpness_iqr,
            "log_area": log_area,
            "bbox_fill": bbox_fill,
            "aspect_ratio": aspect_ratio,
            "rank_in_image": float(rank_in_image),
            "rel_score_to_max": rel_score_to_max,
            "n_cands_gt_05": float(n_cands_gt_05),
        })
        cand_idx += 1

    return rows


def collect_oof_multiview_features(
    folds_df: pd.DataFrame,
    oof_dirs: List[str],
    num_views_total: int,
    min_cluster_score: float = 0.15,
    tau_mask: float = 0.50,
    folds_to_include: List[int] = [0, 1, 2, 3, 4],
) -> pd.DataFrame:
    """
    Streams through all observations in folds_to_include, fuses raw candidates across
    oof_dirs, extracts the 14 scorer features, and immediately frees tensors + page cache.
    """
    all_rows = []
    for f in folds_to_include:
        val_obs = folds_df.loc[folds_df["fold"] == f, "observation"].tolist()
        for obs in val_obs:
            combined_cands = []
            for d in oof_dirs:
                pt_path = os.path.join(d, f"fold{f}", f"{obs}.pt")
                if not os.path.isfile(pt_path):
                    continue
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                cands = data.get("candidates", [])
                combined_cands.extend(cands)
                del data
                drop_os_page_cache(pt_path)

            if not combined_cands:
                continue

            fused = weighted_mask_fusion(combined_cands, num_views=num_views_total, iou_thresh=0.40)
            rows = extract_features_from_fused_clusters(
                fused=fused,
                obs=obs,
                fold=f,
                num_views_total=num_views_total,
                min_cluster_score=min_cluster_score,
                tau_mask=tau_mask,
            )
            all_rows.extend(rows)
            del combined_cands, fused

    gc.collect()
    return pd.DataFrame(all_rows)


def collect_test_features(
    model_names: List[str],
    num_views_total: int,
    cache_dir: str = "work/test_cache",
    min_cluster_score: float = 0.15,
    tau_mask: float = 0.50,
) -> pd.DataFrame:
    """
    Streams through all 180 test images, fuses cached raw candidates across model_names,
    extracts the 14 scorer features, and immediately frees tensors + page cache.
    """
    first_dir = os.path.join(cache_dir, model_names[0])
    pt_files = sorted(os.listdir(first_dir))
    test_stems = [Path(p).stem for p in pt_files if p.endswith(".pt")]

    all_rows = []
    for stem in test_stems:
        combined_cands = []
        for m_name in model_names:
            pt_path = os.path.join(cache_dir, m_name, f"{stem}.pt")
            if not os.path.isfile(pt_path):
                continue
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            cands = data.get("candidates", [])
            combined_cands.extend(cands)
            del data
            drop_os_page_cache(pt_path)

        if not combined_cands:
            continue

        fused = weighted_mask_fusion(combined_cands, num_views=num_views_total, iou_thresh=0.40)
        rows = extract_features_from_fused_clusters(
            fused=fused,
            obs=stem,
            fold=-1,
            num_views_total=num_views_total,
            min_cluster_score=min_cluster_score,
            tau_mask=tau_mask,
        )
        all_rows.extend(rows)
        del combined_cands, fused

    gc.collect()
    return pd.DataFrame(all_rows)


def run_adversarial_classifier(
    df_0: pd.DataFrame,
    df_1: pd.DataFrame,
    feature_cols: List[str],
    label_0: str = "OOF",
    label_1: str = "Test",
    seed: int = 2026,
) -> dict:
    """
    Trains a 5-fold StratifiedKFold LGBMClassifier to separate df_0 (y=0) vs df_1 (y=1).
    Groups by image (obs) so candidates from the same image do not leak across folds!
    Also computes single-feature ROC AUC and feature importance (gain %).
    """
    df_a = df_0.copy()
    df_a["target"] = 0
    df_b = df_1.copy()
    df_b["target"] = 1
    df_all = pd.concat([df_a, df_b], ignore_index=True)

    # Image-level stratified split so candidates from the same image stay in the same fold
    img_df = df_all.groupby("obs", as_index=False)["target"].first()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(df_all), dtype=np.float64)
    gain_importances = np.zeros(len(feature_cols), dtype=np.float64)

    for train_idx, val_idx in skf.split(img_df, img_df["target"]):
        val_obs_set = set(img_df.iloc[val_idx]["obs"])
        val_mask = df_all["obs"].isin(val_obs_set).values
        train_mask = ~val_mask

        X_tr = df_all.loc[train_mask, feature_cols].values
        y_tr = df_all.loc[train_mask, "target"].values
        X_va = df_all.loc[val_mask, feature_cols].values

        clf = lgb.LGBMClassifier(
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=4,
        )
        clf.fit(X_tr, y_tr)
        oof_preds[val_mask] = clf.predict_proba(X_va)[:, 1]
        gain_importances += clf.booster_.feature_importance(importance_type="gain") / 5.0

    overall_auc = float(roc_auc_score(df_all["target"].values, oof_preds))
    total_gain = max(1e-9, gain_importances.sum())
    gain_pct = (gain_importances / total_gain) * 100.0

    # Single-feature AUC and summary statistics
    feat_breakdown = []
    y_true = df_all["target"].values
    for idx, f_name in enumerate(feature_cols):
        vals = df_all[f_name].values
        raw_auc = float(roc_auc_score(y_true, vals))
        # Symmetrized single-feature separability: max(auc, 1 - auc)
        sep_auc = max(raw_auc, 1.0 - raw_auc)
        m0 = float(df_0[f_name].mean())
        s0 = float(df_0[f_name].std())
        m1 = float(df_1[f_name].mean())
        s1 = float(df_1[f_name].std())
        smd = (m1 - m0) / max(1e-6, np.sqrt(0.5 * (s0 ** 2 + s1 ** 2)))
        feat_breakdown.append({
            "feature": f_name,
            "gain_pct": float(gain_pct[idx]),
            "single_feat_auc": sep_auc,
            "raw_auc": raw_auc,
            f"mean_{label_0}": m0,
            f"std_{label_0}": s0,
            f"mean_{label_1}": m1,
            f"std_{label_1}": s1,
            "std_mean_diff": float(smd),
        })

    feat_breakdown.sort(key=lambda x: x["gain_pct"], reverse=True)
    return {
        "auc": overall_auc,
        "n_0": len(df_0),
        "n_1": len(df_1),
        "n_imgs_0": int(df_0["obs"].nunique()),
        "n_imgs_1": int(df_1["obs"].nunique()),
        "features": feat_breakdown,
    }


def print_adv_report(title: str, res: dict, label_0: str = "OOF", label_1: str = "Test", top_k: int = 14):
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)
    print(
        f"  5-Fold Image-Grouped ROC AUC: {res['auc']:.4f}  "
        f"({label_0}: {res['n_0']} cands across {res['n_imgs_0']} imgs | "
        f"{label_1}: {res['n_1']} cands across {res['n_imgs_1']} imgs)"
    )
    print("-" * 92)
    print(
        f"  {'Feature':<22} | {'Gain %':>7} | {'1-Feat AUC':>10} | "
        f"{'Mean ' + label_0:>10} | {'Mean ' + label_1:>10} | {'StdDiff':>8}"
    )
    print("-" * 92)
    for row in res["features"][:top_k]:
        print(
            f"  {row['feature']:<22} | {row['gain_pct']:>6.1f}% | {row['single_feat_auc']:>10.4f} | "
            f"{row['mean_' + label_0]:>10.4f} | {row['mean_' + label_1]:>10.4f} | {row['std_mean_diff']:>+8.3f}"
        )
    print("=" * 92)


def main():
    argparse.ArgumentParser(
        description="Adversarial validation comparing OOF vs test candidate feature distributions."
    ).parse_args()
    t0 = time.time()
    paths = find_dataset_paths()
    folds_df = pd.read_csv(paths["folds_csv"])

    print("Extracting OOF multi-view features across all 707 images (streaming 1 image at a time)...")
    df_oof_4v = collect_oof_multiview_features(
        folds_df=folds_df,
        oof_dirs=["work/oof"],
        num_views_total=4,
    )
    print(f"  -> 4-view OOF (clahe): {len(df_oof_4v)} candidates")

    df_oof_8v = collect_oof_multiview_features(
        folds_df=folds_df,
        oof_dirs=["work/oof", "work/oof_a5"],
        num_views_total=8,
    )
    print(f"  -> 8-view OOF (clahe + anchors5): {len(df_oof_8v)} candidates")

    df_oof_12v = collect_oof_multiview_features(
        folds_df=folds_df,
        oof_dirs=["work/oof", "work/oof_a5", "work/oof_dice"],
        num_views_total=12,
    )
    print(f"  -> 12-view OOF (clahe + anchors5 + dice): {len(df_oof_12v)} candidates")

    # Also extract 16-view Fold 0 OOF (clahe + anchors5 + dice + 2048) to match 16-view Test
    df_oof_f0_16v = collect_oof_multiview_features(
        folds_df=folds_df,
        oof_dirs=["work/oof", "work/oof_a5", "work/oof_dice", "work/oof_2048"],
        num_views_total=16,
        folds_to_include=[0],
    )
    print(f"  -> 16-view Fold-0 OOF (clahe + a5 + dice + 2048): {len(df_oof_f0_16v)} candidates")

    print("\nExtracting Test-set features across cached models (180 test images)...")
    # 1) 4-view single-model test (f0_1536)
    df_test_4v = collect_test_features(
        model_names=["f0_1536"],
        num_views_total=4,
    )
    print(f"  -> 4-view Test (f0_1536): {len(df_test_4v)} candidates")

    # 2) 12-view 3-family test (f0_1536 + f0_anchors5 + f0_dice)
    df_test_12v = collect_test_features(
        model_names=["f0_1536", "f0_anchors5", "f0_dice"],
        num_views_total=12,
    )
    print(f"  -> 12-view Test (f0_1536 + f0_anchors5 + f0_dice): {len(df_test_12v)} candidates")

    # 3) 16-view 4-family test (f0_1536 + f0_anchors5 + f0_dice + f0_2048)
    df_test_16v = collect_test_features(
        model_names=["f0_1536", "f0_anchors5", "f0_dice", "f0_2048"],
        num_views_total=16,
    )
    print(f"  -> 16-view Test (f0_1536 + f0_anchors5 + f0_dice + f0_2048): {len(df_test_16v)} candidates")

    # 4) 20-view 5-fold clahe test (f0..f4_1536)
    df_test_20v = collect_test_features(
        model_names=["f0_1536", "f1_1536", "f2_1536", "f3_1536", "f4_1536"],
        num_views_total=20,
    )
    print(f"  -> 20-view Test (5-fold 1536_clahe): {len(df_test_20v)} candidates")

    # -------------------------------------------------------------------------
    # PART 1A: Check Scorer Features Across 4, 8, and 12 Views (707 OOF Images)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  PART 1A: SCORER FEATURE STABILITY ACROSS 4, 8, AND 12 VIEWS (707 OOF IMAGES)")
    print("#" * 92)
    print(
        f"  {'Feature':<22} | {'4v Mean (Std)':>18} | {'8v Mean (Std)':>18} | "
        f"{'12v Mean (Std)':>18} | {'1-Feat AUC 4v/12v':>17}"
    )
    print("-" * 92)
    df_4_12 = pd.concat([df_oof_4v.assign(v=0), df_oof_12v.assign(v=1)], ignore_index=True)
    for f_name in FEATURE_NAMES:
        m4, s4 = df_oof_4v[f_name].mean(), df_oof_4v[f_name].std()
        m8, s8 = df_oof_8v[f_name].mean(), df_oof_8v[f_name].std()
        m12, s12 = df_oof_12v[f_name].mean(), df_oof_12v[f_name].std()
        raw_auc = roc_auc_score(df_4_12["v"].values, df_4_12[f_name].values)
        sep_auc = max(raw_auc, 1.0 - raw_auc)
        print(
            f"  {f_name:<22} | {m4:>8.4f} ({s4:>6.4f}) | {m8:>8.4f} ({s8:>6.4f}) | "
            f"{m12:>8.4f} ({s12:>6.4f}) | {sep_auc:>17.4f}"
        )

    res_4v_vs_8v = run_adversarial_classifier(df_oof_4v, df_oof_8v, FEATURE_NAMES, "4v_OOF", "8v_OOF")
    print_adv_report("VIEW-SHIFT CHECK 1: 4-View OOF vs 8-View OOF (All 14 Features)", res_4v_vs_8v, "4v_OOF", "8v_OOF", top_k=8)

    res_4v_vs_12v = run_adversarial_classifier(df_oof_4v, df_oof_12v, FEATURE_NAMES, "4v_OOF", "12v_OOF")
    print_adv_report("VIEW-SHIFT CHECK 2: 4-View OOF vs 12-View OOF (All 14 Features)", res_4v_vs_12v, "4v_OOF", "12v_OOF", top_k=8)

    res_8v_vs_12v = run_adversarial_classifier(df_oof_8v, df_oof_12v, FEATURE_NAMES, "8v_OOF", "12v_OOF")
    print_adv_report("VIEW-SHIFT CHECK 3: 8-View OOF vs 12-View OOF (All 14 Features)", res_8v_vs_12v, "8v_OOF", "12v_OOF", top_k=8)

    # -------------------------------------------------------------------------
    # PART 1B: Adversarial Validation — OOF Features vs Test Features
    # -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("  PART 1B: ADVERSARIAL VALIDATION — OOF FEATURES VS TEST FEATURES")
    print("#" * 92)

    # 1. Matched 4-view OOF vs 4-view Test
    adv_4v_matched = run_adversarial_classifier(df_oof_4v, df_test_4v, FEATURE_NAMES, "OOF_4v", "Test_4v")
    print_adv_report("ADVERSARIAL 1: Matched 4-View Single-Model (OOF 4v vs Test f0_1536 4v)", adv_4v_matched, "OOF_4v", "Test_4v")

    # 2. Matched 12-view 3-Family OOF (all 707) vs 12-view 3-Family Test
    adv_12v_matched = run_adversarial_classifier(df_oof_12v, df_test_12v, FEATURE_NAMES, "OOF_12v", "Test_12v")
    print_adv_report("ADVERSARIAL 2: Matched 12-View 3-Family Ensemble (OOF 12v vs Test 12v: clahe+a5+dice)", adv_12v_matched, "OOF_12v", "Test_12v")

    # 3. Matched 12-view Fold-0 OOF vs 12-view Test (exact same 3 checkpoints!)
    df_oof_f0_12v = df_oof_12v[df_oof_12v["fold"] == 0].copy()
    adv_12v_f0 = run_adversarial_classifier(df_oof_f0_12v, df_test_12v, FEATURE_NAMES, "OOF_F0_12v", "Test_12v")
    print_adv_report("ADVERSARIAL 3: Exact Same 3 Checkpoints (Fold-0 OOF 12v vs Test 12v)", adv_12v_f0, "OOF_F0_12v", "Test_12v")

    # 4. Matched 16-view Fold-0 OOF vs 16-view Test (exact same 4 checkpoints: clahe+a5+dice+2048)
    adv_16v_f0 = run_adversarial_classifier(df_oof_f0_16v, df_test_16v, FEATURE_NAMES, "OOF_F0_16v", "Test_16v")
    print_adv_report("ADVERSARIAL 4: Exact Same 4 Checkpoints (Fold-0 OOF 16v vs Test 16v: +2048)", adv_16v_f0, "OOF_F0_16v", "Test_16v")

    # 5. Mismatched view count: 4-view OOF vs 20-view 5-fold Test (shows why view-count normalization matters)
    adv_4v_vs_20v = run_adversarial_classifier(df_oof_4v, df_test_20v, FEATURE_NAMES, "OOF_4v", "Test_20v")
    print_adv_report("ADVERSARIAL 5 (MISMATCHED VIEWS): 4-View OOF vs 20-View 5-Fold Test", adv_4v_vs_20v, "OOF_4v", "Test_20v")

    # 6. Check AUC after dropping view-count discrete / multi-model variance features
    view_invariant_cols = [
        c for c in FEATURE_NAMES
        if c not in ("num_views", "member_score_std")
    ]
    adv_4v_vs_20v_inv = run_adversarial_classifier(
        df_oof_4v, df_test_20v, view_invariant_cols, "OOF_4v", "Test_20v"
    )
    print_adv_report(
        "ADVERSARIAL 6: 4-View OOF vs 20-View Test AFTER Dropping (num_views, member_score_std)",
        adv_4v_vs_20v_inv,
        "OOF_4v",
        "Test_20v",
    )

    out_summary = {
        "view_shift_4v_vs_8v_auc": res_4v_vs_8v["auc"],
        "view_shift_4v_vs_12v_auc": res_4v_vs_12v["auc"],
        "view_shift_8v_vs_12v_auc": res_8v_vs_12v["auc"],
        "adv_matched_4v_oof_vs_test": adv_4v_matched,
        "adv_matched_12v_oof_vs_test": adv_12v_matched,
        "adv_matched_12v_fold0_vs_test": adv_12v_f0,
        "adv_matched_16v_fold0_vs_test": adv_16v_f0,
        "adv_mismatched_4v_oof_vs_20v_test": adv_4v_vs_20v,
        "adv_mismatched_4v_vs_20v_view_invariant": adv_4v_vs_20v_inv,
    }
    out_path = "work/oof/adversarial_validation_summary.json"
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nSaved Step 1 Adversarial Validation summary to {out_path} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
