#!/usr/bin/env python3
"""
scripts/run_pipeline.py - Official End-to-End Inference Pipeline for Solar Filament Segmentation.

Executes the submission pipeline on any folder of solar images:
1. Single-Model Mode (--mode single):
   - Fast, lightweight inference on CPU or GPU using a single trained Mask R-CNN.
   - 4-View Flip TTA or single pass, Weighted Mask Fusion (WMF), and strictly disjoint canvas postprocessing.
2. Multi-Model Ensemble Mode (--mode ensemble):
   - The authoritative competition submission pipeline (identical to submitted Kaggle files).
   - 8-model multi-family ensemble (5-fold CLAHE + 2048px + Anchors5 + Soft Dice).
   - 32-view Weighted Mask Fusion (WMF) with cluster score S = sum(s_k) / 32.
   - Decision via Threshold Rule B (tau_score=0.875, tau_mask=0.550) or LightGBM Scorer v2.
   - Canvas disjointness enforcing max sum(M_i) <= 1 across all pixels.
   - Outputs valid submission CSV with empty image handling (<stem>_1,PPPP4).

Usage:
  # Single model on sample images (CPU fallback)
  python scripts/run_pipeline.py --input_dir sample_images --mode single --output_csv work/sample_sub.csv

  # Full 8-model ensemble on test set
  python scripts/run_pipeline.py --input_dir data/test_images --mode ensemble --decision threshold --output_csv work/submission.csv
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    EMPTY_RLE,
    extract_state_dict,
    find_dataset_paths,
    get_autocast_context,
    get_maskrcnn_model,
    make_masks_disjoint,
    preprocess_solar_image,
    rle_encode_mask,
)
from scripts.instance_scorer import encode_crop_to_rle
from scripts.multifamily_pipeline import (
    SCORER_V2_VIEW_INVARIANT_FEATURES,
    compute_binary_crop_iou,
    weighted_mask_fusion_tagged,
)
from scripts.predict_ensemble import (
    apply_flip_transform,
    unflip_boxes,
    unflip_mask_probabilities,
)
from scripts.regenerate_submission import (
    ENSEMBLE_8MODEL_SPECS,
    FAMILY_VIEWS,
    TOTAL_VIEWS,
    apply_scorer_v2_decision,
    apply_threshold_decision,
    check_ensemble_weights_available,
    load_ensemble_config,
)

FRAME_SIZE = 2048


def locate_single_model_checkpoint(
    weights_dir: Path,
    explicit_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Finds best available single-model checkpoint in order of preference."""
    if repo_root is None:
        repo_root = REPO_ROOT

    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        p_rel = repo_root / explicit_path
        if p_rel.is_file():
            return p_rel
        raise FileNotFoundError(f"Specified checkpoint not found: {explicit_path}")

    upstream_dir = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(repo_root.parent / "solar")))

    candidates = [
        weights_dir / "r50_1536_clahe_f0.pt",
        repo_root / "weights/r50_1536_clahe_f0.pt",
        repo_root / "work/checkpoints/r50_1536_clahe_f0.pt",
        repo_root / "work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt",
        repo_root / "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt",
        upstream_dir / "work/checkpoints/r50_1536_clahe_f0.pt",
        upstream_dir / "work/runs/r50_1536_clahe_e30/checkpoints/maskrcnn_fold0_best.pt",
        upstream_dir / "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt",
        weights_dir / "maskrcnn_fold0_best.pt",
        repo_root / "work/checkpoints/maskrcnn_fold0_best.pt",
        upstream_dir / "work/checkpoints/maskrcnn_fold0_best.pt",
    ]

    for cand in candidates:
        if cand.is_file():
            return cand

    checked_str = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No model checkpoint found! Checked candidates:{checked_str}\n"
        "Please download weights using 'bash scripts/download_weights.sh' or specify --checkpoint <path>."
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    default_img_size: int = 1536,
    default_anchors5: bool = False,
) -> Tuple[torch.nn.Module, int, bool]:
    """Loads a Mask R-CNN model onto the specified device."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    img_size = cfg.get("img_size", default_img_size)
    use_anchors5 = cfg.get("anchors5", default_anchors5)

    model = get_maskrcnn_model(
        pretrained=False,
        min_size=img_size,
        max_size=img_size,
        num_classes=2,
        anchors5=use_anchors5,
    )
    model.load_state_dict(state_dict)
    model.float()
    model.to(device)
    model.eval()
    return model, img_size, use_anchors5


def run_pipeline(
    input_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    mode: str = "single",
    decision: str = "threshold",
    rule: str = "B",
    config_path: Optional[Union[str, Path]] = None,
    weights_dir: Optional[Union[str, Path]] = None,
    checkpoint: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    tau_score: Optional[float] = None,
    tau_mask: Optional[float] = None,
    tau_scorer: Optional[float] = None,
    cand_score_thresh: Optional[float] = None,
    tta: bool = True,
    limit: Optional[int] = None,
    validate: bool = True,
    allow_dummy: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the inference pipeline on a directory of images.

    Parameters:
      input_dir: Directory containing full-disk solar observation images.
      output_csv: Optional file path where prediction CSV will be written.
      mode: 'single' (lightweight single-model) or 'ensemble' (8-model WMF).
      decision: 'threshold' (Rule B) or 'scorer-v2' (LightGBM regressor).
      config_path: Path to JSON ensemble specification (defaults to configs/final_ensemble.json).
      weights_dir: Directory containing pruned model weights.
      checkpoint: Optional explicit path to single-model weights file.
      device: 'auto', 'cpu', 'cuda', or torch.device.
      tau_score: Candidate/cluster score threshold (defaults to config or 0.35/0.875).
      tau_mask: Mask probability threshold (defaults to config or 0.50/0.55/0.525).
      tau_scorer: LightGBM regressor cutoff for Scorer v2 (defaults to config: 0.2440).
      cand_score_thresh: Per-view candidate proposal filter prior to WMF clustering.
        Replaces the legacy `eff_tau_score * 0.5` heuristic in single mode with a principled
        multi-view consensus threshold (default: 0.25 under TTA, eff_tau_score without TTA).
      tta: Whether to apply 4-flip test-time augmentation.
      limit: Maximum number of images to process.
      validate: Whether to run validate_submission.py on the output CSV.
      allow_dummy: If True, fall back to dummy model when weights are missing.

    Returns:
      (df_predictions, results_per_image)
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Gather images
    image_paths = sorted(
        list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    )
    if not image_paths:
        raise ValueError(f"No image files (.jpeg, .jpg, .png) found in {input_dir}")

    if limit and limit > 0:
        image_paths = image_paths[:limit]

    # Resolve device
    if device is None or device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    # Resolve weights directory (prefer in-repo weights/ if work/checkpoints is absent)
    if weights_dir is None:
        default_ckpt_dir = REPO_ROOT / "work/checkpoints"
        weights_dir = default_ckpt_dir if default_ckpt_dir.is_dir() else (REPO_ROOT / "weights")
    else:
        weights_dir = Path(weights_dir)

    views = [
        ("id", False, False),
        ("h", False, True),
        ("v", True, False),
        ("hv", True, True),
    ] if tta else [("id", False, False)]
    num_views_tta = len(views)

    results_per_image: Dict[str, dict] = {}
    all_rows: List[dict] = []
    t_start = time.time()

    def _rel(p: Union[str, Path]) -> str:
        try:
            return str(Path(p).resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(p)

    print(f"\n{'=' * 80}")
    print(f"  Solar Filament Inference Pipeline")
    print(f"  Mode:           {mode.upper()}")
    print(f"  Decision Rule:  {decision}")
    print(f"  Device:         {target_device} (float32 compute)")
    print(f"  Images to Run:  {len(image_paths)} from {_rel(input_dir)}")
    view_noun = "view" if num_views_tta == 1 else "views"
    print(f"  TTA Enabled:    {tta} ({num_views_tta} {view_noun} per model)")
    print(f"{'=' * 80}\n")

    # ────────────────────────── SINGLE-MODEL MODE ────────────────────────────
    if mode == "single":
        eff_tau_score = tau_score if tau_score is not None else 0.35
        eff_tau_mask = tau_mask if tau_mask is not None else 0.50

        print(f"Locating model checkpoint in {_rel(weights_dir)}...")
        try:
            ckpt_path = locate_single_model_checkpoint(weights_dir, explicit_path=checkpoint, repo_root=REPO_ROOT)
            print(f"Using checkpoint: {_rel(ckpt_path)}")
            model, img_size, is_a5 = load_model_from_checkpoint(ckpt_path, device=target_device)
        except FileNotFoundError as e:
            if allow_dummy:
                print(f"[WARN] {e}")
                print("[WARN] --allow_dummy set: Initializing untrained Mask R-CNN for dry-run verification.")
                img_size = 1536
                is_a5 = False
                model = get_maskrcnn_model(pretrained=False, min_size=img_size, max_size=img_size, num_classes=2, anchors5=is_a5)
                model.to(target_device)
                model.eval()
            else:
                raise

        print(f"Model ready (img_size={img_size}, anchors5={is_a5}). Running inference...")

        for idx, img_path in enumerate(image_paths, start=1):
            t0 = time.time()
            stem = img_path.stem
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise IOError(f"Could not read image: {img_path}")

            # Multi-scale CLAHE preprocessing
            preprocessed_np = preprocess_solar_image(img_gray, target_size=img_size)
            base_t = torch.from_numpy(preprocessed_np).float()
            scale_to_2048 = 2048.0 / float(img_size)

            all_view_candidates: List[dict] = []
            with torch.no_grad():
                for v_name, f_v, f_h in views:
                    flipped_t = apply_flip_transform(base_t, flip_v=f_v, flip_h=f_h).to(target_device)
                    with get_autocast_context(target_device):
                        out = model([flipped_t])[0]

                    scs = out["scores"].detach().cpu()
                    # Candidate proposal extraction filter prior to WMF clustering / postprocessing.
                    # Under TTA, raw view proposals with score >= cand_score_thresh (default 0.25, matching
                    # predict_ensemble.py) are collected so multi-view consensus can form.
                    cand_filter = cand_score_thresh if cand_score_thresh is not None else (0.25 if tta else eff_tau_score)
                    keep = scs >= cand_filter
                    if not keep.any():
                        continue

                    scs_k = scs[keep].numpy()
                    boxes_k = out["boxes"][keep]
                    masks_k = out["masks"][keep]

                    # Unflip
                    unf_boxes = (unflip_boxes(boxes_k, flip_v=f_v, flip_h=f_h, height=img_size, width=img_size) * scale_to_2048).detach().cpu().numpy()
                    unf_masks = unflip_mask_probabilities(masks_k, flip_v=f_v, flip_h=f_h)
                    if img_size == FRAME_SIZE:
                        masks_up = unf_masks.squeeze(1).detach().cpu().numpy()
                    else:
                        masks_up = F.interpolate(
                            unf_masks.float(), size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
                        ).squeeze(1).detach().cpu().numpy()

                    for k in range(len(scs_k)):
                        b = unf_boxes[k]
                        x1 = max(0, int(np.floor(b[0])))
                        y1 = max(0, int(np.floor(b[1])))
                        x2 = min(FRAME_SIZE, int(np.ceil(b[2])) + 1)
                        y2 = min(FRAME_SIZE, int(np.ceil(b[3])) + 1)
                        if (x2 - x1) < 2 or (y2 - y1) < 2:
                            continue
                        crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float16)
                        all_view_candidates.append({
                            "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                            "score": float(scs_k[k]),
                            "prob_crop": crop_soft,
                            "family": "clahe",
                        })

            # Cluster with WMF if TTA is used, else treat as candidates
            if tta:
                fused = weighted_mask_fusion_tagged(
                    all_view_candidates,
                    num_views=num_views_tta,
                    iou_thresh=0.40,
                    family_views={"clahe": num_views_tta},
                )
            else:
                fused = all_view_candidates
                fused.sort(key=lambda c: c["score"], reverse=True)

            # Apply disjoint canvas
            occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            img_masks = []
            img_scores = []
            img_rles = []

            for c in fused:
                sc = float(c["score"])
                if sc < eff_tau_score:
                    break
                box = c["box"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bc = (c["prob_crop"] >= eff_tau_mask)
                oa = int(bc.sum())
                if oa == 0:
                    continue
                sub_occ = occupied[y1:y2, x1:x2]
                clean_crop = bc & (~sub_occ)
                ca = int(clean_crop.sum())
                if ca < 10 or (ca / float(oa)) < 0.50:
                    continue
                occupied[y1:y2, x1:x2] |= clean_crop

                # Build full 2048x2048 mask for verification/display
                m_full = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
                m_full[y1:y2, x1:x2] = clean_crop
                img_masks.append(m_full)
                img_scores.append(sc)
                img_rles.append(encode_crop_to_rle(box, clean_crop))

            # Check mathematical disjointness
            if len(img_masks) > 0:
                stack = np.stack(img_masks, axis=0)
                max_ov = int(stack.sum(axis=0).max())
                assert max_ov <= 1, f"Canvas disjointness violated on {stem}: max overlap={max_ov}"

            if not img_rles:
                all_rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
            else:
                for k, enc in enumerate(img_rles, start=1):
                    all_rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": enc})

            elapsed = time.time() - t0
            results_per_image[stem] = {
                "stem": stem,
                "img_gray": img_gray,
                "masks": img_masks,
                "scores": img_scores,
                "rles": img_rles,
                "elapsed_s": elapsed,
            }
            print(f"  [{idx:03d}/{len(image_paths):03d}] {stem}: {len(img_masks)} filaments ({elapsed:.2f}s)")

    # ────────────────────────── ENSEMBLE / SINGLE-FAMILY MODE ────────────────
    elif mode in ("ensemble", "single-family"):
        if config_path is None:
            if mode == "single-family":
                config_path = REPO_ROOT / "configs/fast_single_family.json"
            else:
                config_path = REPO_ROOT / "configs/final_ensemble.json"
        elif mode == "single-family" and str(config_path) == "configs/final_ensemble.json":
            config_path = REPO_ROOT / "configs/fast_single_family.json"
        cfg = load_ensemble_config(config_path)
        dec_cfg = cfg.get("decision_rules", {})
        thresh_cfg = dec_cfg.get("threshold", {})
        scorer_cfg = dec_cfg.get("scorer_v2", {})

        rule_key = f"rule_{rule.upper()}"
        if rule_key in dec_cfg:
            r_cfg = dec_cfg[rule_key]
            eff_tau_score = tau_score if tau_score is not None else r_cfg.get("tau_score", 0.750)
            eff_tau_mask = tau_mask if tau_mask is not None else r_cfg.get("tau_mask", 0.550)
            eff_gamma = r_cfg.get("gamma", 1.0)
            eff_family_weights = r_cfg.get("family_weights", {})
        else:
            eff_tau_score = tau_score if tau_score is not None else thresh_cfg.get("tau_score", 0.875)
            eff_tau_mask = tau_mask if tau_mask is not None else thresh_cfg.get("tau_mask", 0.550)
            eff_gamma = 1.0
            eff_family_weights = {}
        eff_tau_scorer = tau_scorer if tau_scorer is not None else scorer_cfg.get("tau_scorer", 0.2440)
        scorer_tau_mask = scorer_cfg.get("tau_mask", 0.525) if tau_mask is None else tau_mask

        if tta:
            total_views = cfg.get("total_views", sum(m.get("views", 4) for m in cfg["models"]))
            family_views = cfg.get("family_views", {})
        else:
            total_views = len(cfg["models"])
            family_views = {k: max(1, v // 4) for k, v in cfg.get("family_views", {}).items()}

        print(f"Loading ensemble ({cfg.get('name', 'custom')}) with {len(cfg['models'])} models from {_rel(weights_dir)}...")
        loaded_models = []
        upstream_dir = os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))

        for m_info in cfg["models"]:
            subdir = m_info["id"]
            pruned_name = m_info["pruned_name"]
            run_relpath = m_info.get("checkpoint_relpath", f"work/runs/{subdir}.pt")
            fam = m_info["family"]
            sz = m_info["img_size"]
            is_a5 = m_info.get("anchors5", False)

            cand_paths = [
                weights_dir / pruned_name,
                weights_dir / f"{subdir}.pt",
                REPO_ROOT / "weights" / pruned_name,
                REPO_ROOT / run_relpath,
                Path(upstream_dir) / "work/checkpoints" / pruned_name,
                Path(upstream_dir) / run_relpath,
            ]
            ckpt_path = next((p for p in cand_paths if p.is_file()), None)
            if not ckpt_path:
                raise FileNotFoundError(f"Could not locate checkpoint for {subdir}. Checked: {cand_paths}")

            ckpt = torch.load(ckpt_path, map_location=target_device, weights_only=False)
            state_dict = extract_state_dict(ckpt)
            model = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=is_a5)
            model.load_state_dict(state_dict)
            model.float()
            model.to(target_device)
            model.eval()
            loaded_models.append((subdir, pruned_name, fam, sz, is_a5, model))
            print(f"  Loaded {subdir:<12} ({fam}, {sz}px) from {ckpt_path.name}")

        scorer_models = None
        if decision == "scorer-v2":
            import joblib
            sc_name = scorer_cfg.get("weights_file", "scorer_v2_models.joblib")
            sc_path = weights_dir / sc_name
            if not sc_path.is_file():
                sc_path = REPO_ROOT / "work/checkpoints" / sc_name
            if not sc_path.is_file():
                sc_path = Path(upstream_dir) / "work/checkpoints" / sc_name
            if not sc_path.is_file():
                raise FileNotFoundError(f"Scorer v2 model file not found at {sc_path}")
            scorer_models = joblib.load(sc_path)
            print(f"  Loaded {len(scorer_models)} LightGBM Scorer v2 models from {sc_path.name}.")

        tta_desc = f"{total_views}-view Flip TTA" if tta else f"{total_views}-view (no TTA)"
        print(f"\nExecuting {len(loaded_models)}-model {tta_desc} WMF inference...")
        for idx, img_path in enumerate(image_paths, start=1):
            t0 = time.time()
            stem = img_path.stem
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise IOError(f"Could not read image: {img_path}")

            # Preprocess only at resolutions used by loaded_models
            needed_sizes = sorted({m[3] for m in loaded_models})
            t_by_size = {
                sz: torch.from_numpy(preprocess_solar_image(img_gray, target_size=sz)).float()
                for sz in needed_sizes
            }

            all_candidates: List[dict] = []
            with torch.no_grad():
                for subdir, pruned_name, fam, sz, is_a5, model in loaded_models:
                    base_t = t_by_size[sz]
                    scale_to_2048 = 2048.0 / float(sz)

                    for v_name, f_v, f_h in views:
                        flipped_t = apply_flip_transform(base_t, flip_v=f_v, flip_h=f_h).to(target_device)
                        with get_autocast_context(target_device):
                            out = model([flipped_t])[0]

                        scs = out["scores"].detach().cpu()
                        cand_filter = cand_score_thresh if cand_score_thresh is not None else 0.25
                        keep = scs >= cand_filter
                        if not keep.any():
                            continue

                        scs_k = scs[keep].numpy()
                        boxes_k = out["boxes"][keep]
                        masks_k = out["masks"][keep]

                        unf_boxes = (unflip_boxes(boxes_k, flip_v=f_v, flip_h=f_h, height=sz, width=sz) * scale_to_2048).detach().cpu().numpy()
                        unf_masks = unflip_mask_probabilities(masks_k, flip_v=f_v, flip_h=f_h)
                        if sz == FRAME_SIZE:
                            masks_up = unf_masks.squeeze(1).detach().cpu().numpy()
                        else:
                            masks_up = F.interpolate(
                                unf_masks.float(), size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
                            ).squeeze(1).detach().cpu().numpy()

                        for k in range(len(scs_k)):
                            b = unf_boxes[k]
                            x1 = max(0, int(np.floor(b[0])))
                            y1 = max(0, int(np.floor(b[1])))
                            x2 = min(FRAME_SIZE, int(np.ceil(b[2])) + 1)
                            y2 = min(FRAME_SIZE, int(np.ceil(b[3])) + 1)
                            if (x2 - x1) < 2 or (y2 - y1) < 2:
                                continue
                            crop_soft = masks_up[k, y1:y2, x1:x2].astype(np.float16)
                            all_candidates.append({
                                "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                                "score": float(scs_k[k]),
                                "prob_crop": crop_soft,
                                "family": fam,
                            })

            if decision == "threshold" and (eff_gamma != 1.0 or any(abs(v - 1.0) > 1e-9 for v in eff_family_weights.values())):
                for cand in all_candidates:
                    w_fam = float(eff_family_weights.get(cand["family"], 1.0))
                    cand["score"] = (float(cand["score"]) ** float(eff_gamma)) * w_fam
                eff_total_views = sum(
                    float(eff_family_weights.get(m_info["family"], 1.0))
                    * (m_info.get("views", 4) if tta else 1)
                    for m_info in cfg["models"]
                )
                eff_family_views = {
                    fam: float(eff_family_weights.get(fam, 1.0)) * cnt
                    for fam, cnt in family_views.items()
                }
            else:
                eff_total_views = total_views
                eff_family_views = family_views

            fused_clusters = weighted_mask_fusion_tagged(
                all_candidates,
                num_views=eff_total_views,
                iou_thresh=0.40,
                family_views=eff_family_views,
            )

            if decision == "scorer-v2":
                rows = apply_scorer_v2_decision(
                    fused_clusters,
                    stem=stem,
                    scorer_models=scorer_models,
                    tau_scorer=eff_tau_scorer,
                    tau_mask=scorer_tau_mask,
                )
            else:
                rows = apply_threshold_decision(
                    fused_clusters,
                    stem=stem,
                    tau_score=eff_tau_score,
                    tau_mask=eff_tau_mask,
                )

            all_rows.extend(rows)
            elapsed = time.time() - t0
            n_filaments = sum(1 for r in rows if r["segmentation_rle"] != EMPTY_RLE)
            print(f"  [{idx:03d}/{len(image_paths):03d}] {stem}: {n_filaments} filaments ({elapsed:.2f}s)")
            results_per_image[stem] = {
                "stem": stem,
                "img_gray": img_gray,
                "rows": rows,
                "elapsed_s": elapsed,
            }
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'single' or 'ensemble'.")

    df_predictions = pd.DataFrame(all_rows)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df_predictions)} prediction rows to: {_rel(output_csv)}")

        if validate:
            validate_script = REPO_ROOT / "scripts/validate_submission.py"
            if validate_script.is_file():
                import subprocess
                cmd = [
                    sys.executable,
                    str(validate_script),
                    "--sub",
                    str(output_csv),
                    "--test_dir",
                    str(input_dir),
                ]
                if not limit:
                    dir_images = (
                        list(input_dir.glob("*.jpeg"))
                        + list(input_dir.glob("*.jpg"))
                        + list(input_dir.glob("*.png"))
                    )
                    if dir_images:
                        cmd.extend(["--expected_count", str(len(dir_images))])
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    print("[VALIDATION PASSED] Submission format is fully competition compliant.")
                else:
                    print(f"[VALIDATION FAILED]\n{res.stdout}\n{res.stderr}")

    total_time = time.time() - t_start
    print(f"\nPipeline finished in {total_time:.1f}s ({total_time / len(image_paths):.2f}s/img).")
    return df_predictions, results_per_image


def main():
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation End-to-End Inference Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing solar images to process")
    parser.add_argument("--output_csv", type=str, default="work/submission_pipeline.csv", help="Path to save submission CSV")
    parser.add_argument("--mode", type=str, choices=["single", "single-family", "ensemble"], default="single",
                        help="Inference mode: 'single', 'single-family' (fast 1536 CLAHE config), or 'ensemble' (8-model WMF)")
    parser.add_argument("--config", type=str, default="configs/final_ensemble.json",
                        help="Path to ensemble configuration JSON (e.g. configs/compact_6model_fold0.json or configs/final_ensemble.json)")
    parser.add_argument("--decision", type=str, choices=["threshold", "scorer-v2"], default="threshold",
                        help="Decision rule: 'threshold' (Rule A/B WMF threshold used in final submissions) or 'scorer-v2' (experimental LightGBM scorer, not used in final submissions)")
    parser.add_argument("--rule", type=str, choices=["A", "B", "a", "b"], default="B",
                        help="Ensemble rule: 'A' (Uniform WMF, LB 0.39) or 'B' (Weighted WMF, LB 0.40, default)")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Directory containing model weights (defaults to work/checkpoints if present, else weights/)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path for single-model mode")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--tau_score", type=float, default=None, help="Candidate / cluster score threshold")
    parser.add_argument("--tau_mask", type=float, default=None, help="Mask probability threshold")
    parser.add_argument("--tau_scorer", type=float, default=None, help="LightGBM scorer threshold (default from config: 0.2440)")
    parser.add_argument("--cand_score_thresh", type=float, default=None,
                        help="Per-view candidate score filter prior to WMF clustering (default: 0.25 under TTA)")
    parser.add_argument("--no_tta", action="store_true", default=False, help="Disable 4-flip TTA (run single forward pass)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--no_validate", action="store_true", default=False, help="Skip submission validation")
    parser.add_argument("--allow_dummy", action="store_true", default=False, help="Allow untrained dummy model if weights missing")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        mode=args.mode,
        decision=args.decision,
        rule=args.rule,
        config_path=args.config,
        weights_dir=args.weights_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        tau_score=args.tau_score,
        tau_mask=args.tau_mask,
        tau_scorer=args.tau_scorer,
        cand_score_thresh=args.cand_score_thresh,
        tta=not args.no_tta,
        limit=args.limit,
        validate=not args.no_validate,
        allow_dummy=args.allow_dummy,
    )


if __name__ == "__main__":
    main()
