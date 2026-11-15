"""
tests/test_ensemble_config.py - Tests for ensemble configuration and Scorer v2 packaging.
"""
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_final_ensemble_config_structure():
    """Verify configs/final_ensemble.json exists and conforms to schema."""
    cfg_path = REPO_ROOT / "configs" / "final_ensemble.json"
    assert cfg_path.is_file(), f"Missing config file: {cfg_path}"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    assert "models" in cfg, "Missing 'models' key in config"
    assert len(cfg["models"]) == 30, f"Expected 30 models in final ensemble, found {len(cfg['models'])}"

    assert cfg.get("total_views") == 120, f"Expected total_views=120, got {cfg.get('total_views')}"
    assert "family_views" in cfg, "Missing 'family_views' mapping"
    assert cfg["family_views"].get("clahe") == 20
    assert cfg["family_views"].get("anchors5") == 20
    assert cfg["family_views"].get("dice") == 20
    assert cfg["family_views"].get("2048") == 20
    assert cfg["family_views"].get("a5dice") == 20
    assert cfg["family_views"].get("1792_clahe") == 20

    # Validate each model specification
    required_keys = {"id", "pruned_name", "checkpoint_relpath", "family", "img_size", "anchors5"}
    valid_families = {"clahe", "anchors5", "dice", "2048", "a5dice", "1792_clahe"}
    for idx, m in enumerate(cfg["models"]):
        missing = required_keys - set(m.keys())
        assert not missing, f"Model #{idx} ({m.get('id')}) missing required keys: {missing}"
        assert m["family"] in valid_families, f"Unexpected family: {m['family']}"
        assert m["img_size"] in {1536, 1792, 2048}
        assert isinstance(m["anchors5"], bool)

    # Validate decision rules (Rule A and Rule B)
    assert "decision_rules" in cfg
    dec_rules = cfg["decision_rules"]
    assert "rule_A" in dec_rules
    assert dec_rules["rule_A"]["tau_score"] == 0.850
    assert dec_rules["rule_A"]["tau_mask"] == 0.550

    assert "rule_B" in dec_rules
    assert dec_rules["rule_B"]["tau_score"] == 0.750
    assert dec_rules["rule_B"]["tau_mask"] == 0.550
    assert dec_rules["rule_B"]["family_weights"]["2048"] == 2.0
    assert dec_rules["rule_B"]["family_weights"]["1792_clahe"] == 2.0


def test_load_ensemble_config():
    """Verify load_ensemble_config loads default config properly."""
    from scripts.regenerate_submission import load_ensemble_config

    cfg = load_ensemble_config()
    assert cfg is not None
    assert len(cfg["models"]) == 30

    with pytest.raises(FileNotFoundError):
        load_ensemble_config(REPO_ROOT / "non_existent_config.json")


def test_fast_single_family_config():
    """Verify configs/fast_single_family.json exists and conforms to single-family schema."""
    fast_path = REPO_ROOT / "configs" / "fast_single_family.json"
    assert fast_path.is_file(), f"Missing fast config: {fast_path}"
    with open(fast_path, "r") as f:
        fast_cfg = json.load(f)
    assert "models" in fast_cfg
    assert len(fast_cfg["models"]) == 1
    assert fast_cfg["total_views"] == 4


def test_check_ensemble_weights_available():
    """Verify all 30 Stage-1 ensemble checkpoints exist strictly inside repo weights/."""
    from scripts.regenerate_submission import check_ensemble_weights_available

    weights_ok, missing = check_ensemble_weights_available(
        weights_dir=REPO_ROOT / "weights",
        strict_weights_only=True,
    )
    assert weights_ok is True, f"Missing repo Stage-1 weights: {missing}"
    assert missing == []


def test_scorer_v2_models_joblib():
    """Verify scorer_v2_models.joblib can be deserialized and produces valid predictions."""
    sc_path = REPO_ROOT / "work" / "checkpoints" / "scorer_v2_models.joblib"
    if not sc_path.is_file():
        pytest.skip("scorer_v2_models.joblib not present in work/checkpoints")

    import joblib
    import numpy as np

    models = joblib.load(sc_path)
    assert isinstance(models, list), "Expected list of models"
    assert len(models) == 5, f"Expected 5 cross-fitted LightGBM models, got {len(models)}"

    # Test dummy prediction with 19 view-invariant features
    dummy_feat = np.ones((1, 19), dtype=np.float32)
    preds = [m.predict(dummy_feat)[0] for m in models]
    mean_pred = float(np.mean(preds))
    assert np.isfinite(mean_pred), "Prediction should be finite float"


def test_extract_state_dict_fp16_compat():
    """Verify extract_state_dict upcasts FP16 tensors to float32 while preserving integer buffers."""
    import torch
    from scripts.common import extract_state_dict

    ckpt = {
        "model_state_dict": {
            "weight": torch.ones((4, 4), dtype=torch.float16),
            "num_batches_tracked": torch.tensor(10, dtype=torch.int64),
        }
    }
    sd = extract_state_dict(ckpt)
    assert sd["weight"].dtype == torch.float32
    assert sd["num_batches_tracked"].dtype == torch.int64


def test_in_repo_weights_under_100mb_and_checksums():
    """Verify all 81 final weight files in weights/ exist, are < 100 MB each, and match SHA-256."""
    import hashlib

    weights_dir = REPO_ROOT / "weights"
    assert weights_dir.is_dir(), f"Missing {weights_dir}"
    checksum_file = weights_dir / "checksums.sha256"
    assert checksum_file.is_file(), f"Missing {checksum_file}"

    expected = {}
    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, fname = line.split(None, 1)
        expected[fname] = digest

    assert len(expected) == 81, f"Expected 81 checksum entries in weights/checksums.sha256, got {len(expected)}"
    for fname, digest in expected.items():
        fpath = weights_dir / fname
        assert fpath.is_file(), f"Missing weight file {fpath}"
        assert not fpath.is_symlink(), f"{fpath} must be a regular file, not a symlink"
        size_bytes = fpath.stat().st_size
        assert size_bytes < 100_000_000, f"{fname} exceeds 100 MB decimal ({size_bytes} bytes)"
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual == digest, f"SHA-256 mismatch for {fname}: {actual} != {digest}"


def test_phase3_refiner_config_and_offline_smoke():
    """Verify configs/phase3b_refiner.json schema, sharded FP16 reassembly, and CPU smoke checks."""
    import numpy as np
    import torch
    from scripts.refiner_pipeline import (
        ResidualCropRefinerUNet,
        ResNet50SemanticUNet,
        ResNeXt101SemanticUNet,
        WideContextRefinerUNetV2,
        load_lgb_boosters,
        load_phase3_state_dict,
        resolve_stage1_checkpoint_path,
    )

    cfg_path = REPO_ROOT / "configs" / "phase3b_refiner.json"
    assert cfg_path.is_file(), f"Missing Phase 3B config: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    assert cfg["base_ensemble_config"] == "configs/final_ensemble.json"
    assert len(cfg["semantic_priors"]["resnet50_unet"]["checkpoints"]) == 5
    assert len(cfg["semantic_priors"]["resnext101_unet"]["checkpoints"]) == 5
    assert len(cfg["roi_refiners"]["crop_refiner_v1"]["checkpoints"]) == 5
    assert len(cfg["roi_refiners"]["wide_refiner_v2"]["checkpoints"]) == 5
    assert cfg["ranker"]["num_boosters"] == 25
    assert cfg["ranker"]["tau_score"] == 0.255
    assert cfg["mask_blending"]["mode"] == "wref_uv1_t50"

    weights_dir = REPO_ROOT / "weights"
    with open(REPO_ROOT / "configs" / "final_ensemble.json", "r") as f:
        ens_cfg = json.load(f)
    for m in ens_cfg["models"]:
        resolved = resolve_stage1_checkpoint_path(m, weights_dir=weights_dir, strict_weights_only=True)
        assert resolved.resolve().is_relative_to(weights_dir.resolve())

    boosters, feat_names = load_lgb_boosters(weights_dir=weights_dir, strict_weights_only=True)
    assert len(boosters) == 25
    assert len(feat_names) == 27
    dummy_x = np.zeros((1, 27), dtype=np.float32)
    lgb_preds = [float(b.predict(dummy_x)[0]) for b in boosters]
    assert all(np.isfinite(p) for p in lgb_preds)

    m_r50 = ResNet50SemanticUNet(out_ch=2)
    m_r50.load_state_dict(load_phase3_state_dict(
        "semantic_unet_f0_fp16.pt", weights_dir=weights_dir, map_location="cpu",
        strict_weights_only=True, verify_fp16=True,
    ))
    m_r50.eval()

    m_rnx = ResNeXt101SemanticUNet(out_ch=3)
    m_rnx.load_state_dict(load_phase3_state_dict(
        "resnext101_unet_f0_fp16.pt", weights_dir=weights_dir, map_location="cpu",
        strict_weights_only=True, verify_fp16=True,
    ))
    m_rnx.eval()

    m1 = ResidualCropRefinerUNet(in_ch=6, base=32)
    m1.load_state_dict(load_phase3_state_dict(
        "phase3/crop_refiner_f0_fp16.pt", weights_dir=weights_dir, map_location="cpu",
        strict_weights_only=True, verify_fp16=True,
    ))
    m1.eval()

    m2 = WideContextRefinerUNetV2(in_ch=11, base=40)
    m2.load_state_dict(load_phase3_state_dict(
        "phase3/wide_refiner_v2_f0_fp16.pt", weights_dir=weights_dir, map_location="cpu",
        strict_weights_only=True, verify_fp16=True,
    ))
    m2.eval()

    with torch.no_grad():
        s_vec = torch.zeros((1, 4), dtype=torch.float32)
        out_r50 = m_r50(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
        out_rnx = m_rnx(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
        out1, iou1 = m1(torch.zeros((1, 6, 160, 160), dtype=torch.float32), s_vec)
        out2, iou2 = m2(torch.zeros((1, 11, 224, 224), dtype=torch.float32), s_vec)
    assert tuple(out_r50.shape) == (1, 2, 64, 64) and torch.isfinite(out_r50).all()
    assert tuple(out_rnx.shape) == (1, 3, 64, 64) and torch.isfinite(out_rnx).all()
    assert tuple(out1.shape) == (1, 160, 160) and torch.isfinite(iou1).all()
    assert tuple(out2.shape) == (1, 224, 224) and torch.isfinite(iou2).all()


def test_fold_split_disjoint_no_leakage():
    """Verify work/folds.csv strictly partitions 707 observations across 5 disjoint folds."""
    import pandas as pd

    folds_path = REPO_ROOT / "work" / "folds.csv"
    assert folds_path.is_file(), f"Missing {folds_path}"
    df = pd.read_csv(folds_path)
    assert len(df) == 707
    assert df["observation"].nunique() == 707
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}
    by_fold = [set(df[df["fold"] == f]["observation"]) for f in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(by_fold[i] & by_fold[j]) == 0
    sub_dir = REPO_ROOT / "submissions"
    if sub_dir.is_dir():
        train_obs = set(df["observation"].astype(str))
        for csv_p in sub_dir.glob("*.csv"):
            sub_df = pd.read_csv(csv_p)
            test_obs = {fid.rsplit("_", 1)[0] for fid in sub_df["filament_id"].astype(str)}
            assert len(train_obs & test_obs) == 0
