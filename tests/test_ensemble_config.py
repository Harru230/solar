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
    """Verify check_ensemble_weights_available returns boolean and list."""
    from scripts.regenerate_submission import check_ensemble_weights_available

    weights_ok, missing = check_ensemble_weights_available()
    assert isinstance(weights_ok, bool)
    assert isinstance(missing, list)


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
    """Verify compact Fold-0 FP16 weights in weights/ exist, are < 100 MB each, and match SHA-256."""
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

    assert len(expected) == 42, f"Expected 42 checksum entries in weights/checksums.sha256, got {len(expected)}"
    for fname, digest in expected.items():
        fpath = weights_dir / fname
        assert fpath.is_file(), f"Missing weight file {fpath}"
        assert not fpath.is_symlink(), f"{fpath} must be a regular file, not a symlink"
        size_mb = fpath.stat().st_size / (1024 * 1024)
        assert size_mb < 100.0, f"{fname} exceeds 100 MB ({size_mb:.2f} MB)"
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual == digest, f"SHA-256 mismatch for {fname}: {actual} != {digest}"


def test_phase3_refiner_config_and_offline_smoke():
    """Verify configs/phase3b_refiner.json schema and run offline CPU forward smoke checks."""
    import numpy as np
    import torch
    from scripts.common import extract_state_dict
    from scripts.refiner_pipeline import (
        ResidualCropRefinerUNet,
        WideContextRefinerUNetV2,
        load_lgb_boosters,
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

    boosters, feat_names = load_lgb_boosters()
    assert len(boosters) == 25
    assert len(feat_names) == 27
    dummy_x = np.zeros((1, 27), dtype=np.float32)
    lgb_preds = [float(b.predict(dummy_x)[0]) for b in boosters]
    assert all(np.isfinite(p) for p in lgb_preds)

    phase3_dir = REPO_ROOT / "weights" / "phase3"
    m1 = ResidualCropRefinerUNet(in_ch=6, base=32)
    m1.load_state_dict(extract_state_dict(torch.load(phase3_dir / "crop_refiner_f0_fp16.pt", map_location="cpu", weights_only=False)))
    m1.eval()
    m2 = WideContextRefinerUNetV2(in_ch=11, base=40)
    m2.load_state_dict(extract_state_dict(torch.load(phase3_dir / "wide_refiner_v2_f0_fp16.pt", map_location="cpu", weights_only=False)))
    m2.eval()

    with torch.no_grad():
        s_vec = torch.zeros((1, 4), dtype=torch.float32)
        out1, iou1 = m1(torch.zeros((1, 6, 160, 160), dtype=torch.float32), s_vec)
        out2, iou2 = m2(torch.zeros((1, 11, 224, 224), dtype=torch.float32), s_vec)
    assert tuple(out1.shape) == (1, 160, 160) and torch.isfinite(iou1).all()
    assert tuple(out2.shape) == (1, 224, 224) and torch.isfinite(iou2).all()
