#!/usr/bin/env python3
"""
scripts/refiner_pipeline.py - Phase 3B Purely Local Refiner + LightGBM Pipeline.

Reproduces the Phase 3B Local Kaggle Winner (`[nested] PQ = 0.468754` across 707 OOF
images; 1,170 disjoint filaments across 180 test images) end-to-end without retraining:
  1. Stage 1 (Rule B Candidate Pool): 30-model, 120-view Mask R-CNN WMF (`sc >= 0.20`).
  2. Full-Disk Semantic Priors (4-flip TTA, 5-fold ensemble):
       - ResNet50SemanticUNet (1024x1024, 2ch: fg, core)
       - ResNeXt101SemanticUNet (1280x1280, 3ch: fg, core, edge)
  3. ROI Boundary Refiners (4-TTA, 5-fold ensemble):
       - ResidualCropRefinerUNet (160x160 u_box crop, 6ch + 4D scale, 4 H/V flips)
       - WideContextRefinerUNetV2 (224x224 w_box pad=56, 11ch + ASPP + SE, 4-TTA)
  4. Ranker & Disjoint Post-Processing:
       - 25 cross-fitted LightGBM boosters (`LGB_S27_yhyb`, 27 features, `tau = 0.255`)
       - `wref_uv1_t50` boundary blending + component cleaning + NMS (`0.03`)
       - Greedy pixel-disjoint 2048x2048 COCO RLE painting.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnext101_32x8d

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    EMPTY_RLE,
    extract_state_dict,
    find_dataset_paths,
    get_autocast_context,
    get_maskrcnn_model,
    preprocess_solar_image,
)
from scripts.instance_scorer import encode_crop_to_rle
from scripts.multifamily_pipeline import (
    FRAME_SIZE,
    compute_binary_crop_iou,
    fast_crop_mask_iou,
)

CROP_SZ_V1 = 160
CROP_SZ_WIDE = 224
PAD_WIDE = 56

FAM_NAMES = ["clahe", "anchors5", "dice", "2048", "a5dice", "1792_clahe"]
RULE_B_WEIGHTS = {
    "clahe": 1.0,
    "anchors5": 1.0,
    "dice": 1.0,
    "a5dice": 1.0,
    "2048": 2.0,
    "1792_clahe": 2.0,
}

S27_FEATURES = [
    "member_score_mean",
    "hires_ratio",
    "prob_mean",
    "prob_p25",
    "prob_core_frac",
    "prob_border_ratio",
    "log_area",
    "bbox_fill",
    "aspect_ratio",
    "n_comp",
    "largest_comp_frac",
    "solidity",
    "ring_contrast_mean",
    "ring_zscore",
    "ring_contrast_core",
    "radial_pos",
    "unoccluded_area_ratio",
    "max_iou_higher_rank",
    "rel_score_to_max",
    "rank_in_image",
    "cnn_iou_score",
    "cnn_mask_iou55",
    "cnn_area_ratio",
    "unet_mean",
    "unet_core",
    "unet_iou55",
    "unet_iou_cnn",
]


# ────────────────────────── ARCHITECTURES ───────────────────────────────────

class ConvRelu(nn.Module):
    """Two 3x3 Conv-BN-SiLU layers used in full-disk semantic U-Net decoders."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet50SemanticUNet(nn.Module):
    """5-fold full-disk semantic U-Net (1024x1024 -> 2 channels: fg, core)."""

    def __init__(self, out_ch: int = 2):
        super().__init__()
        base = resnet50(weights=None)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool0 = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.up3 = ConvRelu(2048 + 1024, 256)
        self.up2 = ConvRelu(256 + 512, 128)
        self.up1 = ConvRelu(128 + 256, 64)
        self.up0 = ConvRelu(64 + 64, 32)
        self.head = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        xn = (x - mean) / std

        e0 = self.enc0(xn)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.up3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        d0 = self.up0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0], dim=1))
        logits_half = self.head(d0)
        return F.interpolate(logits_half, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ResNeXt101SemanticUNet(nn.Module):
    """5-fold full-disk semantic U-Net (1280x1280 -> 3 channels: fg, core, edge)."""

    def __init__(self, out_ch: int = 3):
        super().__init__()
        base = resnext101_32x8d(weights=None)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool0 = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.up3 = ConvRelu(2048 + 1024, 256)
        self.up2 = ConvRelu(256 + 512, 128)
        self.up1 = ConvRelu(128 + 256, 64)
        self.up0 = ConvRelu(64 + 64, 32)
        self.refine_full = nn.Sequential(
            nn.Conv2d(32 + 3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=1),
        )

    def _decode_and_head(self, xn, e0, e1, e2, e3, e4) -> torch.Tensor:
        d3 = self.up3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        d0 = self.up0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0], dim=1))
        d_full = F.interpolate(d0, size=xn.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine_full(torch.cat([d_full, xn], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        xn = (x - mean) / std
        e0 = self.enc0(xn)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self._decode_and_head(xn, e0, e1, e2, e3, e4)


class ConvBlock(nn.Module):
    """Conv-BN-SiLU pair used inside ResidualCropRefinerUNet."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualCropRefinerUNet(nn.Module):
    """Scale-conditioned 160x160 ROI crop residual refiner + IoU quality head."""

    def __init__(self, in_ch: int = 6, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch + 2, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bot = ConvBlock(base * 4, base * 8)

        # Preserve exact attribute capitalization (`Scale_proj`) for checkpoint state_dict parity
        self.Scale_proj = nn.Sequential(
            nn.Linear(4, base * 8),
            nn.SiLU(inplace=True),
        )

        self.up3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.up2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.up1 = ConvBlock(base * 2 + base, base)

        self.delta_head = nn.Conv2d(base, 1, kernel_size=1)
        self.iou_head = nn.Sequential(
            nn.Linear(base * 8 + base + 4, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_sz, _, h, w = x.shape
        s_map = s[:, :2].view(b_sz, 2, 1, 1).expand(b_sz, 2, h, w)
        x_in = torch.cat([x, s_map], dim=1)

        e1 = self.enc1(x_in)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bot(F.avg_pool2d(e3, 2)) + self.Scale_proj(s).view(b_sz, -1, 1, 1)

        d3 = self.up3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))

        u_prob = torch.clamp(x[:, 0:1], 1e-3, 1.0 - 1e-3)
        base_logit = torch.log(u_prob / (1.0 - u_prob))
        delta = 3.5 * torch.tanh(self.delta_head(d1) / 3.5)
        out_logit = base_logit + delta

        p_out = torch.sigmoid(out_logit)
        b_pool = F.adaptive_avg_pool2d(b, 1).flatten(1)
        d_pool = (d1 * p_out).sum(dim=(-2, -1)) / (p_out.sum(dim=(-2, -1)) + 1.0)
        iou_logit = self.iou_head(torch.cat([b_pool, d_pool, s], dim=1)).squeeze(-1)
        iou_pred = torch.sigmoid(iou_logit)
        return out_logit.squeeze(1), iou_pred


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(8, c // r)),
            nn.SiLU(inplace=True),
            nn.Linear(max(8, c // r), c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class ResSEBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, in_c: int, out_c: int, dilation: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.se = SEBlock(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.conv(x)) + self.proj(x))


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling bottleneck for wide-context filament continuity."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        bc = out_c // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_c, bc, 1, bias=False), nn.BatchNorm2d(bc), nn.SiLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_c, bc, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(bc), nn.SiLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, bc, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(bc), nn.SiLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(in_c, bc, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(bc), nn.SiLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(bc * 4, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class WideContextRefinerUNetV2(nn.Module):
    """11-channel wide-context (pad=56, 224x224) Residual + SE + ASPP Refiner V2."""

    def __init__(self, in_ch: int = 11, base: int = 40):
        super().__init__()
        self.enc1 = ResSEBlock(in_ch + 2, base)
        self.enc2 = ResSEBlock(base, base * 2)
        self.enc3 = ResSEBlock(base * 2, base * 4)
        self.bot_r = ResSEBlock(base * 4, base * 6)
        self.aspp = ASPPModule(base * 6, base * 8)

        self.s_proj = nn.Sequential(nn.Linear(4, base * 8), nn.SiLU(inplace=True))
        self.up3 = ResSEBlock(base * 8 + base * 4, base * 4)
        self.up2 = ResSEBlock(base * 4 + base * 2, base * 2)
        self.up1 = ResSEBlock(base * 2 + base, base)

        self.delta_head = nn.Conv2d(base, 1, kernel_size=1)
        self.iou_head = nn.Sequential(
            nn.Linear(base * 8 + base + 4, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_sz, _, h, w = x.shape
        s_map = s[:, :2].view(b_sz, 2, 1, 1).expand(b_sz, 2, h, w)
        e1 = self.enc1(torch.cat([x, s_map], dim=1))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.aspp(self.bot_r(F.avg_pool2d(e3, 2))) + self.s_proj(s).view(b_sz, -1, 1, 1)

        d3 = self.up3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))

        base_p = torch.clamp(x[:, 0:1], 0.02, 0.99)
        base_logit = torch.log(base_p / (1.0 - base_p))
        out_logit = (base_logit + self.delta_head(d1)).squeeze(1)

        pb = F.adaptive_avg_pool2d(b, 1).flatten(1)
        pd1 = F.adaptive_avg_pool2d(d1, 1).flatten(1)
        iou_p = torch.sigmoid(self.iou_head(torch.cat([pb, pd1, s], dim=1)).squeeze(1))
        return out_logit, iou_p


# ────────────────────────── CHECKPOINT RESOLUTION ───────────────────────────

def _asset_candidate_paths(
    rel_or_name: str,
    weights_dir: Optional[Path] = None,
    strict_weights_only: bool = False,
) -> List[Path]:
    """Return ordered search locations for a Phase 3 asset."""
    name = Path(rel_or_name).name
    candidates: List[Path] = []
    if weights_dir is not None:
        w = Path(weights_dir)
        candidates.extend([w / rel_or_name, w / "phase3" / name, w / name])
    if strict_weights_only:
        if weights_dir is None:
            candidates.extend([
                REPO_ROOT / "weights" / rel_or_name,
                REPO_ROOT / "weights" / "phase3" / name,
                REPO_ROOT / "weights" / name,
            ])
        return candidates

    upstream_dir = Path(os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar")))
    candidates.extend([
        REPO_ROOT / "weights" / rel_or_name,
        REPO_ROOT / "weights" / "phase3" / name,
        REPO_ROOT / "weights" / name,
        REPO_ROOT / "work" / "checkpoints" / name,
        upstream_dir / "work" / "checkpoints" / name,
        upstream_dir / "work" / "s20_gpu" / "release_pkg" / name,
        upstream_dir / "work" / "s20_gpu" / name,
    ])
    return candidates


def resolve_phase3_asset_paths(
    rel_or_name: str,
    weights_dir: Optional[Path] = None,
    strict_weights_only: bool = False,
) -> List[Path]:
    """Locate a single Phase 3 asset or its ordered lossless ``.partNN`` binary shards."""
    candidates = _asset_candidate_paths(rel_or_name, weights_dir=weights_dir, strict_weights_only=strict_weights_only)
    for cand in candidates:
        if cand.is_file():
            return [cand]
        if cand.parent.is_dir():
            shards = sorted(cand.parent.glob(f"{cand.name}.part*"))
            if shards and all(s.is_file() for s in shards):
                return shards
    raise FileNotFoundError(
        f"Missing Phase 3 asset '{rel_or_name}' (strict={strict_weights_only}). "
        f"Checked: {[str(c) for c in candidates[:6]]}."
    )


def resolve_phase3_asset(
    rel_or_name: str,
    weights_dir: Optional[Path] = None,
    strict_weights_only: bool = False,
) -> Path:
    """Locate a single-file Phase 3 asset (booster, metadata, or unsharded checkpoint)."""
    paths = resolve_phase3_asset_paths(rel_or_name, weights_dir=weights_dir, strict_weights_only=strict_weights_only)
    if len(paths) != 1 or ".part" in paths[0].name:
        raise FileNotFoundError(
            f"Asset '{rel_or_name}' is sharded across {len(paths)} parts; use load_phase3_state_dict()."
        )
    return paths[0]


def _assert_fp16_state_dict(ckpt: Any, label: str) -> None:
    """Assert raw checkpoint tensors contain float16 floating-point weights prior to FP32 upcast."""
    raw_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    float_dtypes = {
        v.dtype for v in raw_sd.values()
        if isinstance(v, torch.Tensor) and v.is_floating_point()
    }
    if torch.float16 not in float_dtypes:
        raise ValueError(f"Expected float16 tensors in '{label}', got dtypes: {sorted(map(str, float_dtypes))}")


def load_phase3_state_dict(
    rel_or_name: str,
    weights_dir: Optional[Path] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict_weights_only: bool = False,
    verify_fp16: bool = False,
    loaded_log: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a Phase 3 state dict from a single `.pt` or concatenated `.partNN` shards."""
    paths = resolve_phase3_asset_paths(rel_or_name, weights_dir=weights_dir, strict_weights_only=strict_weights_only)
    if strict_weights_only and weights_dir is not None:
        root = Path(weights_dir).resolve()
        for p in paths:
            if not p.resolve().is_relative_to(root):
                raise RuntimeError(f"Strict weights violation: {p} is outside {root}")

    if len(paths) == 1:
        ckpt = torch.load(paths[0], map_location=map_location, weights_only=False)
    else:
        buf = io.BytesIO()
        for shard in paths:
            buf.write(shard.read_bytes())
        buf.seek(0)
        ckpt = torch.load(buf, map_location=map_location, weights_only=False)

    if verify_fp16:
        _assert_fp16_state_dict(ckpt, rel_or_name)
    sd = extract_state_dict(ckpt)
    if loaded_log is not None:
        loaded_log.extend(str(p.resolve()) for p in paths)
    return sd


def load_lgb_boosters(
    weights_dir: Optional[Path] = None,
    meta_filename: str = "lgb_metadata.json",
    strict_weights_only: bool = False,
    loaded_log: Optional[List[str]] = None,
) -> Tuple[List[lgb.Booster], List[str]]:
    """Load all 25 cross-fitted LightGBM boosters (`5 seeds x 5 folds`) and feature names."""
    meta_path = resolve_phase3_asset(
        f"phase3/{meta_filename}", weights_dir=weights_dir, strict_weights_only=strict_weights_only
    )
    if loaded_log is not None:
        loaded_log.append(str(meta_path.resolve()))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    features = list(meta["features"])
    seeds = list(meta["seeds"])
    boosters: List[lgb.Booster] = []
    for seed in seeds:
        for fold in range(5):
            b_name = f"lgb_s27_yhyb_s{seed}_f{fold}.txt"
            b_path = resolve_phase3_asset(
                f"phase3/{b_name}", weights_dir=weights_dir, strict_weights_only=strict_weights_only
            )
            if loaded_log is not None:
                loaded_log.append(str(b_path.resolve()))
            boosters.append(lgb.Booster(model_file=str(b_path)))
    return boosters, features


# ────────────────────────── STAGE-1 FUSION & S20 FEATURES ───────────────────

def fuse_rule_b_with_members(
    raw_cands: List[dict],
    fam_models_count: Dict[str, int],
    iou_thresh: float = 0.40,
) -> List[dict]:
    """Rule-B weighted mask fusion retaining member raw scores & family contributions."""
    if not raw_cands:
        return []
    eff_total = sum(RULE_B_WEIGHTS[f] * 4.0 * fam_models_count[f] for f in FAM_NAMES)
    w_cands = []
    for c in raw_cands:
        sc = float(c["score"])
        fam = c["family"]
        fw = RULE_B_WEIGHTS[fam]
        w_cands.append({
            "box": c["box"],
            "score": (sc ** 1.5) * fw,
            "raw_score": sc,
            "prob_crop": c["prob_crop"],
            "family": fam,
        })

    w_cands.sort(key=lambda x: x["score"], reverse=True)
    clusters: List[List[dict]] = []
    for cand in w_cands:
        best_idx = -1
        best_iou = -1.0
        for ci, cl in enumerate(clusters):
            iou = fast_crop_mask_iou(cand, cl[0])
            if iou > best_iou:
                best_iou = iou
                best_idx = ci
        if best_iou > iou_thresh:
            clusters[best_idx].append(cand)
        else:
            clusters.append([cand])

    fused: List[dict] = []
    for cl in clusters:
        w_scores = [float(x["score"]) for x in cl]
        w_sum = sum(w_scores)
        cluster_score = w_sum / float(eff_total)

        raw_scores = [float(x["raw_score"]) for x in cl]
        fam_raw_pow: Dict[str, float] = defaultdict(float)
        for x in cl:
            fam_raw_pow[x["family"]] += float(x["raw_score"]) ** 1.5
        fam_fracs = {
            f: fam_raw_pow[f] / float(max(1.0, 4.0 * fam_models_count[f]))
            for f in FAM_NAMES
        }

        all_b = np.stack([x["box"] for x in cl], axis=0)
        ux1 = int(np.floor(all_b[:, 0].min()))
        uy1 = int(np.floor(all_b[:, 1].min()))
        ux2 = int(np.ceil(all_b[:, 2].max()))
        uy2 = int(np.ceil(all_b[:, 3].max()))
        union_box = np.array([ux1, uy1, ux2, uy2], dtype=np.int32)
        fused_prob = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.float32)
        for x in cl:
            w_norm = float(x["score"]) / w_sum
            b = x["box"]
            fused_prob[b[1] - uy1:b[3] - uy1, b[0] - ux1:b[2] - ux1] += w_norm * x["prob_crop"].astype(np.float32)

        fused.append({
            "box": union_box,
            "score": float(cluster_score),
            "prob_crop": fused_prob,
            "raw_scores": raw_scores,
            "fam_fracs": fam_fracs,
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def extract_s20_pool_features(
    stem: str,
    fused: List[dict],
    img_gray: np.ndarray,
    min_cluster_score: float = 0.20,
) -> Tuple[List[dict], List[dict]]:
    """Extract candidate pool dicts (`u_box`, `u_prob`, etc.) and S20 tabular features."""
    all_scores = [float(c["score"]) for c in fused]
    max_score_in_img = max(all_scores) if all_scores else 1e-6

    sim_occupied = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    higher_rank_boxes_and_crops: List[Tuple[np.ndarray, np.ndarray]] = []

    feat_rows: List[dict] = []
    pool_cands: List[dict] = []

    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for rank_0, c in enumerate(fused):
        sc = float(c["score"])
        if sc < min_cluster_score:
            break

        prob_crop = c["prob_crop"]
        box = c["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        binary_crop = prob_crop >= 0.55
        area = int(binary_crop.sum())
        if area < 10:
            continue

        ys, xs = np.nonzero(binary_crop)
        tx1, ty1 = int(xs.min()), int(ys.min())
        tx2, ty2 = int(xs.max()) + 1, int(ys.max()) + 1
        bw = max(1, tx2 - tx1)
        bh = max(1, ty2 - ty1)
        tight_box = np.array([x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2], dtype=np.int32)
        tight_bin = binary_crop[ty1:ty2, tx1:tx2].copy()

        pad_u = 12
        px1 = max(0, tx1 - pad_u)
        py1 = max(0, ty1 - pad_u)
        px2 = min(x2 - x1, tx2 + pad_u)
        py2 = min(y2 - y1, ty2 + pad_u)
        u_box = np.array([x1 + px1, y1 + py1, x1 + px2, y1 + py2], dtype=np.int32)
        u_prob = prob_crop[py1:py2, px1:px2].astype(np.float16)

        masked_probs = prob_crop[binary_crop].astype(np.float32)
        prob_mean = float(masked_probs.mean())
        prob_p25 = float(np.percentile(masked_probs, 25))
        prob_core_frac = float((masked_probs >= 0.75).sum()) / float(area)
        area_45 = max(1, int((prob_crop >= 0.45).sum()))
        prob_border_ratio = float(area_45) / float(area)

        num_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(tight_bin.astype(np.uint8), connectivity=8)
        n_comp = max(1, num_labels - 1)
        if num_labels > 1:
            comp_areas = stats_cc[1:, cv2.CC_STAT_AREA]
            largest_comp_frac = float(comp_areas.max()) / float(area)
        else:
            largest_comp_frac = 1.0

        pts = np.column_stack([xs, ys]).astype(np.int32)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            hull_area = max(float(area), float(cv2.contourArea(hull)))
            solidity = float(area) / hull_area
        else:
            solidity = 1.0

        log_area = float(np.log10(area + 1.0))
        bbox_fill = float(area) / float(bw * bh)
        aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))

        ff = c["fam_fracs"]
        hires_sum = float(ff["2048"] + ff["1792_clahe"])
        tot_fsum = float(sum(ff[f] for f in FAM_NAMES))
        hires_ratio = hires_sum / max(1e-6, tot_fsum)

        m_arr = np.asarray(c["raw_scores"], dtype=np.float32)
        member_score_mean = float(m_arr.mean())

        pad = 24
        gx0, gy0 = max(0, x1 + tx1 - pad), max(0, y1 + ty1 - pad)
        gx1, gy1 = min(FRAME_SIZE, x1 + tx2 + pad), min(FRAME_SIZE, y1 + ty2 + pad)
        sub_mask = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
        sub_mask[(y1 + ty1 - gy0):(y1 + ty2 - gy0), (x1 + tx1 - gx0):(x1 + tx2 - gx0)] = tight_bin.astype(np.uint8)
        dil_outer = cv2.dilate(sub_mask, k_ring) > 0
        dil_inner = cv2.dilate(sub_mask, k_inner) > 0
        sub_img = img_gray[gy0:gy1, gx0:gx1].astype(np.float32)
        fg_vals = sub_img[sub_mask > 0]
        bg_ring = dil_outer & (~dil_inner) & (sub_img >= 15.0)
        bg_vals = sub_img[bg_ring]
        if len(bg_vals) >= 20 and len(fg_vals) > 0:
            bg_mean = float(bg_vals.mean())
            bg_std = float(bg_vals.std())
            bg_p50 = float(np.median(bg_vals))
            fg_mean = float(fg_vals.mean())
            fg_p25 = float(np.percentile(fg_vals, 25))
            ring_contrast_mean = (bg_mean - fg_mean) / max(1.0, bg_mean)
            ring_zscore = (bg_mean - fg_mean) / max(1.0, bg_std)
            ring_contrast_core = (bg_p50 - fg_p25) / max(1.0, bg_p50)
        else:
            ring_contrast_mean = 0.0
            ring_zscore = 0.0
            ring_contrast_core = 0.0

        cx = float(x1 + xs.mean())
        cy = float(y1 + ys.mean())
        radial_pos = float(np.hypot(cx - 1024.0, cy - 1024.0) / 1024.0)

        sub_occ = sim_occupied[tight_box[1]:tight_box[3], tight_box[0]:tight_box[2]]
        unocc_area = int((tight_bin & (~sub_occ)).sum())
        unoccluded_area_ratio = float(unocc_area) / float(area)
        max_iou_higher = 0.0
        for h_box, h_crop in higher_rank_boxes_and_crops:
            iou_h = compute_binary_crop_iou(tight_box, tight_bin, h_box, h_crop)
            if iou_h > max_iou_higher:
                max_iou_higher = iou_h
        if sc >= 0.50 and unoccluded_area_ratio >= 0.50:
            sim_occupied[tight_box[1]:tight_box[3], tight_box[0]:tight_box[2]] |= (tight_bin & (~sub_occ))
        higher_rank_boxes_and_crops.append((tight_box, tight_bin))

        rel_score_to_max = sc / max(1e-6, max_score_in_img)

        feat_rows.append({
            "obs": stem,
            "cand_idx": len(pool_cands),
            "cluster_score": sc,
            "member_score_mean": member_score_mean,
            "hires_ratio": hires_ratio,
            "prob_mean": prob_mean,
            "prob_p25": prob_p25,
            "prob_core_frac": prob_core_frac,
            "prob_border_ratio": prob_border_ratio,
            "log_area": log_area,
            "bbox_fill": bbox_fill,
            "aspect_ratio": aspect_ratio,
            "n_comp": float(n_comp),
            "largest_comp_frac": largest_comp_frac,
            "solidity": solidity,
            "ring_contrast_mean": ring_contrast_mean,
            "ring_zscore": ring_zscore,
            "ring_contrast_core": ring_contrast_core,
            "radial_pos": radial_pos,
            "unoccluded_area_ratio": unoccluded_area_ratio,
            "max_iou_higher_rank": max_iou_higher,
            "rel_score_to_max": rel_score_to_max,
            "rank_in_image": float(rank_0 + 1),
        })
        pool_cands.append({
            "box": tight_box,
            "binary_55": tight_bin,
            "prob_crop": prob_crop[ty1:ty2, tx1:tx2].astype(np.float16),
            "u_box": u_box,
            "u_prob": u_prob,
            "orig_area": area,
            "rule_b_score": sc,
        })

    return feat_rows, pool_cands


# ────────────────────────── SEMANTIC + REFINER FEATURE EXTRACTION ───────────

def make_clahe_uint8(img_gray: np.ndarray, sz: int) -> np.ndarray:
    """Resize and stack [linear, CLAHE(2, 8x8), CLAHE(4, 16x16)] uint8 channels."""
    img_r = cv2.resize(img_gray, (sz, sz), interpolation=cv2.INTER_AREA)
    c1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_r)
    c2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(img_r)
    return np.stack([img_r, c1, c2], axis=0)


def predict_4tta_semantic(
    model: nn.Module,
    img_u8: np.ndarray,
    device: torch.device,
    out_ch: int,
) -> np.ndarray:
    """Run 4-flip TTA on a full-disk semantic U-Net in strict float32."""
    sz = int(img_u8.shape[-1])
    x_base = torch.from_numpy(img_u8).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
    acc = torch.zeros((out_ch, sz, sz), device=device, dtype=torch.float32)
    with torch.no_grad():
        for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
            tx = x_base
            if fh:
                tx = torch.flip(tx, dims=[3])
            if fv:
                tx = torch.flip(tx, dims=[2])
            lg = model(tx)[0]
            p = torch.sigmoid(lg.float())
            if fv:
                p = torch.flip(p, dims=[1])
            if fh:
                p = torch.flip(p, dims=[2])
            acc += p / 4.0
    return acc.cpu().numpy()


def extract_crop_refiner_inputs(
    cands: List[dict],
    img_gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build `(N, 6, 160, 160)` float16 inputs and `(N, 4)` float32 scale vectors."""
    if not cands:
        return np.zeros((0, 6, CROP_SZ_V1, CROP_SZ_V1), dtype=np.float16), np.zeros((0, 4), dtype=np.float32)

    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clahe_op = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe_op.apply(img_gray)

    tensors_list = []
    scales_list = []
    for ci, c in enumerate(cands):
        ub = c["u_box"]
        ux1, uy1, ux2, uy2 = int(ub[0]), int(ub[1]), int(ub[2]), int(ub[3])
        uh, uw = max(1, uy2 - uy1), max(1, ux2 - ux1)
        u_prob = c["u_prob"].astype(np.float32)

        bin55 = (u_prob >= 0.55).astype(np.uint8)
        if bin55.any() and not bin55.all():
            d_in = cv2.distanceTransform(bin55, cv2.DIST_L2, 3)
            d_out = cv2.distanceTransform(1 - bin55, cv2.DIST_L2, 3)
            sdist = (d_in - d_out) / 6.0
        elif bin55.all():
            sdist = np.ones_like(u_prob, dtype=np.float32)
        else:
            sdist = -np.ones_like(u_prob, dtype=np.float32)
        sdist = np.clip(sdist, -1.5, 1.5)

        other_p = np.zeros_like(u_prob, dtype=np.float32)
        for cj, c_other in enumerate(cands):
            if ci == cj or c_other["rule_b_score"] < 0.35:
                continue
            ob = c_other["u_box"]
            ix1, iy1 = max(ux1, int(ob[0])), max(uy1, int(ob[1]))
            ix2, iy2 = min(ux2, int(ob[2])), min(uy2, int(ob[3]))
            if ix2 > ix1 and iy2 > iy1:
                sub_o = c_other["u_prob"][iy1 - int(ob[1]):iy2 - int(ob[1]), ix1 - int(ob[0]):ix2 - int(ob[0])].astype(np.float32)
                cur = other_p[iy1 - uy1:iy2 - uy1, ix1 - ux1:ix2 - ux1]
                np.maximum(cur, sub_o, out=cur)

        pad = 24
        gx0, gy0 = max(0, ux1 - pad), max(0, uy1 - pad)
        gx1, gy1 = min(FRAME_SIZE, ux2 + pad), min(FRAME_SIZE, uy2 + pad)
        sub_m = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
        sub_m[(uy1 - gy0):(uy2 - gy0), (ux1 - gx0):(ux2 - gx0)] = bin55
        sub_i = img_gray[gy0:gy1, gx0:gx1].astype(np.float32)
        bg = (cv2.dilate(sub_m, k_ring) > 0) & (~(cv2.dilate(sub_m, k_inner) > 0)) & (sub_i >= 15.0)
        if bg.sum() >= 20:
            bg_v = sub_i[bg]
            bg_med = float(np.median(bg_v))
            bg_std = max(2.0, float(bg_v.std()))
        else:
            bg_med = float(np.median(sub_i))
            bg_std = max(2.0, float(sub_i.std()))

        crop_i = img_gray[uy1:uy2, ux1:ux2].astype(np.float32)
        dark_z = np.clip((bg_med - crop_i) / bg_std / 3.0, -1.5, 1.5)
        crop_cl = (img_clahe[uy1:uy2, ux1:ux2].astype(np.float32) / 255.0) - 0.5

        gx = cv2.Sobel(crop_i, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop_i, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.clip(np.hypot(gx, gy) / (bg_std * 4.0), 0.0, 1.5)

        raw_stack = np.stack([u_prob, sdist, other_p, dark_z, crop_cl, grad], axis=-1)
        resized = cv2.resize(raw_stack, (CROP_SZ_V1, CROP_SZ_V1), interpolation=cv2.INTER_LINEAR)
        tensors_list.append(np.transpose(resized, (2, 0, 1)).astype(np.float16))
        scales_list.append(np.array([
            math.log2(max(1.0, uw) / 64.0),
            math.log2(max(1.0, uh) / 64.0),
            math.log10(c["orig_area"] + 1.0) - 3.0,
            float(c["rule_b_score"]),
        ], dtype=np.float32))

    return np.stack(tensors_list, axis=0), np.stack(scales_list, axis=0)


def quantize_11ch(arr_11hw: np.ndarray) -> np.ndarray:
    """Quantize 11-channel wide crop features into uint8 for memory-efficient inference parity."""
    q = np.zeros_like(arr_11hw, dtype=np.uint8)
    for ch in (0, 2, 3, 4, 5, 6, 7, 10):
        q[ch] = np.clip(np.round(arr_11hw[ch] * 255.0), 0, 255).astype(np.uint8)
    q[1] = np.clip(np.round((arr_11hw[1] + 1.0) * 127.5), 0, 255).astype(np.uint8)
    q[8] = np.clip(np.round((arr_11hw[8] + 1.5) / 3.0 * 255.0), 0, 255).astype(np.uint8)
    q[9] = np.clip(np.round((arr_11hw[9] + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return q


def dequantize_11ch(x_u8: torch.Tensor) -> torch.Tensor:
    """Invert `quantize_11ch` on CPU or CUDA."""
    x = x_u8.float() / 255.0
    return torch.cat([
        x[:, 0:1],
        x[:, 1:2] * 2.0 - 1.0,
        x[:, 2:8],
        x[:, 8:9] * 3.0 - 1.5,
        x[:, 9:10] - 0.5,
        x[:, 10:11],
    ], dim=1)


def extract_wide_refiner_inputs(
    cands: List[dict],
    img_gray: np.ndarray,
    r50_2048: np.ndarray,
    rnx_2048: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(N, 11, 224, 224)` uint8 inputs, `(N, 4)` float32 scales, and `(N, 4)` `w_box`."""
    if not cands:
        return (
            np.zeros((0, 11, CROP_SZ_WIDE, CROP_SZ_WIDE), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.int32),
        )

    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    img_cl = clahe.apply(img_gray)

    x_list = []
    s_list = []
    w_boxes = []
    for ci, c in enumerate(cands):
        ub = c["u_box"]
        ux1, uy1, ux2, uy2 = int(ub[0]), int(ub[1]), int(ub[2]), int(ub[3])
        wx1 = max(0, ux1 - PAD_WIDE)
        wy1 = max(0, uy1 - PAD_WIDE)
        wx2 = min(FRAME_SIZE, ux2 + PAD_WIDE)
        wy2 = min(FRAME_SIZE, uy2 + PAD_WIDE)
        ww, wh = wx2 - wx1, wy2 - wy1
        wb = np.array([wx1, wy1, wx2, wy2], dtype=np.int32)
        c["w_box"] = wb
        w_boxes.append(wb)

        u_wide = np.zeros((wh, ww), dtype=np.float32)
        u_wide[(uy1 - wy1):(uy2 - wy1), (ux1 - wx1):(ux2 - wx1)] = c["u_prob"].astype(np.float32)

        in_ub = np.zeros((wh, ww), dtype=np.float32)
        in_ub[(uy1 - wy1):(uy2 - wy1), (ux1 - wx1):(ux2 - wx1)] = 1.0

        bin55_w = (u_wide >= 0.55).astype(np.uint8)
        if bin55_w.any():
            d_in = cv2.distanceTransform(bin55_w, cv2.DIST_L2, 3)
            d_out = cv2.distanceTransform(1 - bin55_w, cv2.DIST_L2, 3)
            sdist = np.clip((d_in - d_out) / 14.0, -1.0, 1.0)
        else:
            sdist = np.full_like(u_wide, -1.0)

        other_hi = np.zeros((wh, ww), dtype=np.float32)
        sc_i = float(c["rule_b_score"])
        for cj, co in enumerate(cands):
            if cj == ci or float(co["rule_b_score"]) < 0.40:
                continue
            if float(co["rule_b_score"]) < sc_i * 0.75:
                continue
            ob = co["u_box"]
            ix1, iy1 = max(wx1, int(ob[0])), max(wy1, int(ob[1]))
            ix2, iy2 = min(wx2, int(ob[2])), min(wy2, int(ob[3]))
            if ix2 > ix1 and iy2 > iy1:
                sub_o = co["u_prob"][iy1 - int(ob[1]):iy2 - int(ob[1]), ix1 - int(ob[0]):ix2 - int(ob[0])].astype(np.float32)
                cur = other_hi[iy1 - wy1:iy2 - wy1, ix1 - wx1:ix2 - wx1]
                np.maximum(cur, sub_o, out=cur)

        pad_bg = 24
        gx0, gy0 = max(0, wx1 - pad_bg), max(0, wy1 - pad_bg)
        gx1, gy1 = min(FRAME_SIZE, wx2 + pad_bg), min(FRAME_SIZE, wy2 + pad_bg)
        sub_m = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
        sub_m[(wy1 - gy0):(wy2 - gy0), (wx1 - gx0):(wx2 - gx0)] = bin55_w
        sub_i = img_gray[gy0:gy1, gx0:gx1].astype(np.float32)
        bg = (cv2.dilate(sub_m, k_ring) > 0) & (~(cv2.dilate(sub_m, k_inner) > 0)) & (sub_i >= 15.0)
        if bg.sum() >= 20:
            bg_v = sub_i[bg]
            bg_med, bg_std = float(np.median(bg_v)), max(2.0, float(bg_v.std()))
        else:
            bg_med, bg_std = float(np.median(sub_i)), max(2.0, float(sub_i.std()))

        crop_i = img_gray[wy1:wy2, wx1:wx2].astype(np.float32)
        dark_z = np.clip((bg_med - crop_i) / (bg_std * 3.0), -1.5, 1.5)
        crop_cl = (img_cl[wy1:wy2, wx1:wx2].astype(np.float32) / 255.0) - 0.5

        stack_11 = np.stack([
            u_wide, sdist, other_hi,
            r50_2048[0, wy1:wy2, wx1:wx2], r50_2048[1, wy1:wy2, wx1:wx2],
            rnx_2048[0, wy1:wy2, wx1:wx2], rnx_2048[1, wy1:wy2, wx1:wx2], rnx_2048[2, wy1:wy2, wx1:wx2],
            dark_z, crop_cl, in_ub,
        ], axis=-1)
        res = cv2.resize(stack_11, (CROP_SZ_WIDE, CROP_SZ_WIDE), interpolation=cv2.INTER_LINEAR)
        x_list.append(quantize_11ch(np.transpose(res, (2, 0, 1))))
        s_list.append(np.array([
            math.log2(max(1.0, ww) / 128.0),
            math.log2(max(1.0, wh) / 128.0),
            math.log10(c["orig_area"] + 1.0) - 3.0,
            sc_i,
        ], dtype=np.float32))

    return np.stack(x_list, axis=0), np.stack(s_list, axis=0), np.stack(w_boxes, axis=0)


def predict_4tta_crop_v1(
    model: ResidualCropRefinerUNet,
    x_np: np.ndarray,
    s_np: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 4 H/V-flip TTA on ResidualCropRefinerUNet in float32."""
    n = len(x_np)
    masks_out = np.zeros((n, CROP_SZ_V1, CROP_SZ_V1), dtype=np.float32)
    iou_out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bx = torch.from_numpy(x_np[i:i + batch_size].astype(np.float32)).to(device)
            bs = torch.from_numpy(s_np[i:i + batch_size]).to(device)
            pm_acc = torch.zeros((len(bx), CROP_SZ_V1, CROP_SZ_V1), device=device, dtype=torch.float32)
            pi_acc = torch.zeros(len(bx), device=device, dtype=torch.float32)
            for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
                tx = bx
                if fh:
                    tx = torch.flip(tx, dims=[3])
                if fv:
                    tx = torch.flip(tx, dims=[2])
                mlg, ip = model(tx, bs)
                pm = torch.sigmoid(mlg.float())
                if fv:
                    pm = torch.flip(pm, dims=[1])
                if fh:
                    pm = torch.flip(pm, dims=[2])
                pm_acc += pm / 4.0
                pi_acc += ip.float() / 4.0
            masks_out[i:i + batch_size] = pm_acc.cpu().numpy()
            iou_out[i:i + batch_size] = pi_acc.cpu().numpy()
    return masks_out, iou_out


def predict_4tta_wide_v2(
    model: WideContextRefinerUNetV2,
    x_u8: np.ndarray,
    s_np: np.ndarray,
    device: torch.device,
    batch_size: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 4-TTA (id, hflip, vflip, transpose w/ swapped scale) on WideContextRefinerUNetV2."""
    n = len(x_u8)
    masks_out = np.zeros((n, CROP_SZ_WIDE, CROP_SZ_WIDE), dtype=np.float32)
    iou_out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bx_u8 = torch.from_numpy(x_u8[i:i + batch_size]).to(device)
            bx = dequantize_11ch(bx_u8)
            bs = torch.from_numpy(s_np[i:i + batch_size]).to(device)

            p0, i0 = model(bx, bs)
            p0 = torch.sigmoid(p0.float())

            p1, i1 = model(torch.flip(bx, dims=[3]), bs)
            p1 = torch.flip(torch.sigmoid(p1.float()), dims=[2])

            p2, i2 = model(torch.flip(bx, dims=[2]), bs)
            p2 = torch.flip(torch.sigmoid(p2.float()), dims=[1])

            bs3 = bs.clone()
            bs3[:, 0], bs3[:, 1] = bs[:, 1], bs[:, 0]
            p3, i3 = model(torch.transpose(bx, 2, 3), bs3)
            p3 = torch.transpose(torch.sigmoid(p3.float()), 1, 2)

            masks_out[i:i + batch_size] = (0.25 * (p0 + p1 + p2 + p3)).cpu().numpy()
            iou_out[i:i + batch_size] = (0.25 * (i0.float() + i1.float() + i2.float() + i3.float())).cpu().numpy()
    return masks_out, iou_out


# ────────────────────────── MASK BLEND & DISJOINT PAINTING ──────────────────

def clean_components(bc: np.ndarray, bc_ref: np.ndarray) -> np.ndarray:
    """Remove spurious tiny satellite islands not overlapping Rule-B core (`binary_55`)."""
    if not bc.any():
        return bc
    nl, labels, stats_cc, _ = cv2.connectedComponentsWithStats(bc.astype(np.uint8), connectivity=8)
    if nl <= 2:
        return bc
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    max_a = int(areas.max())
    keep = np.zeros(nl, dtype=bool)
    for lbl_i in range(1, nl):
        a_i = int(stats_cc[lbl_i, cv2.CC_STAT_AREA])
        if a_i == max_a or (a_i >= 15 and ((a_i >= 0.05 * max_a) or (bc_ref & (labels == lbl_i)).any())):
            keep[lbl_i] = True
    return keep[labels]


def compute_wref_uv1_mask(cand: dict, tau_mask: float = 0.50) -> Tuple[np.ndarray, np.ndarray]:
    """Blend CropRefinerV1, WideRefinerV2, and ResNet50 Semantic UNet inside `u_box` + wide halo."""
    ub = cand["u_box"]
    ux1, uy1, ux2, uy2 = int(ub[0]), int(ub[1]), int(ub[2]), int(ub[3])
    wb = cand["w_box"]
    wx1, wy1, wx2, wy2 = int(wb[0]), int(wb[1]), int(wb[2]), int(wb[3])
    ww, wh = wx2 - wx1, wy2 - wy1

    b55_u = cand["u_prob"].astype(np.float32) >= 0.55
    b55_w = np.zeros((wh, ww), dtype=bool)
    b55_w[(uy1 - wy1):(uy2 - wy1), (ux1 - wx1):(ux2 - wx1)] = b55_u

    p_wref = cand["wref_prob"].astype(np.float32)
    p_comb = p_wref.copy()

    oy0, oy1 = uy1 - wy1, uy2 - wy1
    ox0, ox1 = ux1 - wx1, ux2 - wx1
    cnn_u = cand["cnn_u_prob"].astype(np.float32)
    unet_u = cand["unet_u_prob"].astype(np.float32)
    p_comb[oy0:oy1, ox0:ox1] = 0.45 * cnn_u + 0.42 * p_wref[oy0:oy1, ox0:ox1] + 0.13 * unet_u
    return wb, clean_components(p_comb >= tau_mask, b55_w)


def materialize_disjoint_rows_phase3b(
    stems: Sequence[str],
    pool_by_stem: Dict[str, List[dict]],
    tau: float = 0.255,
    nms_iou: float = 0.03,
    tau_mask: float = 0.50,
) -> Tuple[pd.DataFrame, Dict]:
    """Sort candidates by LightGBM score, apply NMS & retention, paint pixel-disjoint RLEs."""
    rows = []
    areas = []
    empty_count = 0
    total_filaments = 0

    for stem in stems:
        cands = sorted(pool_by_stem.get(stem, []), key=lambda x: x["lgb_score"], reverse=True)
        occ_sel = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        occ_pnt = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        acc: List[Tuple[np.ndarray, np.ndarray]] = []
        k_out = 0

        for c in cands:
            if float(c["lgb_score"]) < tau:
                break
            b = c["box"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            bc55 = c["binary_55"]
            oa = int(c["orig_area"])
            if oa < 10:
                continue

            dup = False
            for ab, ac in acc:
                if min(x2, int(ab[2])) > max(x1, int(ab[0])) and min(y2, int(ab[3])) > max(y1, int(ab[1])):
                    if compute_binary_crop_iou(b, bc55, ab, ac) > nms_iou:
                        dup = True
                        break
            if dup:
                continue

            clean55 = bc55 & (~occ_sel[y1:y2, x1:x2])
            if int(clean55.sum()) < 10 or (int(clean55.sum()) / float(oa)) < 0.50:
                continue
            occ_sel[y1:y2, x1:x2] |= clean55
            acc.append((b, bc55))

            mb, bm = compute_wref_uv1_mask(c, tau_mask=tau_mask)
            mx1, my1, mx2, my2 = int(mb[0]), int(mb[1]), int(mb[2]), int(mb[3])
            clean_p = bm & (~occ_pnt[my1:my2, mx1:mx2])
            area_p = int(clean_p.sum())
            if area_p >= 10:
                occ_pnt[my1:my2, mx1:mx2] |= clean_p
                rows.append({
                    "filament_id": f"{stem}_{k_out + 1}",
                    "segmentation_rle": encode_crop_to_rle(mb, clean_p),
                })
                areas.append(area_p)
                k_out += 1
                total_filaments += 1

        if k_out == 0:
            rows.append({"filament_id": f"{stem}_1", "segmentation_rle": EMPTY_RLE})
            empty_count += 1

    df_sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    stats = {
        "rule": "C",
        "mode": "phase3b_local",
        "num_images": len(stems),
        "total_rows": len(df_sub),
        "total_filaments": total_filaments,
        "empty_images": empty_count,
        "median_area": float(np.median(areas)) if areas else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "tau": float(tau),
    }
    return df_sub, stats


# ────────────────────────── END-TO-END ORCHESTRATION ────────────────────────

def resolve_stage1_checkpoint_path(
    model_spec: dict,
    weights_dir: Path,
    strict_weights_only: bool = False,
) -> Path:
    """Resolve a Stage-1 Mask R-CNN checkpoint from repo weights or fallbacks."""
    subdir = model_spec["id"]
    pruned_name = model_spec.get("pruned_name", f"{subdir}.pt")
    run_relpath = model_spec.get("checkpoint_relpath", f"work/runs/{subdir}.pt")
    cand_paths = [weights_dir / pruned_name, weights_dir / f"{subdir}.pt"]
    if not strict_weights_only:
        upstream_root = os.environ.get("SOLAR_UPSTREAM_DIR", str(REPO_ROOT.parent / "solar"))
        cand_paths.extend([
            REPO_ROOT / "weights" / pruned_name,
            REPO_ROOT / "work" / "checkpoints" / pruned_name,
            REPO_ROOT / run_relpath,
            Path(upstream_root) / "work" / "checkpoints" / pruned_name,
            Path(upstream_root) / run_relpath,
        ])
    ckpt_path = next((p for p in cand_paths if p.is_file()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not locate checkpoint for {subdir} (strict={strict_weights_only}). "
            f"Checked: {[str(x) for x in cand_paths[:5]]}"
        )
    if strict_weights_only and not ckpt_path.resolve().is_relative_to(weights_dir.resolve()):
        raise RuntimeError(f"Strict weights violation: {ckpt_path} is outside {weights_dir}")
    return ckpt_path


def _load_stage1_model(
    model_spec: dict,
    weights_dir: Path,
    device: torch.device,
    strict_weights_only: bool = False,
    verify_fp16: bool = False,
    loaded_log: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, Path]:
    """Load one Stage-1 Mask R-CNN checkpoint into strict FP32 eval mode."""
    sz = int(model_spec["img_size"])
    is_a5 = bool(model_spec.get("anchors5", False))
    ckpt_path = resolve_stage1_checkpoint_path(
        model_spec, weights_dir=weights_dir, strict_weights_only=strict_weights_only
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if verify_fp16:
        _assert_fp16_state_dict(ckpt, ckpt_path.name)
    state_dict = extract_state_dict(ckpt)
    model = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=is_a5)
    model.load_state_dict(state_dict)
    model.float().to(device).eval()
    if loaded_log is not None:
        loaded_log.append(str(ckpt_path.resolve()))
    del ckpt, state_dict
    return model, ckpt_path


def _build_4tta_views(base_t: torch.Tensor, device: torch.device) -> List[Tuple[Tuple[str, bool, bool], torch.Tensor]]:
    """Precompute the 4 H/V flip views of a single-image tensor on ``device``."""
    base_dev = base_t.to(device)
    out: List[Tuple[Tuple[str, bool, bool], torch.Tensor]] = []
    for vname, f_v, f_h in (("id", False, False), ("h", False, True), ("v", True, False), ("hv", True, True)):
        t_in = base_dev
        if f_v:
            t_in = torch.flip(t_in, dims=[1])
        if f_h:
            t_in = torch.flip(t_in, dims=[2])
        out.append(((vname, f_v, f_h), t_in))
    return out


def _infer_stage1_4tta_single(
    model: torch.nn.Module,
    base_t: Optional[torch.Tensor],
    sz: int,
    fam: str,
    device: torch.device,
    gpu_views: Optional[List[Tuple[Tuple[str, bool, bool], torch.Tensor]]] = None,
) -> List[dict]:
    """Run 4-flip TTA Mask R-CNN inference on a single image tensor (no batch/neighbor mixing)."""
    views = gpu_views if gpu_views is not None else _build_4tta_views(base_t, device)
    scale_to_2048 = 2048.0 / float(sz)
    out_cands: List[dict] = []
    for (_, f_v, f_h), t_in in views:
        with get_autocast_context(device):
            out = model([t_in])[0]
        scores = out["scores"]
        keep = scores >= 0.25
        if not keep.any():
            continue
        scores_np = scores[keep].detach().cpu().numpy()
        boxes = out["boxes"][keep].clone()
        masks = out["masks"][keep]
        if f_v:
            boxes[:, [1, 3]] = float(sz) - boxes[:, [3, 1]]
            masks = torch.flip(masks, dims=[-2])
        if f_h:
            boxes[:, [0, 2]] = float(sz) - boxes[:, [2, 0]]
            masks = torch.flip(masks, dims=[-1])
        boxes_2048 = (boxes * scale_to_2048).detach().cpu().numpy()
        masks_sq = masks.squeeze(1)

        valid: List[Tuple[int, int, int, int, int]] = []
        for k in range(len(scores_np)):
            b = boxes_2048[k]
            x1 = max(0, int(np.floor(b[0])))
            y1 = max(0, int(np.floor(b[1])))
            x2 = min(FRAME_SIZE, int(np.ceil(b[2])) + 1)
            y2 = min(FRAME_SIZE, int(np.ceil(b[3])) + 1)
            if x2 > x1 and y2 > y1:
                valid.append((k, x1, y1, x2, y2))
        if not valid:
            continue

        # Upsample per-batch chunks (<=16 masks = <=256 MB) on GPU and coalesce
        # all ROI crops into a single FP16 device-to-host transfer per view.
        if sz == FRAME_SIZE:
            m_half = masks_sq.half()
            crops = [m_half[k, y1:y2, x1:x2] for k, x1, y1, x2, y2 in valid]
        else:
            crops = []
            for start in range(0, len(valid), 16):
                batch_valid = valid[start : start + 16]
                k_idxs = [v[0] for v in batch_valid]
                m_full = F.interpolate(
                    masks_sq[k_idxs].unsqueeze(1).float(),
                    size=(FRAME_SIZE, FRAME_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].half()
                for ii, (_, x1, y1, x2, y2) in enumerate(batch_valid):
                    crops.append(m_full[ii, y1:y2, x1:x2])

        crop_sizes = [c.numel() for c in crops]
        crop_shapes = [(c.shape[0], c.shape[1]) for c in crops]
        flat_cpu = torch.cat([c.reshape(-1) for c in crops]).detach().cpu().numpy()
        off = 0
        for (k, x1, y1, x2, y2), sz_c, sh_c in zip(valid, crop_sizes, crop_shapes):
            out_cands.append({
                "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                "score": float(scores_np[k]),
                "prob_crop": flat_cpu[off : off + sz_c].reshape(sh_c).copy(),
                "family": fam,
            })
            off += sz_c
    return out_cands


def _collect_stage1_candidates_from_weights(
    stems: Sequence[str],
    img_by_stem: Dict[str, np.ndarray],
    cfg: dict,
    weights_dir: Path,
    device: torch.device,
    strict_weights_only: bool = False,
    verify_fp16: bool = False,
    loaded_log: Optional[List[str]] = None,
    stage1_cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[dict]]:
    """Run all 30 Mask R-CNN checkpoints (4-flip TTA, strict FP32) with bounded host RAM."""
    models = cfg["models"]
    raw_by_stem: Dict[str, List[dict]] = {s: [] for s in stems}
    needed_sizes = sorted({int(m["img_size"]) for m in models})

    s1_cache_path = Path(stage1_cache_dir) if stage1_cache_dir else None
    if s1_cache_path is not None:
        s1_cache_path.mkdir(parents=True, exist_ok=True)
        pending_stems = [s for s in stems if not (s1_cache_path / f"{s}.pkl").is_file()]
        if len(pending_stems) < len(stems):
            print(
                f"  [Stage 1 Cache] Found {len(stems) - len(pending_stems)}/{len(stems)} cached stems; "
                f"{len(pending_stems)} remaining.",
                flush=True,
            )
    else:
        pending_stems = list(stems)

    can_preload_all = (
        device.type == "cuda"
        and torch.cuda.get_device_properties(device).total_memory >= 12 * (1024 ** 3)
    )

    if pending_stems:
        with torch.inference_mode():
            if can_preload_all:
                loaded = []
                for m_idx, m in enumerate(models, start=1):
                    model, ckpt_p = _load_stage1_model(
                        m, weights_dir=weights_dir, device=device,
                        strict_weights_only=strict_weights_only,
                        verify_fp16=verify_fp16, loaded_log=loaded_log,
                    )
                    loaded.append((m["id"], m["family"], int(m["img_size"]), model))
                    if m_idx % 10 == 0 or m_idx == len(models):
                        print(f"  [Stage 1 Load] {m_idx:02d}/{len(models)} models ready (last={ckpt_p.name})", flush=True)

                t_s1 = time.time()
                for img_idx, stem in enumerate(pending_stems, start=1):
                    raw_img = img_by_stem[stem]
                    views_by_sz = {
                        sz: _build_4tta_views(torch.from_numpy(preprocess_solar_image(raw_img, sz)).float(), device)
                        for sz in needed_sizes
                    }
                    stem_cands: List[dict] = []
                    for _, fam, sz, model in loaded:
                        stem_cands.extend(
                            _infer_stage1_4tta_single(model, None, sz=sz, fam=fam, device=device, gpu_views=views_by_sz[sz])
                        )
                    del views_by_sz
                    if s1_cache_path is not None:
                        tmp_p = s1_cache_path / f"{stem}.pkl.tmp"
                        with open(tmp_p, "wb") as f:
                            pickle.dump(stem_cands, f, protocol=pickle.HIGHEST_PROTOCOL)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_p, s1_cache_path / f"{stem}.pkl")
                        del stem_cands
                    else:
                        raw_by_stem[stem] = stem_cands
                    if img_idx % 10 == 0 or img_idx == len(pending_stems):
                        print(f"  [Stage 1 Infer] [{img_idx:03d}/{len(pending_stems):03d}] images ({time.time() - t_s1:.1f}s)", flush=True)

                del loaded
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Low-RAM / CPU fallback: group by input size in chunks of 30 stems
                chunk_sz = 30
                models_by_sz: Dict[int, List[dict]] = defaultdict(list)
                for m in models:
                    models_by_sz[int(m["img_size"])].append(m)
                for sz in needed_sizes:
                    sz_models = models_by_sz[sz]
                    for c_start in range(0, len(pending_stems), chunk_sz):
                        chunk_stems = pending_stems[c_start : c_start + chunk_sz]
                        prepped = {
                            s: torch.from_numpy(preprocess_solar_image(img_by_stem[s], sz)).float()
                            for s in chunk_stems
                        }
                        for m in sz_models:
                            log_target = loaded_log if c_start == 0 else None
                            model, _ = _load_stage1_model(
                                m, weights_dir=weights_dir, device=device,
                                strict_weights_only=strict_weights_only,
                                verify_fp16=verify_fp16, loaded_log=log_target,
                            )
                            for s in chunk_stems:
                                raw_by_stem[s].extend(
                                    _infer_stage1_4tta_single(model, prepped[s], sz=sz, fam=m["family"], device=device)
                                )
                            del model
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        del prepped
                        gc.collect()
                    print(f"  [Stage 1 Chunked] {sz}px ({len(sz_models)} models) complete.", flush=True)
                if s1_cache_path is not None:
                    for s in pending_stems:
                        tmp_p = s1_cache_path / f"{s}.pkl.tmp"
                        with open(tmp_p, "wb") as f:
                            pickle.dump(raw_by_stem[s], f, protocol=pickle.HIGHEST_PROTOCOL)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_p, s1_cache_path / f"{s}.pkl")
                        raw_by_stem[s] = []

    if s1_cache_path is not None:
        for s in stems:
            with open(s1_cache_path / f"{s}.pkl", "rb") as f:
                raw_by_stem[s] = pickle.load(f)

    return raw_by_stem


def run_phase3b_from_stage1(
    stems: Sequence[str],
    img_by_stem: Dict[str, np.ndarray],
    raw_by_stem: Dict[str, List[dict]],
    cfg: dict,
    w_dir: Path,
    device: torch.device,
    tau: float = 0.255,
    strict_weights_only: bool = False,
    verify_fp16: bool = False,
    loaded_log: Optional[List[str]] = None,
    t_start: Optional[float] = None,
    free_raw_after_pool: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Execute Phase 3B Stage 2+ (WMF pool -> UNets -> Refiners -> LGB -> RLE) from Stage-1 candidates."""
    t0 = t_start if t_start is not None else time.time()

    # 1. Rule-B fusion -> S20 pool
    fam_models_count: Dict[str, int] = defaultdict(int)
    for m in cfg["models"]:
        fam_models_count[m["family"]] += 1

    pool_by_stem: Dict[str, List[dict]] = {}
    rows_by_stem: Dict[str, List[dict]] = {}
    total_cands = 0
    for stem in stems:
        stem_raw = raw_by_stem.pop(stem) if free_raw_after_pool else raw_by_stem[stem]
        fused = fuse_rule_b_with_members(stem_raw, fam_models_count)
        f_rows, p_cands = extract_s20_pool_features(stem, fused, img_by_stem[stem], min_cluster_score=0.20)
        rows_by_stem[stem] = f_rows
        pool_by_stem[stem] = p_cands
        total_cands += len(p_cands)
    if free_raw_after_pool:
        raw_by_stem.clear()
        gc.collect()
    print(f"  [Pool Built] {total_cands} candidates across {len(stems)} stems ({time.time() - t0:.1f}s)", flush=True)

    # 2. 5-Fold ResNet50SemanticUNet (1024x1024, 2ch)
    n_imgs = len(stems)
    r50_accum = np.zeros((n_imgs, 2, 1024, 1024), dtype=np.float32)
    x_1024 = [make_clahe_uint8(img_by_stem[s], 1024) for s in stems]
    for fold in range(5):
        sd = load_phase3_state_dict(
            f"semantic_unet_f{fold}_fp16.pt", weights_dir=w_dir, map_location=device,
            strict_weights_only=strict_weights_only, verify_fp16=verify_fp16, loaded_log=loaded_log,
        )
        model = ResNet50SemanticUNet(out_ch=2)
        model.load_state_dict(sd)
        model.float().to(device).eval()
        for i in range(n_imgs):
            r50_accum[i] += predict_4tta_semantic(model, x_1024[i], device, out_ch=2) / 5.0
        del model, sd
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    r50_u8 = np.clip(np.round(r50_accum * 255.0), 0, 255).astype(np.uint8)
    del r50_accum, x_1024
    print(f"  [Semantic R50 UNet] 5 folds complete ({time.time() - t0:.1f}s)", flush=True)

    # 3. 5-Fold ResNeXt101SemanticUNet (1280x1280, 3ch, shard-aware)
    rnx_accum = np.zeros((n_imgs, 3, 1280, 1280), dtype=np.float32)
    x_1280 = [make_clahe_uint8(img_by_stem[s], 1280) for s in stems]
    for fold in range(5):
        sd = load_phase3_state_dict(
            f"resnext101_unet_f{fold}_fp16.pt", weights_dir=w_dir, map_location=device,
            strict_weights_only=strict_weights_only, verify_fp16=verify_fp16, loaded_log=loaded_log,
        )
        model = ResNeXt101SemanticUNet(out_ch=3)
        model.load_state_dict(sd)
        model.float().to(device).eval()
        for i in range(n_imgs):
            rnx_accum[i] += predict_4tta_semantic(model, x_1280[i], device, out_ch=3) / 5.0
        del model, sd
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rnx_u8 = np.clip(np.round(rnx_accum * 255.0), 0, 255).astype(np.uint8)
    del rnx_accum, x_1280
    print(f"  [Semantic Rnx101 UNet] 5 folds complete ({time.time() - t0:.1f}s)", flush=True)

    # 4. Prepare CropRefinerV1 (160x160) & WideRefinerV2 (224x224) inputs
    v1_x_parts, v1_s_parts, v2_x_parts, v2_s_parts, meta_pairs = [], [], [], [], []
    r50_crops_fg: List[np.ndarray] = []
    r50_crops_cr: List[np.ndarray] = []

    for i, stem in enumerate(stems):
        cands = pool_by_stem[stem]
        if not cands:
            continue
        r50_2048 = np.stack([
            cv2.resize(r50_u8[i, ch].astype(np.float32) / 255.0, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            for ch in range(2)
        ], axis=0)
        rnx_2048 = np.stack([
            cv2.resize(rnx_u8[i, ch].astype(np.float32) / 255.0, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            for ch in range(3)
        ], axis=0)

        vx1, vs1 = extract_crop_refiner_inputs(cands, img_by_stem[stem])
        vx2, vs2, _ = extract_wide_refiner_inputs(cands, img_by_stem[stem], r50_2048, rnx_2048)
        v1_x_parts.append(vx1)
        v1_s_parts.append(vs1)
        v2_x_parts.append(vx2)
        v2_s_parts.append(vs2)

        for ci, c in enumerate(cands):
            ub = c["u_box"]
            ux1, uy1, ux2, uy2 = int(ub[0]), int(ub[1]), int(ub[2]), int(ub[3])
            u_fg = r50_2048[0, uy1:uy2, ux1:ux2]
            u_cr = r50_2048[1, uy1:uy2, ux1:ux2]
            c["unet_u_prob"] = u_fg.astype(np.float16)
            r50_crops_fg.append(u_fg)
            r50_crops_cr.append(u_cr)
            meta_pairs.append((stem, ci))

    del r50_u8, rnx_u8
    gc.collect()

    if meta_pairs:
        X_v1 = np.concatenate(v1_x_parts, axis=0)
        S_v1 = np.concatenate(v1_s_parts, axis=0)
        X_v2 = np.concatenate(v2_x_parts, axis=0)
        S_v2 = np.concatenate(v2_s_parts, axis=0)
        del v1_x_parts, v1_s_parts, v2_x_parts, v2_s_parts, r50_2048, rnx_2048
        gc.collect()

        # 5. ResidualCropRefinerUNet 5-fold ensemble
        v1_m_acc = np.zeros((len(meta_pairs), CROP_SZ_V1, CROP_SZ_V1), dtype=np.float32)
        v1_i_acc = np.zeros(len(meta_pairs), dtype=np.float32)
        for fold in range(5):
            sd = load_phase3_state_dict(
                f"phase3/crop_refiner_f{fold}_fp16.pt", weights_dir=w_dir, map_location=device,
                strict_weights_only=strict_weights_only, verify_fp16=verify_fp16, loaded_log=loaded_log,
            )
            m1 = ResidualCropRefinerUNet(in_ch=6, base=32)
            m1.load_state_dict(sd)
            m1.float().to(device).eval()
            pm, pi = predict_4tta_crop_v1(m1, X_v1, S_v1, device)
            v1_m_acc += pm / 5.0
            v1_i_acc += pi / 5.0
            del m1, sd
        del X_v1, S_v1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 6. WideContextRefinerUNetV2 5-fold ensemble
        v2_m_acc = np.zeros((len(meta_pairs), CROP_SZ_WIDE, CROP_SZ_WIDE), dtype=np.float32)
        for fold in range(5):
            sd = load_phase3_state_dict(
                f"phase3/wide_refiner_v2_f{fold}_fp16.pt", weights_dir=w_dir, map_location=device,
                strict_weights_only=strict_weights_only, verify_fp16=verify_fp16, loaded_log=loaded_log,
            )
            m2 = WideContextRefinerUNetV2(in_ch=11, base=40)
            m2.load_state_dict(sd)
            m2.float().to(device).eval()
            pm2, _ = predict_4tta_wide_v2(m2, X_v2, S_v2, device)
            v2_m_acc += pm2 / 5.0
            del m2, sd
        del X_v2, S_v2
        v2_u8 = np.clip(np.round(v2_m_acc * 255.0), 0, 255).astype(np.uint8)
        del v2_m_acc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [CropV1 + WideRefV2] 5-fold inference complete ({time.time() - t0:.1f}s)", flush=True)

        # 7. Attach S27 features & LightGBM scoring
        feat_dicts = []
        for idx, (stem, ci) in enumerate(meta_pairs):
            c = pool_by_stem[stem][ci]
            row = rows_by_stem[stem][ci]
            ub = c["u_box"]
            uw, uh = int(ub[2] - ub[0]), int(ub[3] - ub[1])
            cnn_pm = cv2.resize(v1_m_acc[idx], (uw, uh), interpolation=cv2.INTER_LINEAR)
            c["cnn_u_prob"] = cnn_pm.astype(np.float16)

            wb = c["w_box"]
            ww, wh = int(wb[2] - wb[0]), int(wb[3] - wb[1])
            c["wref_prob"] = cv2.resize(v2_u8[idx].astype(np.float32) / 255.0, (ww, wh), interpolation=cv2.INTER_LINEAR).astype(np.float16)

            u_prob = c["u_prob"].astype(np.float32)
            b55_u = u_prob >= 0.55
            b_cnn = cnn_pm >= 0.50
            u_fg = r50_crops_fg[idx]
            u_cr = r50_crops_cr[idx]
            b_un = u_fg >= 0.45

            inter_c55 = int((b55_u & b_cnn).sum())
            uni_c55 = max(1, int((b55_u | b_cnn).sum()))
            row["cnn_iou_score"] = float(v1_i_acc[idx])
            row["cnn_mask_iou55"] = float(inter_c55) / float(uni_c55)
            row["cnn_area_ratio"] = float(b_cnn.sum()) / float(max(1, int(b55_u.sum())))
            row["unet_mean"] = float(u_fg[b55_u].mean()) if b55_u.any() else 0.0
            row["unet_core"] = float(u_cr[b55_u].mean()) if b55_u.any() else 0.0
            row["unet_iou55"] = float((b55_u & b_un).sum()) / float(max(1, int((b55_u | b_un).sum())))
            row["unet_iou_cnn"] = float((b_cnn & b_un).sum()) / float(max(1, int((b_cnn | b_un).sum())))
            feat_dicts.append(row)

        boosters, feat_names = load_lgb_boosters(
            weights_dir=w_dir, strict_weights_only=strict_weights_only, loaded_log=loaded_log
        )
        X_mat = pd.DataFrame(feat_dicts)[feat_names].values.astype(np.float32)
        scores_acc = np.zeros(len(X_mat), dtype=np.float64)
        for booster in boosters:
            scores_acc += booster.predict(X_mat) / float(len(boosters))
        for idx, (stem, ci) in enumerate(meta_pairs):
            pool_by_stem[stem][ci]["lgb_score"] = float(scores_acc[idx])

    df_sub, stats = materialize_disjoint_rows_phase3b(stems, pool_by_stem, tau=tau)
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return df_sub, stats


def run_phase3b_pipeline(
    test_dir: str,
    weights_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    device_str: str = "auto",
    limit: Optional[int] = None,
    use_cache: bool = False,
    tau: float = 0.255,
    strict_weights_only: bool = False,
    verify_fp16: bool = False,
    loaded_log: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Execute full Phase 3B Local pipeline (Stage 1 -> Semantic UNets -> V1/V2 Refiners -> LGB -> RLE)."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    def _res(p_str: Optional[str], default_rel: str) -> Path:
        raw = Path(p_str) if p_str else (REPO_ROOT / default_rel)
        return raw if (raw.is_absolute() or raw.exists()) else (REPO_ROOT / raw)

    base_cfg_p = _res(config_path, "configs/final_ensemble.json")
    if base_cfg_p.name == "phase3b_refiner.json":
        with open(base_cfg_p, "r") as f:
            p3_cfg = json.load(f)
        tau = float(p3_cfg["ranker"]["tau_score"])
        base_rel = p3_cfg.get("base_ensemble_config", "configs/final_ensemble.json")
        base_cfg_p = REPO_ROOT / base_rel

    with open(base_cfg_p, "r") as f:
        cfg = json.load(f)

    w_dir = _res(weights_dir, "weights")
    t_dir = Path(test_dir) if (Path(test_dir).is_absolute() or Path(test_dir).exists()) else (REPO_ROOT / test_dir)
    image_paths = sorted(
        list(t_dir.glob("*.jpeg")) + list(t_dir.glob("*.jpg")) + list(t_dir.glob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    stems = [p.stem for p in image_paths]
    print(
        f"[Phase 3B Local] Device={device} (strict float32) | Images={len(stems)} | "
        f"use_cache={use_cache} | strict_weights={strict_weights_only}",
        flush=True,
    )
    t0 = time.time()

    img_by_stem: Dict[str, np.ndarray] = {}
    for p in image_paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise IOError(f"Could not read image: {p}")
        img_by_stem[p.stem] = g

    if use_cache:
        from scripts.regenerate_submission import load_test_candidates_for_stem
        if cache_dir is None:
            cand_cache = REPO_ROOT / "work" / "test_cache"
            if not cand_cache.is_dir():
                alt = REPO_ROOT.parent / "solar" / "work" / "test_cache"
                cand_cache = alt if alt.is_dir() else cand_cache
        else:
            cand_cache = Path(cache_dir) if (Path(cache_dir).is_absolute() or Path(cache_dir).exists()) else (REPO_ROOT / cache_dir)
        model_tuples = [(m["id"], m["family"], int(m["img_size"])) for m in cfg["models"]]
        raw_by_stem = {stem: load_test_candidates_for_stem(stem, model_tuples, str(cand_cache)) for stem in stems}
    else:
        raw_by_stem = _collect_stage1_candidates_from_weights(
            stems, img_by_stem, cfg, w_dir, device,
            strict_weights_only=strict_weights_only, verify_fp16=verify_fp16, loaded_log=loaded_log,
        )

    return run_phase3b_from_stage1(
        stems=stems,
        img_by_stem=img_by_stem,
        raw_by_stem=raw_by_stem,
        cfg=cfg,
        w_dir=w_dir,
        device=device,
        tau=tau,
        strict_weights_only=strict_weights_only,
        verify_fp16=verify_fp16,
        loaded_log=loaded_log,
        t_start=t0,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3B Local Refiner + LightGBM Pipeline.")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory of input images (default: test_images)")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Directory with release weights")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory with cached 30-model test detections")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached Mask R-CNN candidates instead of running 30 checkpoints")
    parser.add_argument("--strict-weights", action="store_true", help="Require all checkpoints from --weights-dir (FP16) with zero fallback")
    parser.add_argument("--config", type=str, default="configs/phase3b_refiner.json", help="Phase 3B config path")
    parser.add_argument("--output", type=str, default="work/submission_C_phase3b_local.csv", help="Output submission CSV")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images")
    parser.add_argument("--tau", type=float, default=0.255, help="LightGBM acceptance threshold")
    args = parser.parse_args()

    paths = find_dataset_paths()
    test_dir = args.test_dir or paths.get("test_dir") or str(REPO_ROOT / "sample_images")
    out_path = Path(args.output) if os.path.isabs(args.output) else (REPO_ROOT / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_sub, stats = run_phase3b_pipeline(
        test_dir=test_dir,
        weights_dir=args.weights_dir,
        cache_dir=args.cache_dir,
        config_path=args.config,
        device_str=args.device,
        limit=args.limit,
        use_cache=args.use_cache,
        tau=args.tau,
        strict_weights_only=args.strict_weights,
        verify_fp16=args.strict_weights,
    )
    df_sub.to_csv(out_path, index=False)
    print(f"[Saved] {out_path} ({stats['total_rows']} rows, {stats['total_filaments']} filaments in {stats['elapsed_s']}s)")


if __name__ == "__main__":
    main()
