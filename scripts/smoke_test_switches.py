#!/usr/bin/env python3
"""
scripts/smoke_test_switches.py - CPU-only smoke test for --anchors5 and --mask_loss bce_dice.

Tests:
1. --anchors5:
   - Replaces anchor generator with 5 aspect ratios ((0.2, 0.5, 1.0, 2.0, 5.0)).
   - Prints shapes of rpn cls_logits and bbox_pred.
   - Verifies that rpn.head.conv and all backbone parameters equal the COCO checkpoint.
   - Forward + backward on CPU with standard dummy input.
   - Forward + backward on CPU with an empty image (no GT boxes).
2. --mask_loss bce_dice:
   - Patches torchvision maskrcnn_loss with 0.5*BCE + 0.5*soft Dice in float32.
   - Forward + backward on CPU with standard dummy input.
   - Forward + backward on CPU with an empty image (no GT boxes).
3. Both switches combined (--anchors5 + --mask_loss bce_dice).
"""
import argparse
import sys
from pathlib import Path

import torch
import torchvision.models.detection.roi_heads as roi_heads
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    apply_anchors5,
    get_maskrcnn_model,
    patch_maskrcnn_loss_bce_dice,
)


def run_smoke_tests():
    print("==================================================================")
    print("   RUNNING CPU SMOKE TEST: --anchors5 AND --mask_loss bce_dice    ")
    print("==================================================================")

    device = torch.device("cpu")

    # Load raw COCO checkpoint for parameter equality verification
    print("\n1. Loading reference COCO Mask R-CNN checkpoint...")
    coco_ref = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    coco_sd = coco_ref.state_dict()

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: --anchors5
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 1: --anchors5 (RPN 5 aspect ratios per level)")
    print("=" * 66)

    model_anchors5 = get_maskrcnn_model(
        pretrained=True,
        min_size=256,
        max_size=256,
        num_classes=2,
        anchors5=True,
    )
    model_anchors5.to(device)

    # 1. Print shapes
    cls_weight = model_anchors5.rpn.head.cls_logits.weight
    cls_bias = model_anchors5.rpn.head.cls_logits.bias
    box_weight = model_anchors5.rpn.head.bbox_pred.weight
    box_bias = model_anchors5.rpn.head.bbox_pred.bias

    print(f"  rpn.head.cls_logits.weight shape: {tuple(cls_weight.shape)} (expected [5, 256, 1, 1])")
    print(f"  rpn.head.cls_logits.bias shape:   {tuple(cls_bias.shape)} (expected [5])")
    print(f"  rpn.head.bbox_pred.weight shape:  {tuple(box_weight.shape)} (expected [20, 256, 1, 1])")
    print(f"  rpn.head.bbox_pred.bias shape:    {tuple(box_bias.shape)} (expected [20])")

    assert cls_weight.shape == torch.Size([5, 256, 1, 1]), f"Unexpected cls_logits shape: {cls_weight.shape}"
    assert cls_bias.shape == torch.Size([5]), f"Unexpected cls_bias shape: {cls_bias.shape}"
    assert box_weight.shape == torch.Size([20, 256, 1, 1]), f"Unexpected bbox_pred shape: {box_weight.shape}"
    assert box_bias.shape == torch.Size([20]), f"Unexpected bbox_bias shape: {box_bias.shape}"
    assert torch.all(cls_bias == 0), "cls_logits bias is not initialized to 0"
    assert torch.all(box_bias == 0), "bbox_pred bias is not initialized to 0"

    # 2. Confirm parameter equality with COCO checkpoint
    model_sd = model_anchors5.state_dict()

    # Parameters that MUST be 100% identical to COCO
    identical_keys = []
    expected_different = {
        "rpn.head.cls_logits.weight",
        "rpn.head.cls_logits.bias",
        "rpn.head.bbox_pred.weight",
        "rpn.head.bbox_pred.bias",
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
        "roi_heads.mask_predictor.conv5_mask.weight",
        "roi_heads.mask_predictor.conv5_mask.bias",
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.bias",
    }

    # Verify rpn.head.conv specifically
    rpn_conv_keys = [k for k in model_sd if k.startswith("rpn.head.conv")]
    print(f"\n  Checking rpn.head.conv parameters ({len(rpn_conv_keys)} tensors):")
    for k in rpn_conv_keys:
        assert k in coco_sd, f"Key {k} missing in COCO checkpoint!"
        is_equal = torch.equal(model_sd[k], coco_sd[k])
        print(f"    {k}: shape {tuple(model_sd[k].shape)} - EQUAL TO COCO: {is_equal}")
        assert is_equal, f"rpn.head.conv parameter {k} does NOT match COCO!"

    # Verify entire backbone
    backbone_keys = [k for k in model_sd if k.startswith("backbone.")]
    for k in backbone_keys:
        assert torch.equal(model_sd[k], coco_sd[k]), f"Backbone parameter {k} does not match COCO!"
    print(f"  All {len(backbone_keys)} backbone parameters confirmed 100% IDENTICAL to COCO.")

    # Check all keys
    for k, v in model_sd.items():
        if k in expected_different:
            continue
        assert k in coco_sd, f"Unexpected parameter {k} not in COCO!"
        assert torch.equal(v, coco_sd[k]), f"Parameter {k} differs from COCO checkpoint!"
        identical_keys.append(k)

    print(f"  TOTAL confirmed identical parameters to COCO: {len(identical_keys)}")

    # 3. Forward + backward with 1 dummy GT box
    model_anchors5.train()
    optimizer = torch.optim.AdamW(model_anchors5.parameters(), lr=1e-4)

    img = torch.rand(3, 256, 256)
    target_with_box = {
        "boxes": torch.tensor([[40.0, 40.0, 120.0, 120.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.ones((1, 256, 256), dtype=torch.uint8),
    }

    optimizer.zero_grad()
    loss_dict = model_anchors5([img], [target_with_box])
    loss = sum(loss_dict.values())
    loss.backward()
    optimizer.step()
    print(f"  Forward+backward (1 GT box): Loss = {loss.item():.4f} - PASS")
    print(f"    Loss components: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")

    # 4. Forward + backward with NO GT boxes
    target_no_box = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "masks": torch.zeros((0, 256, 256), dtype=torch.uint8),
    }

    optimizer.zero_grad()
    loss_dict_empty = model_anchors5([img], [target_no_box])
    loss_empty = sum(loss_dict_empty.values())
    loss_empty.backward()
    optimizer.step()
    print(f"  Forward+backward (0 GT boxes): Loss = {loss_empty.item():.4f} - PASS")
    print(f"    Loss components: { {k: round(v.item(), 4) for k, v in loss_dict_empty.items()} }")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: --mask_loss bce_dice
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 2: --mask_loss bce_dice (0.5*BCE + 0.5*soft Dice)")
    print("=" * 66)

    # Reset maskrcnn_loss patch
    patch_maskrcnn_loss_bce_dice()

    model_bce_dice = get_maskrcnn_model(
        pretrained=True,
        min_size=256,
        max_size=256,
        num_classes=2,
        anchors5=False,  # baseline anchors
    )
    model_bce_dice.to(device)
    model_bce_dice.train()
    optimizer2 = torch.optim.AdamW(model_bce_dice.parameters(), lr=1e-4)

    # 1. Forward + backward with 1 GT box
    optimizer2.zero_grad()
    loss_dict2 = model_bce_dice([img], [target_with_box])
    loss2 = sum(loss_dict2.values())
    loss2.backward()
    optimizer2.step()
    print(f"  Forward+backward (1 GT box): Loss = {loss2.item():.4f} - PASS")
    print(f"    loss_mask = {loss_dict2['loss_mask'].item():.4f} (includes BCE + soft Dice)")
    print(f"    All losses: { {k: round(v.item(), 4) for k, v in loss_dict2.items()} }")

    # 2. Forward + backward with NO GT boxes (tests mask_targets.numel() == 0 guard)
    optimizer2.zero_grad()
    loss_dict2_empty = model_bce_dice([img], [target_no_box])
    loss2_empty = sum(loss_dict2_empty.values())
    loss2_empty.backward()
    optimizer2.step()
    print(f"  Forward+backward (0 GT boxes): Loss = {loss2_empty.item():.4f} - PASS")
    print(f"    loss_mask = {loss_dict2_empty['loss_mask'].item():.4f} (guard correctly returned 0)")
    assert loss_dict2_empty["loss_mask"].item() == 0.0, "loss_mask should be 0.0 when no GT boxes"

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: COMBINED (--anchors5 + --mask_loss bce_dice)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 3: COMBINED (--anchors5 AND --mask_loss bce_dice)")
    print("=" * 66)

    model_both = get_maskrcnn_model(
        pretrained=True,
        min_size=256,
        max_size=256,
        num_classes=2,
        anchors5=True,
    )
    model_both.to(device)
    model_both.train()
    optimizer3 = torch.optim.AdamW(model_both.parameters(), lr=1e-4)

    # 1. Forward + backward with 1 GT box
    optimizer3.zero_grad()
    loss3 = sum(model_both([img], [target_with_box]).values())
    loss3.backward()
    optimizer3.step()
    print(f"  Forward+backward (1 GT box): Loss = {loss3.item():.4f} - PASS")

    # 2. Forward + backward with NO GT boxes
    optimizer3.zero_grad()
    loss3_empty = sum(model_both([img], [target_no_box]).values())
    loss3_empty.backward()
    optimizer3.step()
    print(f"  Forward+backward (0 GT boxes): Loss = {loss3_empty.item():.4f} - PASS")

    print("\n==================================================================")
    print("      ALL CPU SMOKE TESTS PASSED CLEANLY & SUCCESSFULLY!          ")
    print("==================================================================")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Run CPU smoke test for --anchors5 and --mask_loss bce_dice switches."
    ).parse_args()
    run_smoke_tests()
