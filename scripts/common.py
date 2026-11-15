"""
common.py - Shared dataset, model, augmentation, and utility functions
for the Solar Filament Segmentation Challenge 2026.
"""
import glob
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mu
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes

import contextlib

EMPTY_RLE = "PPPP4"  # 2048x2048 empty mask in COCO compressed RLE


def get_autocast_context(
    device: Optional[Union[str, torch.device]] = None,
    use_amp: bool = False,
):
    """Return an autocast context (defaults to FP32 nullcontext for inference).

    Set ``use_amp=True`` explicitly for mixed-precision training; inference defaults
    to strict float32 compute to preserve FP16-upcast parity.
    """
    if not use_amp:
        return contextlib.nullcontext()
    if device is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, torch.device):
        device_type = device.type
    elif isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = "cpu"

    if device_type == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict and cast FP16/BF16 floating weights to float32."""
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    return {
        k: (
            v.float()
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32
            else v
        )
        for k, v in state_dict.items()
    }


def set_seed(seed: int = 2026):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup a logger that prints cleanly to stdout and optionally logs to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def find_dataset_paths(repo_root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Locate train_images, test_images, train JSON, folds.csv, and gt_all.csv."""
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parent.parent)

    env_data_dir = os.environ.get("SOLAR_DATA_DIR")
    search_dirs = [env_data_dir] if env_data_dir else []
    search_dirs.append(repo_root)

    train_dirs: List[str] = []
    test_dirs: List[str] = []
    train_jsons: List[str] = []
    for d in search_dirs:
        if d and os.path.isdir(d):
            train_dirs.extend(glob.glob(os.path.join(d, "**/train_images"), recursive=True))
            test_dirs.extend(glob.glob(os.path.join(d, "**/test_images"), recursive=True))
            train_jsons.extend(glob.glob(os.path.join(d, "**/*train*.json"), recursive=True))
            if not test_dirs:
                jpegs = glob.glob(os.path.join(d, "*.jp*g"))
                if jpegs:
                    test_dirs.append(d)

    sample_dir = os.path.join(repo_root, "sample_images")
    if not test_dirs and os.path.isdir(sample_dir):
        test_dirs = [sample_dir]

    folds_csv = os.path.join(repo_root, "work/folds.csv")
    gt_all_csv = os.path.join(repo_root, "work/gt_all.csv")

    return {
        "train_dir": train_dirs[0] if train_dirs else None,
        "test_dir": test_dirs[0] if test_dirs else None,
        "train_json": train_jsons[0] if train_jsons else None,
        "folds_csv": folds_csv if os.path.exists(folds_csv) else None,
        "gt_all_csv": gt_all_csv if os.path.exists(gt_all_csv) else None,
        "repo_root": repo_root,
    }


def get_run_dir(repo_root: str, run_name: str) -> str:
    """Returns the canonical output directory for a named run: work/runs/<run_name>/."""
    return os.path.join(repo_root, "work", "runs", run_name)


def get_maskrcnn_model(
    pretrained: bool = True,
    min_size: int = 1024,
    max_size: int = 1024,
    num_classes: int = 2,
    anchors5: bool = False,
    model_version: str = "v2",
    logger: Optional[logging.Logger] = None,
) -> torch.nn.Module:
    """
    Instantiates torchvision Mask R-CNN (ResNet50-FPN v1 or v2).
    Uses COCO pretrained weights if available, falls back to ImageNet backbone weights.
    Adjusts internal transform min/max size and replaces box & mask heads for 2 classes.
    """
    ver = str(model_version).lower().strip()
    if ver in ("v1", "1"):
        if pretrained:
            try:
                weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                model = maskrcnn_resnet50_fpn(
                    weights=weights, min_size=min_size, max_size=max_size
                )
                if logger:
                    logger.info("Loaded Mask R-CNN ResNet50-FPN v1 with COCO-pretrained weights.")
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Could not load COCO v1 weights directly ({e}), initializing without weights."
                    )
                model = maskrcnn_resnet50_fpn(weights=None, min_size=min_size, max_size=max_size)
        else:
            model = maskrcnn_resnet50_fpn(weights=None, min_size=min_size, max_size=max_size)
    else:
        # Default v2
        weights = None
        if pretrained:
            try:
                weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = maskrcnn_resnet50_fpn_v2(
                    weights=weights, min_size=min_size, max_size=max_size
                )
                if logger:
                    logger.info("Loaded Mask R-CNN ResNet50-FPN v2 with COCO-pretrained weights.")
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Could not load COCO weights directly ({e}), falling back to ImageNet backbone."
                    )
                try:
                    from torchvision.models import ResNet50_Weights

                    model = maskrcnn_resnet50_fpn_v2(
                        weights_backbone=ResNet50_Weights.DEFAULT,
                        min_size=min_size,
                        max_size=max_size,
                    )
                except Exception as e2:
                    if logger:
                        logger.warning(f"Could not load ImageNet weights ({e2}), initializing randomly.")
                    model = maskrcnn_resnet50_fpn_v2(
                        weights=None, min_size=min_size, max_size=max_size
                    )
        else:
            model = maskrcnn_resnet50_fpn_v2(weights=None, min_size=min_size, max_size=max_size)

    # Replace box predictor (classifier & bbox regressor) for 2 classes (0: bg, 1: filament)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor for 2 classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

    # Upgrade mask head resolution (56x56 masks via 28x28 RoI pool) & detection capacity
    model.roi_heads.mask_roi_pool.output_size = (28, 28)
    model.roi_heads.detections_per_img = 200

    if anchors5:
        apply_anchors5(model, logger=logger)

    return model


def apply_anchors5(model: torch.nn.Module, logger: Optional[logging.Logger] = None) -> torch.nn.Module:
    """
    Reconfigures model.rpn.anchor_generator to use 5 aspect ratios per level:
      sizes = ((32,), (64,), (128,), (256,), (512,))
      aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 5.0),) * 5
    Re-initializes ONLY rpn.head.cls_logits and rpn.head.bbox_pred (N(0, 0.01), bias 0)
    for 5 anchors/location, keeping rpn.head.conv pretrained.
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 5.0),) * len(sizes)
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    in_channels = model.rpn.head.cls_logits.in_channels
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]  # 5

    new_cls_logits = torch.nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
    torch.nn.init.normal_(new_cls_logits.weight, std=0.01)
    torch.nn.init.zeros_(new_cls_logits.bias)

    new_bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
    torch.nn.init.normal_(new_bbox_pred.weight, std=0.01)
    torch.nn.init.zeros_(new_bbox_pred.bias)

    model.rpn.head.cls_logits = new_cls_logits
    model.rpn.head.bbox_pred = new_bbox_pred

    if logger:
        logger.info(
            f"Configured --anchors5: 5 aspect ratios per level (0.2, 0.5, 1.0, 2.0, 5.0). "
            f"cls_logits: {tuple(new_cls_logits.weight.shape)}, bbox_pred: {tuple(new_bbox_pred.weight.shape)}"
        )
    return model


def patch_maskrcnn_loss_bce_dice(logger: Optional[logging.Logger] = None):
    """
    Patches torchvision.models.detection.roi_heads.maskrcnn_loss with:
        0.5 * BCE + 0.5 * soft Dice (per instance, smooth 1.0, computed in float32 under AMP).
    Retains the 'mask_targets.numel() == 0 -> mask_logits.sum() * 0' guard.
    """
    import torchvision.models.detection.roi_heads as roi_heads

    def custom_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        discretization_size = mask_logits.shape[-1]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        # Compute in float32 under AMP for numerical stability
        selected_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels].to(torch.float32)
        targets = mask_targets.to(torch.float32)

        # BCE loss (averaged over all mask elements)
        bce_loss = F.binary_cross_entropy_with_logits(selected_logits, targets)

        # Soft Dice loss per instance (smooth 1.0)
        p = torch.sigmoid(selected_logits)
        p_flat = p.flatten(start_dim=1)
        t_flat = targets.flatten(start_dim=1)
        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice_per_instance = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = (1.0 - dice_per_instance).mean()

        return 0.5 * bce_loss + 0.5 * dice_loss

    roi_heads.maskrcnn_loss = custom_maskrcnn_loss
    if logger:
        logger.info("Patched torchvision maskrcnn_loss with 0.5*BCE + 0.5*soft Dice (float32).")



def preprocess_solar_image(img_gray: np.ndarray, target_size: int) -> np.ndarray:
    """
    Preprocess a grayscale solar H-alpha image into a 3-channel Multi-CLAHE float32 array
    of shape (3, target_size, target_size) with values in range [0.0, 1.0]:
      - Channel 0: img_resized / 255.0
      - Channel 1: CLAHE (clipLimit=2.0, tileGridSize=(8, 8)) / 255.0
      - Channel 2: CLAHE (clipLimit=4.0, tileGridSize=(16, 16)) / 255.0
    """
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    img_resized = cv2.resize(img_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

    ch0 = img_resized.astype(np.float32) / 255.0
    ch1 = clahe1.apply(img_resized).astype(np.float32) / 255.0
    ch2 = clahe2.apply(img_resized).astype(np.float32) / 255.0

    return np.stack([ch0, ch1, ch2], axis=0)


class SolarAugmentation:
    """
    Topology-preserving spatial and photometric augmentations for solar H-alpha images:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90-degree rotations (0, 90, 180, 270)
    - Mild contrast and brightness jitter (preserves filament structure and flux connectivity)
    Strictly NO elastic deformation to preserve physical magnetic flux tube topology.
    """

    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(
        self, img_tensor: torch.Tensor, masks_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # img_tensor: [3, H, W] float in [0, 1]
        # masks_tensor: [N, H, W] uint8 in {0, 1}
        if not self.is_train:
            return img_tensor, masks_tensor

        # Horizontal flip
        if torch.rand(1).item() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            if masks_tensor.numel() > 0:
                masks_tensor = torch.flip(masks_tensor, dims=[2])

        # Vertical flip
        if torch.rand(1).item() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[1])
            if masks_tensor.numel() > 0:
                masks_tensor = torch.flip(masks_tensor, dims=[1])

        # Random 90-deg rotation
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])
            if masks_tensor.numel() > 0:
                masks_tensor = torch.rot90(masks_tensor, k, dims=[1, 2])

        # Mild contrast and brightness jitter
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.25
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.25
        mean = img_tensor.mean(dim=[-2, -1], keepdim=True)
        img_tensor = (img_tensor - mean) * contrast + mean
        img_tensor = img_tensor * brightness
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        return img_tensor, masks_tensor


def rasterize_polygons(
    segmentations: List[List[List[float]]], scale: float = 0.5, size: Tuple[int, int] = (1024, 1024)
) -> List[np.ndarray]:
    """
    Rasters polygon coordinate lists directly at scaled target size (e.g. 1024x1024)
    using cv2.fillPoly. 100x faster than full 2048x2048 rasterization + downsampling.
    """
    h, w = size
    masks = []
    for poly_list in segmentations:
        m = np.zeros((h, w), dtype=np.uint8)
        for poly in poly_list:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = (np.array(poly, dtype=np.float32).reshape(-1, 2) * scale).round().astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
        masks.append(m)
    return masks


def cluster_and_union_masks(
    masks: List[np.ndarray], iou_thresh: float = 0.15
) -> List[np.ndarray]:
    """
    Given masks from multiple annotators on the same physical image,
    clusters overlapping instances (IoU > iou_thresh) and computes their union.
    Unmatched instances are preserved as separate filaments.
    """
    if len(masks) <= 1:
        return masks

    n = len(masks)
    areas = [int(m.sum()) for m in masks]
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        if areas[i] == 0:
            continue
        for j in range(i + 1, n):
            if areas[j] == 0:
                continue
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            if inter > 0:
                union = areas[i] + areas[j] - inter
                if union > 0 and (inter / union) > iou_thresh:
                    adj[i, j] = adj[j, i] = True

    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(adj)
    merged_masks = []
    for c in range(n_comp):
        indices = np.where(labels == c)[0]
        comp_masks = [masks[idx] for idx in indices if areas[idx] > 0]
        if comp_masks:
            merged = np.logical_or.reduce(comp_masks).astype(np.uint8)
            if merged.sum() > 0:
                merged_masks.append(merged)

    return merged_masks


class SolarDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Solar Filament Segmentation.
    Loads 2048x2048 H-alpha images, resizes to target_size (default 1024x1024),
    and generates corresponding instance binary masks and bounding boxes.

    Modes:
      - target='all': every (annotator, observation) pair is an independent sample.
      - target='union': masks across annotators for each observation are merged.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        folds_csv: Optional[str] = None,
        fold: Optional[int] = None,
        is_train: bool = True,
        target_mode: str = "all",  # 'all' or 'union'
        target_size: int = 1024,
        limit: Optional[int] = None,
        seed: int = 2026,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.target_size = target_size
        self.scale = target_size / 2048.0
        self.is_train = is_train
        self.target_mode = target_mode
        self.augment = SolarAugmentation(is_train=is_train)

        with open(json_path) as f:
            coco = json.load(f)

        # Build folds lookup: observation -> fold
        obs_fold = {}
        if folds_csv and os.path.exists(folds_csv):
            f_df = pd.read_csv(folds_csv)
            obs_fold = dict(zip(f_df["observation"], f_df["fold"]))

        # Group annotations by image_id (which is "<annotator>-<observation>")
        anns_by_img_id = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img_id[ann["image_id"]].append(ann)

        # Filter COCO images by fold
        valid_coco_images = []
        for im in coco["images"]:
            obs = im["id"].split("-", 1)[1]
            if fold is not None and obs in obs_fold:
                im_fold = obs_fold[obs]
                if is_train and im_fold == fold:
                    continue  # exclude validation fold during training
                if not is_train and im_fold != fold:
                    continue  # keep only validation fold during validation
            valid_coco_images.append(im)

        if target_mode == "all":
            # Each (annotator, observation) is a distinct sample
            self.samples = []
            for im in valid_coco_images:
                obs = im["id"].split("-", 1)[1]
                img_path = os.path.join(image_dir, f"{obs}.jpeg")
                if not os.path.exists(img_path):
                    # fallback check for jpg
                    img_path = os.path.join(image_dir, f"{obs}.jpg")
                self.samples.append({
                    "sample_id": im["id"],
                    "obs": obs,
                    "img_path": img_path,
                    "annotations": anns_by_img_id.get(im["id"], []),
                })
        elif target_mode in ("union", "consensus"):
            # Group by physical observation
            obs_to_images = defaultdict(list)
            for im in valid_coco_images:
                obs = im["id"].split("-", 1)[1]
                obs_to_images[obs].append(im)

            self.samples = []
            for obs, imgs in obs_to_images.items():
                img_path = os.path.join(image_dir, f"{obs}.jpeg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, f"{obs}.jpg")
                combined_anns = []
                ann_owners = []
                for im in imgs:
                    ann_id = im["id"].split("-", 1)[0]
                    for a in anns_by_img_id.get(im["id"], []):
                        combined_anns.append(a)
                        ann_owners.append(ann_id)
                self.samples.append({
                    "sample_id": obs,
                    "obs": obs,
                    "img_path": img_path,
                    "annotations": combined_anns,
                    "ann_owners": ann_owners,
                    "num_annotators": len(imgs),
                })
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}. Must be 'all', 'union', or 'consensus'.")

        if limit is not None and limit > 0:
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        item = self.samples[idx]

        # Load image (grayscale)
        img_gray = cv2.imread(item["img_path"], cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Image not found at {item['img_path']}")

        # 3-channel Multi-CLAHE preprocessing at target_size
        img_preprocessed = preprocess_solar_image(img_gray, self.target_size)
        img_tensor = torch.from_numpy(img_preprocessed)

        # Rasterize segmentations at target_size
        anns = item["annotations"]
        seg_lists = [a["segmentation"] for a in anns]
        masks = rasterize_polygons(seg_lists, scale=self.scale, size=(self.target_size, self.target_size))

        if self.target_mode == "union" and len(masks) > 1:
            masks = cluster_and_union_masks(masks, iou_thresh=0.15)
        elif self.target_mode == "consensus" and len(masks) > 1:
            from scripts.make_consensus import cluster_and_fuse_observation
            margin_scaled = max(4, int(round(16 * self.scale)))
            masks, _ = cluster_and_fuse_observation(
                masks=masks,
                annotator_ids=item["ann_owners"],
                total_annotators_K=item["num_annotators"],
                iou_thresh=0.25,
                min_support=0.5,
                margin=margin_scaled,
                area_ratio_thresh=0.7,
            )

        # Filter empty masks
        masks = [m for m in masks if m.sum() > 0]

        if len(masks) > 0:
            masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        else:
            masks_tensor = torch.zeros((0, self.target_size, self.target_size), dtype=torch.uint8)

        # Apply augmentation
        img_tensor, masks_tensor = self.augment(img_tensor, masks_tensor)

        # Compute bounding boxes from masks
        if masks_tensor.shape[0] > 0:
            boxes = masks_to_boxes(masks_tensor)
            # Filter degenerate boxes where x2 <= x1 or y2 <= y1
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks_tensor = masks_tensor[valid]
            labels = torch.ones(boxes.shape[0], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes.to(torch.float32),
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target, item["obs"]


def collate_fn(batch):
    """Collate function for Mask R-CNN detection dataloaders."""
    images, targets, obs_ids = zip(*batch)
    return list(images), list(targets), list(obs_ids)


def apply_mask_nms(
    rle_strings: List[str], scores: List[float], iou_thresh: float = 0.5
) -> List[int]:
    """
    Greedy Mask NMS on compressed COCO RLE strings sorted by score descending.
    Suppresses any mask that has IoU > iou_thresh with any higher-scoring kept mask.
    Returns indices in the original list that are kept.
    """
    if len(rle_strings) <= 1:
        return list(range(len(rle_strings)))

    order = np.argsort(scores)[::-1]
    sorted_rles = [rle_strings[i] for i in order]

    rle_objs = [{"size": [2048, 2048], "counts": s.encode("ascii")} for s in sorted_rles]
    # Pairwise IoU matrix (len x len)
    ious = mu.iou(rle_objs, rle_objs, [0] * len(sorted_rles))

    keep = []
    for i in range(len(sorted_rles)):
        overlap = False
        for k in keep:
            if ious[i, k] > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(i)

    return [int(order[i]) for i in keep]


def rle_encode_mask(mask_2048: np.ndarray) -> str:
    """Encodes a binary mask of shape (2048, 2048) into a compressed COCO RLE ascii string."""
    fortran_mask = np.asfortranarray(mask_2048.astype(np.uint8))
    encoded = mu.encode(fortran_mask)
    return encoded["counts"].decode("ascii")


def make_masks_disjoint(
    masks: List[np.ndarray],
    scores: List[float],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Ensures predicted binary masks for an image are strictly pixel-wise disjoint
    at final 2048x2048 resolution before RLE encoding:
    1. Sorts instances by descending score.
    2. Maintains a boolean `occupied` canvas of shape (2048, 2048).
    3. For each binary mask:
       - Computes `clean_mask = mask & (~occupied)`.
       - If clean_mask.sum() == 0 or clean_mask.sum() / mask.sum() < 0.5 or clean_mask.sum() < 10,
         discards the instance.
       - Otherwise, keeps clean_mask and updates `occupied |= clean_mask`.
    4. Asserts that the sum of all kept binary masks for the image has max <= 1 (zero overlapping pixels).
    """
    if not masks:
        return [], []

    order = np.argsort(scores)[::-1]
    sorted_masks = [masks[idx] for idx in order]
    sorted_scores = [scores[idx] for idx in order]

    occupied = np.zeros((2048, 2048), dtype=bool)

    clean_masks: List[np.ndarray] = []
    clean_scores: List[float] = []

    for m, s in zip(sorted_masks, sorted_scores):
        m_bool = m.astype(bool)
        orig_area = int(m_bool.sum())
        if orig_area == 0:
            continue

        clean_mask = m_bool & (~occupied)
        clean_area = int(clean_mask.sum())

        if clean_area == 0 or (clean_area / orig_area) < 0.5 or clean_area < 10:
            continue

        clean_masks.append(clean_mask.astype(np.uint8))
        clean_scores.append(float(s))
        occupied |= clean_mask

    if clean_masks:
        mask_stack = np.stack(clean_masks, axis=0)
        max_overlap = int(mask_stack.sum(axis=0).max())
        assert max_overlap <= 1, f"Assertion failed: overlapping pixels detected (max overlap = {max_overlap})!"

    return clean_masks, clean_scores


# ─────────────────────────────── TTA helpers ─────────────────────────────────

# Flip codes: (flip_ud, flip_lr) → transform name
_FLIP_VARIANTS = [
    (False, True),   # horizontal flip
    (True,  False),  # vertical flip
    (True,  True),   # both flips (180° rotation equivalent)
]


def apply_flip_to_tensor(img: torch.Tensor, flip_ud: bool, flip_lr: bool) -> torch.Tensor:
    """Apply spatial flips to a CHW image tensor."""
    if flip_lr:
        img = torch.flip(img, dims=[2])
    if flip_ud:
        img = torch.flip(img, dims=[1])
    return img


def unflip_mask_tensor(mask: torch.Tensor, flip_ud: bool, flip_lr: bool) -> torch.Tensor:
    """Undo spatial flips on a (K, 1, H, W) or (K, H, W) mask tensor."""
    # Inverse of flip is the same flip
    if flip_lr:
        mask = torch.flip(mask, dims=[-1])
    if flip_ud:
        mask = torch.flip(mask, dims=[-2])
    return mask


def run_tta_inference(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    score_thresh: float,
    mask_thresh: float,
    mask_nms_thresh: float,
    img_size: int,
    tta_mode: str = "flip",
) -> Tuple[List[str], List[float]]:
    """
    Run inference with TTA (flip variants), un-flip masks, and merge via Mask NMS.

    Returns:
        (rle_list, score_list) after merging all TTA variants with Mask NMS.

    The final Mask NMS after merging TTA variants is what keeps pred_multi_hit near 0.
    """
    assert tta_mode == "flip", f"Unsupported TTA mode: {tta_mode}"

    all_masks: List[np.ndarray] = []
    all_rles: List[str] = []
    all_scores: List[float] = []

    # Original pass + 3 flip variants = 4 passes total
    variants = [(False, False)] + list(_FLIP_VARIANTS)

    for flip_ud, flip_lr in variants:
        aug_img = apply_flip_to_tensor(img_tensor.clone(), flip_ud, flip_lr)
        aug_gpu = [aug_img.to(device)]

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(aug_gpu)[0]

        scores = output["scores"].detach().cpu().numpy()
        masks = output["masks"]  # [K, 1, img_size, img_size]

        keep_score = scores >= score_thresh
        if not np.any(keep_score):
            continue

        scores_kept = scores[keep_score]
        masks_kept = masks[keep_score]  # [M, 1, img_size, img_size]

        # Un-flip masks before upsampling
        masks_kept = unflip_mask_tensor(masks_kept, flip_ud, flip_lr)

        # Upsample to 2048x2048 in chunks
        if masks_kept.shape[0] > 0:
            bin_masks_list = []
            chunk_size = 32
            for c_start in range(0, masks_kept.shape[0], chunk_size):
                c_chunk = masks_kept[c_start : c_start + chunk_size]
                c_up = F.interpolate(c_chunk, size=(2048, 2048), mode="bilinear", align_corners=False)
                bin_masks_list.append((c_up.squeeze(1) >= mask_thresh).byte().cpu().numpy())
                del c_up
            bin_masks = np.concatenate(bin_masks_list, axis=0)
        else:
            bin_masks = np.zeros((0, 2048, 2048), dtype=np.uint8)

        for m_idx in range(bin_masks.shape[0]):
            m = bin_masks[m_idx]
            if m.sum() > 0:
                all_masks.append(m)
                all_rles.append(rle_encode_mask(m))
                all_scores.append(float(scores_kept[m_idx]))

    if not all_rles:
        return [], []

    # Merge all TTA variants with greedy Mask NMS to eliminate duplicates
    kept_indices = apply_mask_nms(all_rles, all_scores, iou_thresh=mask_nms_thresh)
    nms_masks = [all_masks[i] for i in kept_indices]
    nms_scores = [all_scores[i] for i in kept_indices]

    clean_masks, clean_scores = make_masks_disjoint(nms_masks, nms_scores)
    if not clean_masks:
        return [], []

    assert np.stack(clean_masks, axis=0).sum(axis=0).max() <= 1, "Assertion failed: overlapping pixels in TTA masks!"

    merged_rles = [rle_encode_mask(m) for m in clean_masks]
    return merged_rles, clean_scores
