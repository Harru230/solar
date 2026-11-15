# Third-Party Notices and Licences

This repository is released under the **Apache License 2.0** (`LICENSE`), except for the third-party components, pretrained weights, and dataset references listed below.

No Ultralytics packages or AGPL-licensed code are used anywhere in this repository.

## 1. Deep Learning & Computer Vision Libraries

| Component | Version (Pinned) | Licence | Usage |
| :--- | :--- | :--- | :--- |
| `torch` | `2.14.1+cu126` / `2.14.1+cpu` | BSD-3-Clause | Tensor operations, AMP training, inference |
| `torchvision` | `0.29.1+cu126` / `0.29.1+cpu` | BSD-3-Clause | `maskrcnn_resnet50_fpn_v2` architecture & RoI Align |
| `opencv-python-headless` | `5.0.0.93` | Apache-2.0 | CLAHE preprocessing, image I/O, contours |
| `pycocotools` | `2.0.11` | BSD-2-Clause | COCO RLE mask encoding/decoding (`mask_util`) |
| `pillow` | `12.3.0` | MIT-CMU | Image loading/rendering |
| `numpy` | `2.5.4` | BSD-3-Clause | Array processing, bootstrap resampling |
| `scipy` | `1.18.1` | BSD-3-Clause | Hungarian matching (`linear_sum_assignment`), morphometry |
| `pandas` | `3.0.6` | BSD-3-Clause | CSV submission & fold table handling |
| `matplotlib` | `3.11.2` | PSF / Matplotlib | Diagnostic figures & demo overlays |
| `tqdm` | `4.70.1` | MPL-2.0 / MIT | Progress reporting |
| `pytest` | `9.2.0` | MIT | Automated test suite |

| Component | Version (Pinned) | Licence | Usage |
| :--- | :--- | :--- | :--- |
| `lightgbm` | `4.7.0` | MIT | Phase 3B Stage-2 `S27` instance ranker (`25` boosters, required for Rule C) & adversarial validation |
| `scikit-learn` | `1.9.1` | BSD-3-Clause | ROC-AUC & stratified fold utilities in audit scripts |
| `joblib` | `1.6.0` | BSD-3-Clause | Parallel/serialization utility in experimental scripts (unused in final Rule A/B/C inference) |

## 2. Pretrained Initialization Weights & Training Provenance

1. **Stage-1 Proposal Detectors (30 Mask R-CNN Models)**: Fine-tuned from `torchvision`'s standard **COCO-V1 / ImageNet-1K** pretrained `MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT` (`COCO_V1`) initialization (`torchvision.models`, BSD-3-Clause; COCO annotations under CC BY 4.0).
2. **Phase 3B Full-Disk Semantic U-Nets (10 Models)**:
   - `ResNet50SemanticUNet` (5 folds, `1024x1024`): Backbone initialized from `torchvision.models.ResNet50_Weights.DEFAULT` (ImageNet-1K, BSD-3-Clause).
   - `ResNeXt101SemanticUNet` (5 folds, `1280x1280`): Backbone initialized from `torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT` (ImageNet-1K, BSD-3-Clause).
3. **Phase 3B Crop Boundary Refiners & Ranker**:
   - `ResidualCropRefinerUNet` (5 folds, `160x160`, 6-ch) and `WideContextRefinerUNetV2` (5 folds, `224x224`, 11-ch): Architecture written in-repo (`scripts/refiner_pipeline.py`) and **trained strictly from scratch** on local cross-fitted crops (no external pretrained weights).
   - `LGB_S27_yhyb` (`25` LightGBM `.txt` boosters): Trained from scratch via `lightgbm==4.7.0` (MIT).
- At inference time, this repository constructs all PyTorch models with `pretrained=False` / `weights=None` / `pretrained_backbone=False` and loads **only local in-repo FP16 checkpoint tensors** from `weights/` (upcast to FP32) without network access.

## 3. Dataset

- **MAGFiLO 1.0 (Solar Filament Segmentation Challenge 2026)**: H-alpha solar full-disk imagery and expert filament annotations provided via the Kaggle competition (`MAGFiLO_1.0_Kaggle_2026`). Raw training/test data are not redistributed in this repository (aside from 3 representative images in `sample_images/` for offline demo verification under the challenge terms).
