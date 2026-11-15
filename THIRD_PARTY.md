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

### Optional / Experimental Dependencies (Not Used in Final Rule A/B Submissions)

| Component | Version (Pinned) | Licence | Usage |
| :--- | :--- | :--- | :--- |
| `lightgbm` | `4.7.0` | MIT | Experimental Scorer v2 & adversarial validation |
| `scikit-learn` | `1.9.1` | BSD-3-Clause | ROC-AUC & stratified fold utilities in audit scripts |
| `joblib` | `1.6.0` | BSD-3-Clause | Serialization of experimental `scorer_v2_models.joblib` |

## 2. Pretrained Initialization Weights

All 30 checkpoints in our ensemble were fine-tuned from `torchvision`'s standard **COCO-V1 / ImageNet-1K** pretrained `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` initialization:

- **ResNet-50 Backbone**: ImageNet-1K pretraining distributed via PyTorch/torchvision (`torchvision.models`, BSD-3-Clause).
- **Mask R-CNN ResNet50-FPN v2 Head**: COCO train2017 pretrained weights distributed via PyTorch/torchvision (`MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT`, BSD-3-Clause; COCO annotations governed by CC BY 4.0).
- At inference time, this repository uses `pretrained=False` and `pretrained_backbone=False` and loads only fine-tuned local checkpoint tensors without network access.

## 3. Dataset

- **MAGFiLO 1.0 (Solar Filament Segmentation Challenge 2026)**: H-alpha solar full-disk imagery and expert filament annotations provided via the Kaggle competition (`MAGFiLO_1.0_Kaggle_2026`). Raw training/test data are not redistributed in this repository (aside from 3 representative images in `sample_images/` for offline demo verification under the challenge terms).
