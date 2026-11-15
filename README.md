# Solar Filament Segmentation Challenge 2026

Semantic instance segmentation of solar H-alpha filaments across full-disk (`2048x2048`) observations for the MAGFiLO 1.0 challenge. Our final solution (**Rule C / Phase 3B Local**) pairs a six-family Mask R-CNN ResNet50-FPN v2 proposal ensemble (30 models, 4-flip TTA = 120 views, Weighted Mask Fusion) with full-disk `ResNet50` (`1024x1024`) and `ResNeXt101` (`1280x1280`) semantic U-Nets, dual cross-fitted boundary refiners (`ResidualCropRefinerUNet` + `WideContextRefinerUNetV2`), a 25-booster LightGBM S27 instance ranker, and greedy pixel-disjoint canvas painting. It achieves `[nested]` 5-fold OOF PQ of **0.4688** (`0.468754`; `+0.0144` over Stage-1 Rule B **0.4543** / Public LB **0.40**, paired bootstrap $P = 1.0000$) on 707 training observations with full local checkpoint reproducibility.

---

## Results Summary

All OOF metrics use the official competition matching engine (`pq_eval.py`: each `(annotator, image)` record scored independently, `IoU > 0.50`). Numbers are explicitly tagged **`[nested]`** (out-of-fold threshold/hyperparameter selection) or **`[in-sample]`** (evaluated at fixed global Rule A / Rule B thresholds).

| Configuration | Scope | OOF PQ (`SQ` / `RQ`) | Public LB / Test (180 imgs) | Source |
| :--- | :--- | :---: | :---: | :--- |
| **Rule A** (30 models, Uniform WMF: $\tau_s=0.85, \tau_m=0.55, \gamma=1.0$) | 707 OOF (`[nested]`) | **0.4508** (0.6761 / 0.6668) | **0.39** (1,165 preds) | [`audit/priority_2_results.json`](audit/priority_2_results.json) |
| **Rule B** (30 models, Weighted WMF: $\tau_s=0.75, \tau_m=0.55, \gamma=1.5$) | 707 OOF (`[nested]`) | **0.4521** (0.6777 / 0.6672) | **0.40** (1,155 preds) | [`audit/priority_2_results.json`](audit/priority_2_results.json) |
| **Rule B** (Fixed global Stage-1 operating point) | 707 OOF (`[in-sample]`) | **0.4543** (0.6773 / 0.6708) | **0.40** (1,155 preds) | [`audit/final_release_analyses.json`](audit/final_release_analyses.json) |
| **Phase 2 Local** (`CropRefinerV1` + `S27` LGB, `med_tau=0.255`) | 707 OOF (`[nested]`) | **0.4676** (0.6857 / 0.6819) | 1,170 preds | [`REPORT.md`](REPORT.md) |
| **Rule C: Phase 3B Local Winner** (`CropV1 + WideRefV2 + R50/RNX UNet + S27 LGB`, `med_tau=0.255`) | 707 OOF (`[nested]`) | **0.4688** (**0.6857 / 0.6836**) | **1,170 preds** (`submission_phase3b_local.csv`) | [`configs/phase3b_refiner.json`](configs/phase3b_refiner.json) |
| *Colab A100 Hybrid (V1 / V1+V2 ConvNeXtAttnRefinerNet)* | 707 OOF (`[nested]`) | 0.4692 / **0.4695** (0.6860 / 0.6844) | 1,182 preds (external `.npz` cache only) | [`REPORT.md`](REPORT.md) |
| *Experimental Scorer v2* (30 models, 6-family, dropped) | 707 OOF (`[nested]`) | 0.4286 (0.6744 / 0.6355) | — | [`audit/priority_2_results.json`](audit/priority_2_results.json) |

- **Phase 3B Local Significance (vs Rule B & Phase 2, `[nested]`, B=10,000)**: $\Delta PQ = +0.014438$ over Rule B (`95% CI [+0.01138, +0.01745]`, $P=1.0000$; `TP=5,344, FP=2,091, FN=2,855`; `k=1` PQ `0.4720`, `multi` PQ `0.4668`), and $+0.001160$ over Phase 2 Local (`P=0.9677`). Per-image Union Dice averages `0.6639` (median `0.7002`), GT split fraction `1.32%`, and prediction merge fraction `1.60%` (`IoU > 0.15`).
- **Disjointness & Multi-Hit Guarantee**: Greedy pixel-disjoint canvas painting enforces $\sum_i \text{IoU}(G, M_i) \le 1$ for any ground-truth mask $G$. On the 707 OOF images (`IoU > 0.50`), **no GT mask was hit more than once (`0 / 8,199`)** under either Rule A or Rule B; **`1` prediction** (out of `7,605` Rule A / `7,497` Rule B) hit two overlapping GT polygons drawn by the same annotator ([`audit/final_release_analyses.json`](audit/final_release_analyses.json), [`REPORT.md`](REPORT.md#L93-L94)).

### Fold-0 Subset Verification (142 Images — Non-Comparable Subset)

> [!NOTE]
> The 6 Compact Fold-0 FP16 checkpoints in `weights/` are evaluated on the **142 Fold-0 OOF images** (`1,902` GT evals), a different subset from the 707-image OOF benchmark above (`0.4520` / `0.4543`). This check verifies weight parity only and does **not** constitute an improvement over the 30-model ensemble.

| Configuration | Scope | Fold-0 PQ (`SQ` / `RQ`) | TP / FP / FN | Source |
| :--- | :--- | :---: | :---: | :--- |
| **Compact 6-Model Fold-0 (`Rule A`)** | 142 Fold-0 (`[in-sample]`) | **0.4554** (0.6739 / 0.6757) | 1,226 / 501 / 676 | [`audit/final_release_analyses.json`](audit/final_release_analyses.json) |
| **Compact 6-Model Fold-0 (`Rule B`)** | 142 Fold-0 (`[in-sample]`) | **0.4586** (0.6756 / 0.6789) | 1,220 / 472 / 682 | [`audit/final_release_analyses.json`](audit/final_release_analyses.json) |

- **Paired bootstrap (Rule B vs Rule A, `[nested]`, B=10,000)**: $\Delta PQ = +0.001334$, 95% CI $[-0.001363, +0.004030]$, $P(\text{Rule B} > \text{Rule A}) = 0.8387$ ([`audit/priority_2_results.json`](audit/priority_2_results.json)).
- **Public LB test-set noise**: With 180 test images (~1,155 instances), single-annotator label noise and finite sample size yield roughly $\pm 0.015\text{ PQ}$ uncertainty; the ~0.05 gap between 5-fold OOF (`~0.45`) and Public LB (`0.39`–`0.40`) matches single-annotator vs multi-annotator reference effects detailed in [`REPORT.md`](REPORT.md).

---

## Weights, Precision & Exact Reproduction

| Weight Set | Location | Size | Purpose & Measured Tolerance |
| :--- | :--- | :---: | :--- |
| **Compact In-Repo Weights** | [`weights/`](weights/) & [`weights/phase3/`](weights/phase3/) (committed in git) | **583 MB** (42 files: 6 Fold-0 `87.8 MB` FP16 + 36 Phase-3 files `< 7 MB`) | Offline CPU demo, Phase-3 refiner/LGB smoke checks & Fold-0 verification ([`configs/compact_6model_fold0.json`](configs/compact_6model_fold0.json)). |
| **Full Release Archives (7 Tarballs)** | GitHub Release assets (`6 x ~814 MB` + `1,183 MB` `refiner`) | **6.4 GB** compressed | Required for **exact** full reproduction of Submissions A, B, and **Rule C** (`scripts/download_weights.sh --all`). |

- **Mandatory FP32 compute**: Checkpoints stored in `float16` in `weights/` are upcast to `float32` on load (`scripts/common.py:extract_state_dict`), and inference executes in strict `float32` (`get_autocast_context()` defaults to `nullcontext()`). Using `bfloat16` autocast drops Fold-0 PQ by `-0.003314`, whereas FP16 $\rightarrow$ FP32 upcast inference tracks FP32 within $\Delta PQ = +0.000604$ on Fold-0 (`r50_1536_clahe_f0.pt`, 142 images; [`docs/weights.md`](docs/weights.md)) and achieves mean pixel-union IoU **0.9990** (Rule A) / **0.9993** (Rule B) across all 6 Fold-0 models on `sample_images/` ([`audit/final_release_analyses.json`](audit/final_release_analyses.json)).

---

## Quickstart & Offline Demo (CPU)

```bash
# 1. Verify in-repo compact weights
(cd weights && sha256sum -c checksums.sha256)

# 2. Install CPU dependencies
pip install -r requirements-cpu.txt

# 3. Execute offline demo notebook on sample_images/ (strict CPU FP32)
CUDA_VISIBLE_DEVICES="" jupyter nbconvert --to notebook --execute --inplace demo.ipynb

# 4. Run single-model or compact 6-model Fold-0 CLI inference on sample_images/
CUDA_VISIBLE_DEVICES="" python scripts/run_pipeline.py \
    --input_dir sample_images --mode single --device cpu --no_tta \
    --checkpoint weights/r50_1536_clahe_f0.pt --output_csv work/demo_single.csv
```

Measured CPU runtime on `sample_images/` (16 CPU cores, strict FP32 compute, no GPU):
- **Single-model single-pass** (`r50_1536_clahe_f0.pt`, `--no_tta`, default uncalibrated `tau_score=0.35`): `~5 s/img` (`5`, `3`, `4` candidate detections across the 3 sample images).
- **Compact 6-model Fold-0 with 4-flip TTA (24 views)** (`configs/compact_6model_fold0.json`, `rule="B"`, calibrated `tau_score=0.75`): **`~131 s/img`** (`131.3 s/img`, `393.97 s` total across all 3 sample images in [`audit/final_release_analyses.json`](audit/final_release_analyses.json), yielding `6` filaments total — identical per-image counts to the 30-model Kaggle Submissions A & B on `sample_images/`). `demo.ipynb` runs single-model inference across all 3 sample images plus full 24-view TTA 6-model inference on `sample_images[0]` (`limit=1`, `~130 s`, `2` consensus filaments).

---

## Reproducing Final Submissions A, B & C (`Phase 3B Local`)

Exact reproduction of the competition submissions (`1,165` filaments for Rule A; `1,155` filaments for Rule B; `1,170` filaments for **Rule C / Phase 3B Local** across 180 test images) requires downloading the 7 release archives from GitHub Releases (`scripts/download_weights.sh`) and providing the competition test images:

```bash
# Download & verify all 7 release archives into work/checkpoints/
bash scripts/download_weights.sh --all --tag v1.0.0 --repo Harru230/solar

# Reproduce Authoritative Winner: Rule C (Phase 3B Local, OOF PQ 0.4688, 1,170 filaments)
python scripts/regenerate_submission.py --rule C --mode full \
    --config configs/phase3b_refiner.json --weights-dir work/checkpoints \
    --test-dir <TEST_IMAGES_DIR> --output work/submission_C_phase3b_local.csv --validate

# Reproduce Stage-1 Submissions A & B (Uniform / Weighted WMF)
python scripts/regenerate_submission.py --rule A --mode full \
    --config configs/final_ensemble.json --weights-dir work/checkpoints \
    --test-dir <TEST_IMAGES_DIR> --output work/submission_A_uniform_30models.csv --validate
python scripts/regenerate_submission.py --rule B --mode full \
    --config configs/final_ensemble.json --weights-dir work/checkpoints \
    --test-dir <TEST_IMAGES_DIR> --output work/submission_B_weighted_30models.csv --validate
```

Fast pre-cached Stage-1 or Phase 3B reproduction (`--mode fast`, reusing `work/test_cache/`) runs with `--rule C --mode fast --test-dir <TEST_IMAGES_DIR>` (or `<15s` on CPU for Rules A/B).

---

## Scripts Overview

| Script | Purpose | Example Command |
| :--- | :--- | :--- |
| [`scripts/refiner_pipeline.py`](scripts/refiner_pipeline.py) | Standalone Phase 3B Local inference (`ResNet50/ResNeXt101` UNets, `CropV1` + `WideRefV2`, `S27` LGB, disjoint RLE) | `python scripts/refiner_pipeline.py --test-dir sample_images --device cpu` |
| [`scripts/run_pipeline.py`](scripts/run_pipeline.py) | End-to-end single-model or ensemble Stage-1 inference + format validation | `python scripts/run_pipeline.py --input_dir sample_images --mode single --device cpu` |
| [`scripts/regenerate_submission.py`](scripts/regenerate_submission.py) | Unified Rule A / Rule B / Rule C submission generator (`fast` cache or `full` weights) | `python scripts/regenerate_submission.py --rule C --mode full --test-dir sample_images --validate` |
| [`scripts/validate_submission.py`](scripts/validate_submission.py) | Validates CSV schema, RLE syntax, test-stem coverage, and pixel disjointness | `python scripts/validate_submission.py --sub work/demo.csv --test_dir sample_images --expected_count 3` |
| [`scripts/download_weights.sh`](scripts/download_weights.sh) | Downloads and SHA-256 checks per-family FP32 Release archives into `work/checkpoints/` | `bash scripts/download_weights.sh --all --tag v1.0.0 --repo <USER>/<REPO>` |
| [`scripts/prune_checkpoints.py`](scripts/prune_checkpoints.py) | Strips optimizer state from raw training checkpoints and exports FP32 or FP16 weights | `python scripts/prune_checkpoints.py --fp16 --output-dir work/fp16` |
| [`scripts/train_maskrcnn.py`](scripts/train_maskrcnn.py) | Multi-GPU PyTorch DDP trainer (`bfloat16` AMP, Multi-CLAHE, optional soft Dice) | `torchrun --nproc_per_node=2 scripts/train_maskrcnn.py --fold 0 --img_size 1536` |
| [`scripts/predict_ensemble.py`](scripts/predict_ensemble.py) | Multi-model / 4-flip TTA OOF and test predictor with WMF fusion | `python scripts/predict_ensemble.py --help` |
| [`scripts/oof_eval.py`](scripts/oof_eval.py) | Evaluates cached OOF proposals across folds and runs nested CV scoring | `python scripts/oof_eval.py --help` |
| [`scripts/tune_threshold.py`](scripts/tune_threshold.py) | Nested-CV grid search over $(\tau_s, \tau_m)$ | `python scripts/tune_threshold.py --help` |
| [`scripts/bootstrap_compare.py`](scripts/bootstrap_compare.py) | Image-level paired bootstrap (`B=10,000`, `seed=2026`) comparing two submission CSVs | `python scripts/bootstrap_compare.py --sub_a a.csv --sub_b b.csv` |
| [`scripts/postprocess.py`](scripts/postprocess.py) | Disjoint canvas painting, small-component pruning, and hole filling utilities | `python scripts/postprocess.py --help` |
| [`scripts/diagnostics.py`](scripts/diagnostics.py) | Per-observation TP/FP/FN, area-binned PQ, and boundary error diagnostics | `python scripts/diagnostics.py --help` |
| [`scripts/check_gpus.py`](scripts/check_gpus.py) | Verifies visible CUDA GPU count before training jobs | `python scripts/check_gpus.py --min-gpus 2` |
| [`pq_eval.py`](pq_eval.py) | Core challenge Panoptic Quality scorer and COCO RLE GT/fold builder | `python pq_eval.py --help` |
| [`audit/run_final_analyses.py`](audit/run_final_analyses.py) | Reproduces 707-image PQ/Dice/IoU distributions, matching counts, Fold-0 parity, and figures | `CUDA_VISIBLE_DEVICES="" nice -n 19 python audit/run_final_analyses.py --help` |
| [`audit/morphology_analysis.py`](audit/morphology_analysis.py) | Computes morphometric descriptors (area, skeleton length, width, solidity) & stratified PQ | `python audit/morphology_analysis.py --help` |

*Experimental scripts (not used in final Rule A/B submissions)*: [`scripts/instance_scorer.py`](scripts/instance_scorer.py), [`scripts/export_scorer_v2.py`](scripts/export_scorer_v2.py), and [`scripts/adversarial_validation.py`](scripts/adversarial_validation.py).

---

## Repository Structure

```text
├── configs/
│   ├── final_ensemble.json          # Authoritative 30-model (6-family x 5-fold) config + Rules A & B
│   ├── phase3b_refiner.json         # Authoritative Phase 3B Local (Rule C) UNets, Refiners & LGB config
│   ├── compact_6model_fold0.json    # Compact 6-model (1 Fold-0 FP16 model per family) config
│   └── fast_single_family.json      # Fast single-model (1536 CLAHE Fold-0) demo config
├── weights/                         # 42 verified files (<100 MB ea): 6 Fold-0 FP16 (527 MB) + phase3/ (56 MB)
├── sample_images/                   # 3 representative 2048x2048 H-alpha test/demo observations
├── scripts/                         # Training, inference, Phase 3B refinement, WMF fusion, validation
├── audit/                           # OOF/metric sensitivity, morphology, error & matching JSONs/scripts
├── report/                          # 4-page Overleaf acmart LaTeX report + compiled main.pdf
├── figures/                         # Generated analysis plots referenced in REPORT.md
├── tests/                           # 50 unit/integration tests (`pytest -q`)
├── docs/weights.md                  # Full 30-model + Phase 3B manifest & SHA-256 hashes
├── demo.ipynb                       # Self-contained CPU inference & validation notebook
├── REPORT.md                        # Full technical report & quantitative analysis
├── LICENSE                          # Apache License 2.0
└── THIRD_PARTY.md                   # Third-party library, pretrained weights, and dataset notices
```

---

## Verification

Offline clean-clone verification covers syntax compilation, the unit/integration test suite (including offline Phase 3B LightGBM + `CropRefiner` + `WideRefinerV2` forward checks), SHA-256 verification of all 42 committed weight/ranker files, and offline CPU execution of `demo.ipynb` (full 180-image A/B/C regeneration requires the external Release archives and competition test images):

```bash
python -m compileall -q scripts tests audit
pytest -q
(cd weights && sha256sum -c checksums.sha256)
CUDA_VISIBLE_DEVICES="" jupyter nbconvert --to notebook --execute --inplace demo.ipynb
```

---

## Limitations

1. **Compact vs Full Ensemble**: In-repo weights cover Fold 0 only (6 of 30 models); full 30-model ensemble reproduction requires the external FP32 release archives ([`docs/weights.md`](docs/weights.md)).
2. **CPU Runtime**: Full 6-model CPU inference takes `131.3 s/img`; full 30-model 180-image inference (`120` views/img) is intended for GPU.
3. **Training Data**: Raw challenge training images are not redistributed (`sample_images/` provides 3 offline demo images).

## Licence

Released under **Apache-2.0** ([`LICENSE`](LICENSE)). Third-party dependencies, COCO/ImageNet initialization weights, and MAGFiLO 1.0 dataset notices are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md); no Ultralytics or AGPL-licensed code is used.
