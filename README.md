# Solar Filament Segmentation Challenge 2026

Semantic instance segmentation of solar H-alpha filaments across full-disk (`2048x2048`) observations for the MAGFiLO 1.0 challenge. Our final solution (**Rule C / Phase 3B Local**, **0.41** Public LB) pairs a six-family Mask R-CNN ResNet50-FPN v2 proposal ensemble (30 models, 4-flip TTA = 120 views, Weighted Mask Fusion) with full-disk `ResNet50` (`1024x1024`) and `ResNeXt101` (`1280x1280`) semantic U-Nets, dual cross-fitted boundary refiners (`ResidualCropRefinerUNet` + `WideContextRefinerUNetV2`), a 25-booster LightGBM S27 instance ranker, and greedy pixel-disjoint canvas painting. It achieves `[nested]` 5-fold OOF PQ of **0.4688** (`0.468754`; `+0.0144` over Stage-1 Rule B **0.4543** / Public LB **0.40**, paired bootstrap $P = 1.0000$) on 707 training observations with **100% self-contained in-repo FP16 checkpoints** (**81 files**, `4.11 GB`, every file `< 100 MB`).

---

## Results Summary

All OOF metrics use the official competition matching engine (`pq_eval.py`: each `(annotator, image)` record scored independently, `IoU > 0.50`). Numbers are explicitly tagged **`[nested]`** (out-of-fold threshold/hyperparameter selection) or **`[in-sample]`** (evaluated at fixed global Rule A / Rule B thresholds).

| Configuration | Scope | OOF PQ (`SQ` / `RQ`) | Public LB / Test (180 imgs) | Source |
| :--- | :--- | :---: | :---: | :--- |
| **Rule A** (30 models, Uniform WMF: $\tau_s=0.85, \tau_m=0.55, \gamma=1.0$) | 707 OOF (`[nested]`) | **0.4508** (0.6761 / 0.6668) | **0.39** (1,165 preds) | [`audit/priority_2_results.json`](audit/priority_2_results.json) |
| **Rule B** (30 models, Weighted WMF: $\tau_s=0.75, \tau_m=0.55, \gamma=1.5$) | 707 OOF (`[nested]`) | **0.4521** (0.6777 / 0.6672) | — | [`audit/priority_2_results.json`](audit/priority_2_results.json) |
| **Rule B** (Fixed global Stage-1 operating point) | 707 OOF (`[in-sample]`) | **0.4543** (0.6773 / 0.6708) | **0.40** (1,155 preds) | [`audit/final_release_analyses.json`](audit/final_release_analyses.json) |
| **Phase 2 Local** (`CropRefinerV1` + `S27` LGB, `med_tau=0.255`) | 707 OOF (`[nested]`) | **0.4676** (— / —) | — | [`REPORT.md`](REPORT.md) |
| **Rule C: Phase 3B Local Winner** (`CropV1 + WideRefV2 + R50/RNX UNet + S27 LGB`, `med_tau=0.255`) | 707 OOF (`[nested]`) | **0.4688** (**0.6857 / 0.6836**) | **0.41** (**1,170 preds**, `submissions/submission_phase3b_local.csv`) | [`configs/phase3b_refiner.json`](configs/phase3b_refiner.json) |
| *Colab A100 Hybrid (`V1` ConvNeXt + `S27_ywhyb`)* | 707 OOF (`[nested]`) | **0.4692** (0.6858 / 0.6841) | **0.40** (1,182 preds, external `.npz` cache only) | [`REPORT.md`](REPORT.md) |
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
- **Public LB test-set noise**: With 180 test images (~1,155–1,170 predicted instances), finite sample size (`std = 0.1494` across OOF images in [`audit/final_release_analyses.json`](audit/final_release_analyses.json), giving $\text{SE} \approx 0.0111\text{ PQ}$ at $N=180$) and single-annotator ground-truth noise explain the ~0.05–0.06 gap between 5-fold OOF (`0.4521`–`0.4688`) and Public LB (`0.39`–`0.41`), as detailed in [`REPORT.md`](REPORT.md).

---

## Weights, Precision & Exact Reproduction

| Weight Set | Location | Size | Purpose & Notes |
| :--- | :--- | :---: | :--- |
| **Full In-Repo FP16 Weights (Self-Contained)** | [`weights/`](weights/) & [`weights/phase3/`](weights/phase3/) (committed on `final`) | **4.11 GB** (`81` files, max `96.06 MB < 100 MB`: 30 Stage-1 FP16 + 5 `ResNet50` UNet + 10 sharded `ResNeXt101` `.part00/.part01` + 5 `CropV1` + 5 `WideRefV2` + 25 LGB + `lgb_metadata.json`) | Self-contained full reproduction of **Rules A, B, and C** (`--strict-weights`) AND `demo.ipynb` (`quick` & `full` modes) with zero external downloads. |
| **External Backup Archives (Optional FP32)** | GitHub Release assets (`6 x ~814 MB` + `1,183 MB` `refiner`) | **6.4 GB** compressed | Optional FP32 backup archives (`scripts/download_weights.sh --all`). |

- **Submitted CSVs (`submissions/`)**: Exact competition files submitted on Kaggle are stored in `submissions/`. **`submission_phase3b_local.csv` (`Rule C`, Public LB `0.41`) is the single Selected final Kaggle submission**; `submission_B_weighted_30models.csv` (`Rule B`, Public LB `0.40`) and `submission_A_uniform_30models.csv` (`Rule A`, Public LB `0.39`) are preserved alongside it strictly for historical ablation reproducibility.
- **Mandatory FP32 compute**: Checkpoints stored in `float16` in `weights/` are upcast to `float32` on load (`scripts/common.py:extract_state_dict()`, `scripts/refiner_pipeline.py:load_phase3_state_dict()`), and inference executes in strict `float32` (`get_autocast_context()` defaults to `nullcontext()`). Using `bfloat16` autocast drops Fold-0 PQ by `-0.003314`, whereas FP16 $\rightarrow$ FP32 upcast inference tracks FP32 within $\Delta PQ = +0.000604$ on Fold-0 (`r50_1536_clahe_f0.pt`, 142 images; [`docs/weights.md`](docs/weights.md)) and achieves mean pixel-union IoU **0.9990** (Rule A) / **0.9993** (Rule B) across all 6 Fold-0 models on `sample_images/` ([`audit/final_release_analyses.json`](audit/final_release_analyses.json)). Across all 180 Kaggle test images regenerated strictly from in-repo `float16` weights in `float32` compute on GPU, predictions closely match the stored Kaggle submissions (mean / median pixel-union IoU vs `submissions/`: **0.9605 / 0.9781** for `Rule C` [min `0.6724`, `1,213 vs 1,170` filaments, identical `3 vs 3` empty images, Hungarian instance IoU mean `0.9696`, cross-run PQ `0.9474`], **0.9431 / 0.9680** for `Rule B` [`1,142 vs 1,155` filaments, instance IoU `0.9566`], and **0.9318 / 0.9620** for `Rule A` [`1,121 vs 1,165` filaments, instance IoU `0.9537`]). The small residual shift comes from `bfloat16` autocast used during original competition Stage-1 test caching, whereas this release enforces strict `float32`; re-running from the original competition Stage-1 cache reproduces `Rule A` and `Rule B` SHA-256 byte-for-byte and yields `1,169 vs 1,170` filaments (`99.91%`) in `Rule C` Stage 2/3.

---

## Quickstart & End-to-End Notebook (`demo.ipynb`)

```bash
# 1. Verify all 81 in-repo weight & booster SHA-256 checksums (all files < 100 MB)
(cd weights && sha256sum -c checksums.sha256)

# 2. Install environment (CUDA 12.6 + Python 3.12 recommended; or requirements-cpu.txt)
pip install -r requirements.txt

# 3. Execute end-to-end demo notebook (MODE="quick" default on 3 sample_images/)
jupyter nbconvert --to notebook --execute --inplace demo.ipynb

# 4. Run full 180-image submission regeneration strictly from in-repo weights/
python scripts/regenerate_submission.py --rule ALL --mode full \
    --strict-weights --weights-dir weights \
    --test-dir <TEST_IMAGES_DIR> --output work/reproduced_csvs --validate
```

## Reproducing Final Submissions A, B & C (`Phase 3B Local`)

All 3 competition submissions (`1,165` filaments for Rule A; `1,155` filaments for Rule B; `1,170` filaments for **Rule C / Phase 3B Local** across 180 test images) can be regenerated jointly in a single GPU pass using strictly in-repo weights (`weights/`, `--strict-weights`):

```bash
python scripts/regenerate_submission.py --rule ALL --mode full \
    --strict-weights --weights-dir weights \
    --test-dir <TEST_IMAGES_DIR> --output work/reproduced_csvs --validate
```

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
├── submissions/                     # Authoritative Kaggle CSVs (Rule A, Rule B, Rule C / phase3b_local)
├── weights/                         # 81 verified files (4.11 GB, all <100 MB): 30 Stage-1 FP16 + 51 phase3/ assets
├── sample_images/                   # 3 representative 2048x2048 H-alpha test/demo observations
├── scripts/                         # Training, inference, Phase 3B refinement, WMF fusion, validation
├── audit/                           # OOF/metric sensitivity, morphology, error & matching JSONs/scripts
├── report/                          # 4-page Overleaf acmart LaTeX source package (main.tex, preamble.tex, main.bib, figures/) + prior main.pdf snapshot
├── figures/                         # Generated analysis plots referenced in REPORT.md
├── tests/                           # 51 unit/integration tests (`pytest -q`)
├── docs/weights.md                  # Full 30-model + Phase 3B manifest & SHA-256 hashes
├── demo.ipynb                       # Self-contained quick (3 img) & full (180 img) pipeline notebook
├── REPORT.md                        # Full technical report & quantitative analysis
├── LICENSE                          # Apache License 2.0
└── THIRD_PARTY.md                   # Third-party library, pretrained weights, and dataset notices
```

---

## Verification

Clean-clone verification covers syntax compilation, the unit/integration test suite (including offline Phase 3B LightGBM + `ResNet50SemanticUNet` + sharded `ResNeXt101SemanticUNet` + `CropRefiner` + `WideRefinerV2` forward checks + disjoint fold leakage tests), SHA-256 verification of all 81 committed weight/ranker files, and execution of `demo.ipynb` (`MODE="quick"` default):

```bash
python -m compileall -q scripts tests audit
pytest -q
(cd weights && sha256sum -c checksums.sha256)
jupyter nbconvert --to notebook --execute --inplace demo.ipynb
```

---

## Limitations

1. **In-Repo FP16 vs Original FP32 Training Weights**: Original training runs used FP32 checkpoints (`~5.26 GB` Stage-1 + full UNets); to meet GitHub's `< 100 MB` per-file repository requirement, `weights/` stores FP16 tensors upcast to FP32 at runtime (and `ResNeXt101` in `.part00/.part01` chunks < `96.1 MB`). See Reproduction Parity for exact measured 180-image IoU vs the Kaggle CSVs.
2. **GPU Runtime**: Full 30-model (`120` views/img) + 10 semantic UNets + ROI refiners across 180 test images requires GPU; `demo.ipynb` provides `MODE="quick"` (3 sample images) by default and `MODE="full"` (180 images).
3. **Training Data**: Raw challenge training images are not redistributed (`sample_images/` provides 3 offline demo images).

## Licence

Released under **Apache-2.0** ([`LICENSE`](LICENSE)). Third-party dependencies, COCO/ImageNet initialization weights, and MAGFiLO 1.0 dataset notices are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md); no Ultralytics or AGPL-licensed code is used.
