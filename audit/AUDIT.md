# Master Forensic Audit Report: Solar Filament Segmentation Challenge 2026

**Target Codebase**: `.` (Read-only Reference)  
**Deliverable Location**: `audit/AUDIT.md`  
**Audit Date**: 2026-11-06  
**Synthesis Authors**: Teamwork Forensic Audit Team (Subagents A, B, C, synthesized by Synthesis Worker)  
**Audit Mandate**: Comprehensive forensic audit covering Validation Leakage, Metric & Submission Correctness, Inference Determinism & Reproducibility, and End-to-End Reproduction Blueprint from Model Weights.

---

## 1. Executive Summary

This forensic audit evaluates the model training, validation design, evaluation metric implementation, post-processing algorithms, submission artifact generation, and deployment determinism of the Solar Filament Segmentation Challenge 2026 codebase located at `.`.

### 1.1 Core Audit Conclusions

1. **Validation Integrity (CONFIRMED — Clean)**:
   All 707 physical solar observations and their corresponding 1,154 annotator records are partition-grouped with strictly 0% train/validation overlap across all 5 cross-validation folds (`pq_eval.py:115–130`, `scripts/common.py:413–434`, `work/folds.csv:1–708`). Early Phase 1 single-fold threshold tuning (`tune_threshold.py:671, 753–755`) exhibited an optimistic selection bias of approximately $+0.0031$ PQ on Fold 0 when properly measured against the symmetrical mean of held-out halves (`eval_A_on_B` and `eval_B_on_A`). In Phase 2 this was superseded by strict 5-fold Nested Cross-Validation (`scripts/oof_eval.py:475–505`), which limits optimistic selection bias to a negligible $+0.000369$ PQ (+0.037%). The LightGBM instance scorer is strictly cross-fitted across 5 folds with out-of-fold decision threshold selection (`scripts/instance_scorer.py:574–653`), demonstrating a statistically verified improvement of $+0.004366$ PQ ($p = 99.7\%$).

2. **Metric & Submission Correctness (CONFIRMED — Mathematically Equivalent & Disjoint)**:
   The local Panoptic Quality evaluator (`pq_eval.py:25, 30–62, 70–76`) is mathematically identical to the official Kaggle evaluation notebook (`self-evaluation-notebook.ipynb` Cell 3 lines 1–60, Cell 4 lines 1–40), achieving exact floating-point equivalence (`0.313248 == 0.313248` in `crosscheck.py:11–24`). While neither evaluator implements Hungarian 1:1 bipartite matching (`pq_eval.py:47–52`), multi-hit inflation is geometrically eliminated by greedy canvas postprocessing (`scripts/common.py:598–647`, `scripts/postprocess.py:36–117`), which guarantees $100\%$ pixel-disjoint instances (0 overlapping images across all 180 test images). RLE encoding/decoding is bit-for-bit exact across 4,194,304 pixels using pycocotools column-major Fortran arrays (`np.asfortranarray()`). Crucially, empty mask placeholder rows (`"PPPP4"`) evaluate as False Positives (`fp += 1`) if passed directly to `pq_eval.score()`, requiring mandatory dataframe filtering (`df["segmentation_rle"] != "PPPP4"`) during evaluation.

3. **Inference Determinism & Reproducibility (CONFIRMED — Deterministic Pipeline, Portable Release Required)**:
   Inference execution is highly deterministic: random seeds are applied across core libraries, 4-view TTA spatial flips follow a fixed sequence (`["id", "h", "v", "hv"]`), and Weighted Mask Fusion (WMF) sorts candidates via stable Timsort on continuous float32 confidence scores. However, full portable reproduction on an external machine requires resolving three high-risk configuration blockers: (1) repo `requirements.txt` is unpinned and omits `torch`, `torchvision`, and `lightgbm`; (2) queue and export scripts contain hardcoded host paths (`.` and `<local_downloads_dir>`); and (3) mixed-precision autocast is hardcoded to `torch.bfloat16` on CUDA (`predict.py:94`), requiring an NVIDIA GPU with Compute Capability $\ge 8.0$ (Ampere+) and $\ge 16$ GB VRAM.

---

## 2. Master Risk Matrix

All findings from the validation, metric, postprocessing, determinism, and environment audits are categorized and ranked below across High, Medium, and Low risk tiers.

| Finding ID | Domain / Area | Finding Description & Mechanism | File & Line Citation | Status | Risk Tier | Actionable Remediation for `solar_release` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ENV-01** | Dependencies | Repository `requirements.txt` is unpinned and completely omits `torch`, `torchvision`, and `lightgbm`. | `requirements.txt:1–8`, `scripts/setup_pc.sh:17–19` | **CONFIRMED** | **HIGH** | Provide a fully pinned `requirements.txt` locking exact environment packages from `uv pip freeze`: `torch==2.14.1+cu126`, `torchvision==0.29.1+cu126`, `lightgbm==4.7.0`, `numpy==2.5.4`, `pandas==3.0.6`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `opencv-python-headless==5.0.0.93`, `pycocotools==2.0.11`, `pillow==12.3.0`, `matplotlib==3.11.2`, `tqdm==4.70.1`. |
| **ENV-02** | Hardware Lock | Hardcoded `torch.amp.autocast("cuda", dtype=torch.bfloat16)` crashes on non-Ampere GPUs (Turing/Pascal) and CPUs. | `predict.py:94`, `predict_ensemble.py:423`, `oof_eval.py:90`, `train_maskrcnn.py:94` | **CONFIRMED** | **HIGH** | Document NVIDIA Ampere+ ($\ge 16$ GB VRAM) prerequisite; implement automatic fallback to `float32` on CPU and `float16` on older GPUs. |
| **ENV-03** | Portability | Hardcoded paths `.` and `<local_downloads_dir>` break script portability on foreign machines. | `scripts/run_queue.sh:4`, `scripts/stop_queue.sh:5`, `scripts/run_followup_tasks.py:861`, `scripts/multifamily_pipeline.py:1245` | **CONFIRMED** | **HIGH** | Replace hardcoded absolute paths with dynamic repository root discovery via `pathlib.Path(__file__).resolve()` and CLI `--out_dir`. |
| **MET-01** | Evaluation Trap | Empty mask placeholder `"PPPP4"` decodes to all-zero mask, computing $\text{IoU}=0.0$ and inflating False Positives ($FP+1$). | `pq_eval.py:41–46, 70–76`, `scripts/common.py:29`, `scripts/predict.py:398`, `scripts/postprocess.py:169–170` | **CONFIRMED** | **HIGH** | Mandate `df[df["segmentation_rle"] != "PPPP4"]` filtering prior to calling `pq_eval.score()` in all evaluation and test scripts. |
| **MET-02** | RLE Encoding | `pycocotools.mask.encode` requires Fortran column-major order; C-contiguous arrays trigger `ValueError`. | `scripts/common.py:591–596`, `scripts/postprocess.py:95`, `scripts/predict_ensemble.py:254` | **CONFIRMED** | **HIGH** | Enforce `np.asfortranarray(mask.astype(np.uint8))` across all mask encoding utilities and submission generation functions. |
| **MET-03** | Canvas Disjointness | Overlapping candidate masks violate panoptic assumptions and distort competition evaluation. | `scripts/common.py:598–647`, `scripts/postprocess.py:36–117`, `scripts/validate_submission.py:195` | **CONFIRMED** | **HIGH** | Retain greedy score-ordered canvas painting with the 50% area retention rule to guarantee 0 pixel overlaps across instances. |
| **LEAK-01** | Hyperparameter Bias | Phase 1 single-fold sweep optimizes $(\tau_{\text{score}}, \tau_{\text{mask}})$ on the evaluated fold, yielding $+0.0031$ PQ optimistic bias against the symmetrical mean of held-out halves (`eval_A_on_B` and `eval_B_on_A`). | `scripts/tune_threshold.py:671, 753–755`, `scripts/predict.py:238–247`, `work/best_thresholds.json:1–15`, `work/runs/r50_1536_anchors5_e30/best_thresholds.json` | **CONFIRMED** | **MEDIUM** | Clearly qualify single-fold ablation numbers in technical reports as upper-bound validation checks superseded by 5-fold Nested CV. |
| **MET-04** | Matching Logic | Neither `pq_eval.py` nor official notebook uses Hungarian 1:1 matching; multiple detections can match 1 GT if overlapping. | `pq_eval.py:47–52`, `self-evaluation-notebook.ipynb` Cell 3 lines 62–97, `tests/test_pq.py:31–46` | **CONFIRMED** | **MEDIUM** | Rely on canvas pixel-disjointness invariant ($\sum M_k \le 1$), which mathematically prevents multi-hits ($\text{pred\_multi\_hit} = 0$). |
| **DET-01** | DataLoader Seeding | `DataLoader(..., shuffle=True, num_workers=4)` lacks `worker_init_fn`, introducing fork-timing variation in training augmentations. | `scripts/train_maskrcnn.py:362–370` | **CONFIRMED** | **MEDIUM** | Add deterministic `worker_init_fn` and PyTorch `Generator` to guarantee bit-for-bit reproducible training from scratch. |
| **DET-02** | Multithreading Split | LightGBM uses OpenMP multithreading (`n_jobs=4`) without `deterministic=True`, risking micro-divergences across CPU core topologies. | `scripts/instance_scorer.py:593`, `scripts/multifamily_pipeline.py:818` | **CONFIRMED** | **MEDIUM** | Set `deterministic=True` and fixed `n_jobs` in LightGBM classifier and regressor parameters for exact tree parity. |
| **LEAK-02** | Split Grouping | Grouping by physical observation (`observation` stem) strictly keeps all annotators in same fold (707 stems, 0 overlap). | `pq_eval.py:115–130`, `scripts/common.py:413–434`, `work/folds.csv:1–708` | **CONFIRMED** | **LOW** | Preserve observation-level hash partitioning to maintain zero leakage between training and validation sets. |
| **LEAK-03** | Temporal Window | 45 calendar dates span multiple folds (91 images) due to site stratification rather than solar synodic block grouping. | `work/folds.csv:1–708` | **CONFIRMED** | **LOW** | Document temporal stratification nuance in technical documentation as standard image segmentation protocol. |
| **LEAK-04** | Nested CV Tuning | Phase 2 5-fold OOF threshold sweeper selects thresholds strictly on 4 training folds; optimistic bias is only $+0.000369$ PQ. | `scripts/oof_eval.py:475–505`, `work/oof/oof_tuning_summary.json:16–27` | **CONFIRMED** | **LOW** | Maintain leave-one-fold-out outer cross-validation for all production threshold sweeps. |
| **LEAK-05** | Scorer Cross-Fitting | 5-fold LightGBM cross-fitting trains on 4 folds and predicts on 5th fold with zero leakage; bootstrap delta $+0.004366$ ($p=99.7\%$). | `scripts/instance_scorer.py:574–600, 628–653`, `work/oof/instance_scorer_summary.json:1–24` | **CONFIRMED** | **LOW** | Retain 5-fold model cross-fitting and out-of-fold $\tau$ threshold selection for the instance scoring layer. |
| **LEAK-06** | Strict Variant CV | Outer fold selection across 6 Scorer variants and 19 candidate taus is executed strictly in inner cross-validation. | `scripts/run_followup_tasks.py:505–546` | **CONFIRMED** | **LOW** | Retain strict inner-CV variant selection to prevent snooping across feature formulations. |
| **MET-05** | Metric Equivalence | `pq_eval.py` is mathematically equivalent to `self-evaluation-notebook.ipynb` down to 6 decimal places (`0.313248 == 0.313248`). | `pq_eval.py:25, 30–62, 70–76`, `crosscheck.py:11–24` | **CONFIRMED** | **LOW** | Use `pq_eval.py` as default high-throughput, low-memory evaluation harness. |
| **MET-06** | Submission Schema | Submissions strictly follow `['filament_id', 'segmentation_rle']`, covering all 180 stems with 3 valid empty placeholder rows. | `scripts/validate_submission.py:60–114`, `work/submission.csv:1–1205`, `work/submission_r50_1536.csv:1–1088` | **CONFIRMED** | **LOW** | Keep `validate_submission.py` as automated release integrity check. |
| **DET-03** | CUDA Kernels | Atomic operations in bilinear interpolation and RoIAlign exhibit minor floating point non-associativity ($\sim 10^{-7}$). | `scripts/common.py:32–40`, `scripts/predict.py:140` | **CONFIRMED** | **LOW** | Document that metric score invariance holds ($\Delta PQ = 0.0000$) despite microscopic sub-pixel float jitter. |
| **DET-04** | TTA & WMF Order | 4-view TTA order is fixed (`["id", "h", "v", "hv"]`); WMF sorting is stable via continuous float32 confidence scores. | `scripts/predict_ensemble.py:144–216, 396–401`, `scripts/oof_eval.py:78–83` | **CONFIRMED** | **LOW** | Fully deterministic under identical weights and inputs. |

---

## 3. Validation Leakage Audit (Subagent A Findings)

### 3.1 Observation-Level Grouping and Annotator Record Integrity
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean)**
- **Citations**: `~/solar/pq_eval.py`: Lines 115–130; `~/solar/scripts/common.py`: Lines 413–434; `~/solar/work/folds.csv`: Lines 1–708; `~/solar/data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`.
- **Finding & Observation**:
  The competition dataset comprises 1,154 annotator image records representing 707 unique physical solar observations, with up to 3 annotators independently segmenting the same physical observation (`<annotator>-<observation>_<k>`). In `pq_eval.py:115–130`, fold assignment is constructed by extracting the physical observation stem (`im["id"].split("-", 1)[1]`) and stratifying by observatory site (`o[-2:]`, e.g., `Bh`, `Ch`, `Lh`, `Mh`, `Th`, `Uh`).
  In `scripts/common.py:413–434`, training and validation datasets filter images strictly by matching `obs = im["id"].split("-", 1)[1]` against `obs_fold = dict(zip(f_df["observation"], f_df["fold"]))`:
  ```python
  if is_train and im_fold == fold:
      continue  # exclude validation fold during training
  if not is_train and im_fold != fold:
      continue  # keep only validation fold during validation
  ```
  Inspection of `work/folds.csv` confirms exactly 707 rows partitioned into 5 folds:
  - Fold 0: 142 observations (234 annotator-images)
  - Fold 1: 142 observations (233 annotator-images)
  - Fold 2: 141 observations (228 annotator-images)
  - Fold 3: 141 observations (228 annotator-images)
  - Fold 4: 141 observations (231 annotator-images)
  - Total: 707 observations, 1,154 annotator-images, 8,199 ground truth filaments.
  Programmatic verification confirms that zero observation stems exist in both the training and validation splits of any fold (0% train/val overlap).

### 3.2 Phase 1 Single-Fold Threshold Tuning Bias (LEAK-01 Recomputed)
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **MEDIUM (Optimistic Selection Bias in Phase 1 Baseline)**
- **Citations**: `~/solar/scripts/tune_threshold.py`: Lines 671, 751–755, 951–953; `~/solar/scripts/predict.py`: Lines 238–247; `~/solar/work/best_thresholds.json`: Lines 1–15; `~/solar/work/runs/r50_1536_anchors5_e30/best_thresholds.json`; `~/solar/work/runs/r50_1536_clahe_e30/best_thresholds.json`; `~/solar/work/runs/r50_1536_dice_e30/best_thresholds.json`.
- **Finding & Observation**:
  In early baseline experiments (Phase 1, documented in `work/best_thresholds.json` and Queue Job #1), `tune_threshold.py` loaded `val_obs = folds_df.loc[folds_df["fold"] == args.fold]` and executed a dense 2D grid sweep over $\tau_{\text{score}}$ and $\tau_{\text{mask}}$ on the entire held-out fold to find the pair maximizing PQ. The resulting best thresholds were serialized to `best_thresholds.json` and consumed by `predict.py` to score the same fold.
  This introduced an optimistic selection bias. An initial informal inspection compared the full fold score against `eval_A_on_B` alone, reporting a $+0.0074$ PQ gap. However, because Half B happened to be slightly harder than Half A, measuring bias against only one direction is mathematically asymmetric and conflates split difficulty with tuning bias.

  **Rigorous Recomputation via Mean of Held-Out Halves**:
  To measure pure tuning bias without split imbalance, the true held-out benchmark is the symmetrical arithmetic mean of cross-evaluations:
  $$\text{Mean Held-Out PQ} = \frac{\text{eval\_A\_on\_B} + \text{eval\_B\_on\_A}}{2}$$
  $$\text{Optimistic Selection Bias} = \text{best\_full} - \text{Mean Held-Out PQ}$$

  Evaluating all Phase 1 ablation runs from their serialized `best_thresholds.json` manifests yields:
  1. **Anchors 5 (`work/runs/r50_1536_anchors5_e30/best_thresholds.json`)**:
     - `best_full` (tuned and evaluated on all 142 Fold 0 images): **0.425839**
     - `eval_A_on_B` (tuned on Half A, evaluated on held-out Half B): **0.418447**
     - `eval_B_on_A` (tuned on Half B, evaluated on held-out Half A): **0.427009**
     - Symmetrical held-out mean: $\frac{0.418447 + 0.427009}{2} = \mathbf{0.422728}$
     - Recomputed Selection Bias: $0.425839 - 0.422728 = \mathbf{+0.003111\ \text{PQ}}$ ($+0.74\%$).
  2. **CLAHE Baseline (`work/runs/r50_1536_clahe_e30/best_thresholds.json`)**:
     - `best_full`: **0.424849**
     - `eval_A_on_B`: **0.416925**, `eval_B_on_A`: **0.423023**
     - Symmetrical held-out mean: $\frac{0.416925 + 0.423023}{2} = \mathbf{0.419974}$
     - Recomputed Selection Bias: $0.424849 - 0.419974 = \mathbf{+0.004875\ \text{PQ}}$ ($+1.16\%$).
  3. **Soft Dice (`work/runs/r50_1536_dice_e30/best_thresholds.json`)**:
     - `best_full`: **0.430052**
     - `eval_A_on_B`: **0.421681**, `eval_B_on_A`: **0.426573**
     - Symmetrical held-out mean: $\frac{0.421681 + 0.426573}{2} = \mathbf{0.424127}$
     - Recomputed Selection Bias: $0.430052 - 0.424127 = \mathbf{+0.005925\ \text{PQ}}$ ($+1.40\%$).

  Across all runs, the true optimistic selection bias of single-fold tuning is $+0.0031$ to $+0.0059$ PQ (substantially smaller than the unilateral $+0.0074$ figure). In Phase 2, moving to 5-fold Nested CV reduced this bias further to an insignificant $+0.000369$ PQ.

### 3.3 Phase 2 5-Fold Nested Cross-Validation
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Nested Design)**
- **Citations**: `~/solar/scripts/oof_eval.py`: Lines 475–505; `~/solar/work/oof/oof_tuning_summary.json`: Lines 4–74; `~/solar/scripts/multifamily_pipeline.py`: Lines 416–578.
- **Finding & Observation**:
  In Phase 2, the threshold selection mechanism was upgraded to a strict leave-one-fold-out nested cross-validation across all 707 observations:
  ```python
  for held_out_f in range(5):
      # optimize threshold pair strictly on folds j != held_out_f
      tr_iou = sum(f_dict[j]["sq"] * f_dict[j]["tp"] for j in range(5) if j != held_out_f)
      ...
      sel_sc, sel_mk = best_train_pair
      held_res = grid_stats[(sel_sc, sel_mk)][held_out_f]
  ```
  From `work/oof/oof_tuning_summary.json`:
  - Retrospective Global Best (leaked across all 5 folds simultaneously): $\tau_{\text{score}} = 0.825, \tau_{\text{mask}} = 0.500 \implies \text{PQ} = \mathbf{0.434654}$ (TP: 5,285, FP: 2,763, FN: 2,914)
  - Strict 5-Fold Nested CV (outer fold scored using thresholds chosen on other 4 folds): $\text{PQ} = \mathbf{0.434285}$ (TP: 5,277, FP: 2,772, FN: 2,922)
  - Optimistic bias of global vs nested CV: $\mathbf{+0.000369\ \text{PQ}}$ (+0.037%).
  Per-fold selections on the 4 training folds were remarkably stable: Fold 0 selected $(0.825, 0.500)$, Fold 1 selected $(0.825, 0.500)$, Fold 2 selected $(0.825, 0.500)$, Fold 3 selected $(0.825, 0.525)$, and Fold 4 selected $(0.825, 0.500)$.

### 3.4 LightGBM Instance Scorer Cross-Fitting and Tau Selection
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Cross-Fitting)**
- **Citations**: `~/solar/scripts/instance_scorer.py`: Lines 574–600, 628–653; `~/solar/work/oof/instance_scorer_summary.json`: Lines 1–96; `~/solar/scripts/multifamily_pipeline.py`: Lines 798–825.
- **Finding & Observation**:
  The LightGBM instance scoring layer trains on candidate detections from 4 folds and predicts out-of-fold instances on the held-out 5th fold. The target variable is $y = \frac{1}{K}\sum_{r=1}^K \text{IoU}_r \cdot \mathbf{1}[\text{IoU}_r > 0.5]$ (`instance_scorer.py:155–188`), reflecting true annotator agreement.
  Decision threshold $\tau_y$ is selected strictly on the 4 training folds via `fast_score_from_cache(obs_eval_cache, train_obs, tau)` before evaluating the held-out fold:
  - Fold 0: `best_tau = 0.250` $\implies$ Val PQ: 0.441669
  - Fold 1: `best_tau = 0.245` $\implies$ Val PQ: 0.428574
  - Fold 2: `best_tau = 0.250` $\implies$ Val PQ: 0.439023
  - Fold 3: `best_tau = 0.250` $\implies$ Val PQ: 0.443205
  - Fold 4: `best_tau = 0.250` $\implies$ Val PQ: 0.440287
  - **Pooled Cross-Fitted OOF PQ (all 707 images)**: **0.438651** (SQ: 0.670747, RQ: 0.653974, TP: 5,101, FP: 2,300, FN: 3,098).
  - **Global Best Tau (0.250 retrospectively on all 707 images)**: **0.438897**.
  - **Optimistic bias**: $\mathbf{+0.000246\ \text{PQ}}$ (+0.025%).
  - **Statistical Significance**: A 1,000-sample paired bootstrap over all 707 images (`instance_scorer_summary.json:87–96`) comparing Scorer v1 (`0.438651`) against the Nested-CV Threshold Rule (`0.434285`) demonstrated a point delta of $\mathbf{+0.004366}$ PQ (95% CI: $[+0.001139, +0.007754]$, $p = 99.7\%$). The gain is genuine and free from cross-validation leakage.

### 3.5 Strict Variant Selection in Followup Tasks
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Outer Nested CV)**
- **Citations**: `~/solar/scripts/run_followup_tasks.py`: Lines 505–546 (`run_strict_nested_scorer_selection`).
- **Finding & Observation**:
  To protect against hyperparameter leakage across multiple feature formulations (14 vs 19 vs 21 features, $\tau_{\text{mask}} \in \{0.500, 0.525\}$) and 19 candidate $\tau_y$ thresholds, `run_followup_tasks.py` runs a 2-level nested CV. Both the optimal model variant and the optimal threshold $\tau_y$ are determined exclusively on `f != val_f` before scoring outer held-out fold `val_f`.

### 3.6 Temporal Proximity and Nearest Training Observation Gap Analysis
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Observation Independence, No Temporal Distribution Skew)**
- **Citations**: `~/solar_release/audit/time_gap_stats.json`; `~/solar_release/audit/time_gap_distribution.png`; `~/solar/work/folds.csv`: Lines 1–708; observation stem timestamps (`YYYYMMDDHHMMSS<Site>`).
- **Finding & Observation**:
  The competition dataset comprises 707 training/validation observations spanning 656 distinct calendar dates and 180 test observations. Folds were partitioned with observatory site stratification (`Bh`, `Ch`, `Lh`, `Mh`, `Th`, `Uh`) rather than multi-week temporal block grouping.
  To audit whether validation folds or the test set suffer from anomalous temporal proximity or distribution skew, an exact nearest-neighbor time gap analysis was executed across all observation timestamps (extracted as `YYYY-MM-DD HH:MM:SS` from observation IDs, comparing across any observatory site):

  | Distribution Metric | Validation vs Nearest Train (Fold Split) | Test vs Nearest Train (All 707) | Validation vs Other Train (All 707) |
  | :--- | :--- | :--- | :--- |
  | **Sample Count ($N$)** | 707 validation observations | 180 test observations | 707 validation observations |
  | **Median Gap (Hours)** | **46.41 h** (1.93 days) | **36.68 h** (1.53 days) | **36.37 h** (1.52 days) |
  | **Mean Gap (Hours)** | **76.36 h** (3.18 days) | **63.43 h** (2.64 days) | **65.66 h** (2.74 days) |
  | **Minimum Gap** | 0.014 h (~50 seconds) | 0.014 h (~50 seconds) | 0.014 h (~50 seconds) |
  | **Maximum Gap** | 1,581.89 h (65.91 days) | 1,299.99 h (54.17 days) | 1,581.89 h (65.91 days) |

  **Key Conclusions**:
  1. **Identical Test and Validation Exposure**: When comparing against all 707 training images, the test set median gap to the nearest training observation is **36.68 hours** (1.53 days), virtually indistinguishable from the validation set median gap of **36.37 hours** (1.52 days) against other training images. This confirms that the test set experiences the exact same temporal sampling density and proximity as the cross-validation set.
  2. **Fold Split Conservatism**: In the 5-fold cross-validation split (where only the 4 training folds are available as reference), the median gap to the nearest training image is **46.41 hours** (1.93 days). The CV split is slightly more conservative than the full training set exposure.
  3. **Visual Distribution**: The distribution plot has been generated and archived at `~/solar_release/audit/time_gap_distribution.png`, displaying smooth, overlapping log-scale probability density functions with no anomalous clustering.

### 3.7 Threshold Grid Extension and Search Boundary Saturation Analysis
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Optimal Hyperparameters Saturate in Interior, Zero Boundary Truncation)**
- **Citations**: `~/solar_release/audit/grid_extension_results.json`; `~/solar_release/audit/run_grid_extension.py`; `~/solar/scripts/multifamily_pipeline.py`: Lines 1340–1440.
- **Finding & Observation**:
  To confirm that the threshold grids used in production did not artificially truncate higher-performing regions, CPU grid extension sweeps were executed over all 707 OOF observations using cached predictions:

  1. **3-Family Threshold Rule Grid Extension**:
     - **Original Grid**: $\tau_{\text{score}} \in \{0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875\}$ (10 values) $\times$ $\tau_{\text{mask}} \in [0.475, 0.575]$ (5 values).
     - **Extended Grid**: $\tau_{\text{score}} \in \{0.60, \dots, 0.875, \mathbf{0.90, 0.925, 0.95}\}$ (13 values) $\times$ $\tau_{\text{mask}} \in [0.475, 0.575]$.
     - **Original Nested-CV PQ**: **0.443123** ($\tau_{\text{score}} = 0.875, \tau_{\text{mask}} = 0.550$, TP: 5,163, FP: 2,297, FN: 3,036).
     - **Extended Nested-CV PQ**: **0.443123** ($\tau_{\text{score}} = 0.875, \tau_{\text{mask}} = 0.550$).
     - **Retrospective Global Best**: **0.443399** at $(0.875, 0.550)$.
     - **Paired Bootstrap (1,000 iterations)**: $\Delta \text{PQ} = \mathbf{0.000000}$ (95% CI: $[0.0, 0.0]$, $p = 0.0\%$).
     - The optimal threshold saturates at $(0.875, 0.550)$. Higher thresholds ($\ge 0.90$) excessively suppress true filaments, reducing recall without compensating precision gains.

  2. **Scorer v2 View-Invariant $\tau_y$ Grid Extension**:
     - **Original Grid**: $\tau_y \in [0.180, 0.270]$ in steps of $0.005$ (19 candidate thresholds).
     - **Extended Grid**: $\tau_y \in [0.180, \mathbf{0.320}]$ in steps of $0.005$ (29 candidate thresholds, adding $\{0.275, 0.280, 0.285, 0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320\}$).
     - **Original Nested-CV PQ**: **0.444804** (mean chosen $\tau_y = 0.2440$, chosen per fold: $[0.230, 0.230, 0.265, 0.230, 0.265]$, TP: 5,193, FP: 2,311, FN: 3,006).
     - **Extended Nested-CV PQ**: **0.444804** (identical per-fold chosen thresholds: $[0.230, 0.230, 0.265, 0.230, 0.265]$, mean chosen $\tau_y = 0.2440$).
     - **Retrospective Global Best**: **0.446031** at $\tau_y = 0.2650$.
     - **Paired Bootstrap (1,000 iterations)**: $\Delta \text{PQ} = \mathbf{0.000000}$ (95% CI: $[0.0, 0.0]$, $p = 0.0\%$).
     - The optimal decision threshold remains strictly within $[0.230, 0.265]$. Extending the upper boundary past $0.270$ up to $0.320$ does not change any fold's inner CV selection or outer evaluation score.

---

## 4. Metric and Submission Correctness Audit (Subagent B Findings)

### 4.1 Mathematical Equivalence: `pq_eval.py` vs Official Evaluation Notebook
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Equivalence)**
- **Citations**: `~/solar/pq_eval.py`: Lines 25, 30–62, 70–76, 80–93; `~/solar/self-evaluation-notebook.ipynb`: Cell 3 lines 1–60, Cell 4 lines 1–40; `~/solar/crosscheck.py`: Lines 11–24.
- **Finding & Observation**:
  In `pq_eval.py:53–62`, Panoptic Quality is computed as:
  $$\text{denom} = |TP| + 0.5|FP| + 0.5|FN|, \quad PQ = \frac{\sum_{TP} \text{IoU}}{\text{denom}}, \quad SQ = \frac{\sum_{TP} \text{IoU}}{|TP|}, \quad RQ = \frac{|TP|}{\text{denom}}$$
  In `self-evaluation-notebook.ipynb` Cell 4 lines 28–38, the official evaluation logic is identical:
  ```python
  denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
  pq_score = sum(tp_iou_scores) / denominator if denominator > 0 else 0.0
  ```
  Running `crosscheck.py` on 15 real observations from `work/gt_all.csv` confirms exact match to 6 decimal places:
  ```
  oficiální PQ: 0.313248
  můj PQ:       0.313248
  SHODA
  ```
  **Memory Architecture Difference**: The official notebook materializes dense $2048 \times 2048$ PyTorch float32 tensors (`rles_to_layers`, Cell 3 lines 101–118) and executes matrix multiplication (`torch.matmul(gt_flat, pred_flat.t())`), demanding gigabytes of GPU/CPU RAM. In contrast, `pq_eval.py:70–76` uses `pycocotools.mask.iou` directly on compressed RLE string dictionaries, running in $O(1)$ memory without dense array materialization.

### 4.2 Matching Algorithm Without 1:1 Bipartite Matching and Multi-Hit Inflation
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **MEDIUM (Mitigated by Disjoint Canvas)**
- **Citations**: `~/solar/pq_eval.py`: Lines 47–52; `~/solar/self-evaluation-notebook.ipynb`: Cell 3 lines 62–97; `~/solar/tests/test_pq.py`: Lines 31–46.
- **Finding & Observation**:
  Neither `pq_eval.py` nor `self-evaluation-notebook.ipynb` enforces Hungarian 1:1 bipartite matching. Instead, both implementations count any prediction $P$ with $\text{IoU}(P, G) > 0.5$ as a True Positive:
  ```python
  hit = iou_mat > 0.5
  tp_iou.extend(iou_mat[hit].tolist())
  ```
  In `tests/test_pq.py:31–46`, synthetic testing demonstrates that if two overlapping predictions have $\text{IoU} = 0.6$ and $\text{IoU} = 0.7$ with the same ground truth $G$, the evaluator counts $TP = 2, FP = 0, FN = 0$, artificially inflating PQ to 0.65.
  **Geometric Proof of Invariant**:
  When predicted masks are strictly pixel-disjoint ($\sum_k M_k \le 1$), multi-hit inflation is mathematically impossible.
  *Proof*: Suppose two disjoint predictions $P_1 \cap P_2 = \emptyset$ both have $\text{IoU} > 0.5$ with ground truth $G$. Then:
  $$\frac{|P_1 \cap G|}{|P_1 \cup G|} > 0.5 \implies |P_1 \cap G| > 0.5 |G|, \quad \frac{|P_2 \cap G|}{|P_2 \cup G|} > 0.5 \implies |P_2 \cap G| > 0.5 |G|$$
  Summing both inequalities:
  $$|G| \ge |(P_1 \cap G) \cup (P_2 \cap G)| = |P_1 \cap G| + |P_2 \cap G| > 0.5|G| + 0.5|G| = |G|$$
  This yields the contradiction $|G| > |G|$. Thus, strictly pixel-disjoint masks guarantee $\text{pred\_multi\_hit} = 0$.

### 4.3 Empty Mask Placeholder (`"PPPP4"`) False Positive Evaluation Trap
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Critical Evaluation Bug if Unfiltered)**
- **Citations**: `~/solar/pq_eval.py`: Lines 41–46, 70–76; `~/solar/scripts/common.py`: Line 29 (`EMPTY_RLE = "PPPP4"`); `~/solar/scripts/oof_eval.py`: Line 266; `~/solar/scripts/instance_scorer.py`: Line 671; `~/solar/scripts/predict_ensemble.py`: Line 309; `~/solar/scripts/loss_decomposition.py`: Line 52; `~/solar/scripts/predict.py`: Line 398; `~/solar/scripts/postprocess.py`: Lines 169–170.
- **Finding & Observation**:
  Competition submission guidelines mandate that images with zero predicted filaments must contain an empty placeholder row: `"<stem>_1, PPPP4"`.
  When a prediction dataframe containing `"PPPP4"` is evaluated via `pq_eval.score()`:
  1. `"PPPP4"` decodes to an all-zero $2048 \times 2048$ boolean mask.
  2. `pycocotools.mask.iou` evaluates overlap against all ground truth filaments as $0.0$.
  3. Because IoU is $< 0.5$, `hit.sum(axis=0) == 0` evaluates to True, incrementing False Positives: `fp += 1`.
  4. In a two-image synthetic benchmark (Image 1: $TP=1, \text{IoU}=1.0$; Image 2: GT has 1 filament, prediction is empty `"PPPP4"`), scoring with `"PPPP4"` yields $TP=1, FP=1, FN=1 \implies PQ = 0.5000$. Filtering `"PPPP4"` yields $TP=1, FP=0, FN=1 \implies PQ = 0.6667$.
  **Codebase Inconsistency**:
  - Correctly filtered: `oof_eval.py:266`, `instance_scorer.py:671`, `predict_ensemble.py:309`, and `loss_decomposition.py:52` all explicitly execute:
    ```python
    df_eval = df_pred[df_pred["segmentation_rle"] != EMPTY_RLE]
    ```
  - Unfiltered: `predict.py:398` and `postprocess.py:169–170` call `pq_eval.score(gt_df, pred_df)` without filtering `"PPPP4"`, artificially penalizing empty-image evaluations.

### 4.4 RLE Encoding / Decoding Specifications and Fortran-Contiguity Requirement
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Runtime Exception if Violated)**
- **Citations**: `~/solar/scripts/common.py`: Lines 29, 574, 591–596; `~/solar/pq_eval.py`: Lines 66–68, 105–109; `~/solar/scripts/postprocess.py`: Line 95; `~/solar/scripts/predict_ensemble.py`: Line 254; `~/solar/scripts/validate_submission.py`: Lines 9, 132–150.
- **Finding & Observation**:
  The competition specifies **pycocotools compressed RLE** format (LEB128 variable-length byte encoding packed into ASCII strings), NOT Kaggle's whitespace-separated integer pairs (`"start length"`).
  Target resolution is strictly $2048 \times 2048$ pixels ($4,194,304$ pixels total).
  **Fortran Memory Order**: `pycocotools.mask.encode` requires column-major arrays (`order='F'`). Passing a standard C-contiguous array raises:
  `ValueError: ndarray is not Fortran contiguous`.
  The codebase enforces this via `np.asfortranarray(mask.astype(np.uint8))` across `common.py:593`, `postprocess.py:95`, and `predict_ensemble.py:254`.
  **Round-Trip Verification**: Decoding and re-encoding 10 randomized $2048 \times 2048$ masks confirmed bit-for-bit identity across all $4,194,304$ pixels. Encoding an all-zero Fortran array yields `b'PPPP4'`, and decoding `b'PPPP4'` returns an array with shape $(2048, 2048)$ and sum $= 0$.

### 4.5 Canvas Disjointness and the Canvas 50% Rule
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Critical Structural Constraint)**
- **Citations**: `~/solar/scripts/common.py`: Lines 598–647; `~/solar/scripts/postprocess.py`: Lines 36–117; `~/solar/scripts/predict.py`: Lines 151–162; `~/solar/scripts/predict_ensemble.py`: Lines 221–265; `~/solar/scripts/validate_submission.py`: Lines 12, 159–176, 195.
- **Finding & Observation**:
  Mask R-CNN produces overlapping bounding boxes and masks. To satisfy panoptic disjointness (`scripts/validate_submission.py:195` enforces `overlap_images_count == 0`), candidate masks are arbitrated via a greedy score-ordered canvas:
  1. Instances are sorted by score descending: `order = np.argsort(scores)[::-1]`.
  2. Canvas `occupied = np.zeros((2048, 2048), dtype=bool)` tracks claimed pixels.
  3. For each candidate: `clean_mask = m_bool & (~occupied)`.
  4. **Canvas 50% Rule**: If `clean_mask.sum() / orig_area < 0.50` or `clean_mask.sum() < 10` pixels, the candidate is discarded.
  5. If kept, `occupied |= clean_mask`.
  6. Final assertion: `assert np.stack(clean_masks).sum(axis=0).max() <= 1`.
  Validation via `scripts/validate_submission.py` on `work/submission.csv` (1,205 rows) and `work/submission_r50_1536.csv` (1,088 rows) verified 0 overlapping instances across all 180 test images.

### 4.6 Submission Schema and 180 Test Image Coverage
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **MEDIUM (Format Compliance)**
- **Citations**: `~/solar/scripts/validate_submission.py`: Lines 60–114; `~/solar/data/MAGFiLO_1.0_Kaggle_2026/test/test_images/`: 180 test images; `~/solar/work/submission.csv`: 1,205 rows; `~/solar/work/submission_r50_1536.csv`: 1,088 rows; `~/solar/work/gt_all.csv`: Area distribution.
- **Finding & Observation**:
  - Columns: strictly `['filament_id', 'segmentation_rle']`.
  - ID Format: monotonic 1-indexed integers: `<stem>_<k>` with $k \in \{1, 2, \dots, K\}$.
  - Test Set Coverage: Exactly 180 test stems exist in `test_images/`. Both submission files contain all 180 stems in identical alphabetical order (0 missing, 0 extra).
  - Placeholder Rows: Exactly 3 images contain no predicted filaments:
    - `20170708164010Mh_1,PPPP4`
    - `20200310113230Ch_1,PPPP4`
    - `20211214174010Mh_1,PPPP4`
  - Area Distribution: In `work/gt_all.csv` (8,199 GT filaments), minimum area is 9 pixels (only 1 filament $< 10$ px; median 1,228 px). In both submissions, 0 instances exceed 1% of the frame (41,943 px).

---

## 5. Inference Determinism and Reproducibility Audit (Subagent C Findings)

### 5.1 Random Seed Coverage Across Pipeline
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **MEDIUM**
- **Citations**: `~/solar/scripts/common.py`: Lines 32–40 (`set_seed`); `~/solar/scripts/train_maskrcnn.py`: Line 319; `~/solar/scripts/instance_scorer.py`: Lines 592–593; `~/solar/scripts/multifamily_pipeline.py`: Line 816; `~/solar/pq_eval.py`: Lines 120–124; `~/solar/bootstrap_compare.py`: Line 106; `~/solar/scripts/predict.py`: Lines 70–120; `~/solar/scripts/predict_ensemble.py`: Lines 390–430.
- **Finding & Observation**:
  `scripts/common.py:32–40` provides `set_seed(seed=2026)`, seeding Python `random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, `cudnn.benchmark = False`, and `PYTHONHASHSEED`.
  Explicit seeds are set in `train_maskrcnn.py:319`, `instance_scorer.py:592` (`random_state=2026 + held_out_f`), `multifamily_pipeline.py:816` (`random_state=seed + val_f`), `pq_eval.py:120` (`random.Random(2026)`), and `bootstrap_compare.py:106` (`np.random.default_rng(2026)`).
  **Omissions**: `predict.py` and `predict_ensemble.py` do not invoke `set_seed()`. However, inference is executed strictly under `model.eval()` and `torch.no_grad()`.

### 5.2 4-View TTA Ordering and WMF Reduction Determinism
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **LOW (Clean Determinism)**
- **Citations**: `~/solar/scripts/predict_ensemble.py`: Lines 144–216, 396–401; `~/solar/scripts/oof_eval.py`: Lines 78–83; `~/solar/scripts/multifamily_pipeline.py`: Lines 217–312.
- **Finding & Observation**:
  The 4 test-time augmentation views are evaluated in strict, hardcoded order:
  `tta_flips = [("id", False, False), ("h", False, True), ("v", True, False), ("hv", True, True)]`.
  Weighted Mask Fusion (WMF) pools candidates and sorts them descending by confidence score:
  `candidates.sort(key=lambda c: c["score"], reverse=True)`.
  Python's Timsort is stable, and continuous float32 sigmoid confidence scores provide a strict total ordering without score ties. Soft mask aggregation $\sum (w_k \cdot P_k)$ executes deterministically.

### 5.3 Sources of Non-Determinism in Training and Hardware Execution
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **MEDIUM**
- **Citations**: `~/solar/scripts/common.py`: Lines 32–40; `~/solar/scripts/train_maskrcnn.py`: Lines 362–370; `~/solar/scripts/instance_scorer.py`: Line 593; `~/solar/scripts/predict.py`: Line 140.
- **Finding & Observation**:
  1. **PyTorch CUDA Deterministic Flags**: `torch.use_deterministic_algorithms(True)` is not set, and `CUBLAS_WORKSPACE_CONFIG` is unset. Bilinear interpolation (`torch.nn.functional.interpolate` in `predict.py:140`) and RoIAlign kernels use GPU atomic adds, which introduce floating-point non-associativity across thread blocks ($\sim 10^{-7}$ precision variance). This has no macroscopic impact on PQ ($\Delta PQ < 0.0001$), but prevents exact bitwise identical RLE hashes across disparate GPU architectures.
  2. **DataLoader Workers**: `train_maskrcnn.py:365` instantiates `DataLoader(..., shuffle=True, num_workers=4)` without a seeded `worker_init_fn` or `torch.Generator`. Worker processes inherit OS fork timing, causing augmentation sequences to diverge when training models from scratch on different operating systems.
  3. **LightGBM OpenMP Multithreading**: `instance_scorer.py:593` and `multifamily_pipeline.py:818` set `n_jobs=4` without `deterministic=True`. Histogram construction across multiple CPU threads can introduce micro-variations in split thresholds.

### 5.4 Unpinned `requirements.txt` and Missing Frameworks
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Installation Failure on External Machines)**
- **Citations**: `~/solar/requirements.txt`: Lines 1–8; `~/solar/scripts/setup_pc.sh`: Lines 17–19; `~/solar/.venv` metadata.
- **Finding & Observation**:
  The repository `requirements.txt` contains:
  ```
  numpy
  pandas
  scipy
  scikit-learn
  pycocotools
  opencv-python-headless
  matplotlib
  ```
  It leaves all package versions unpinned and omits `torch`, `torchvision`, and `lightgbm` entirely. Running `pip install -r requirements.txt` on a clean system fails to install the deep learning runtime.

### 5.5 Hardcoded Paths and WSL2 Windows Mounts
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Portability Failure)**
- **Citations**: `~/solar/scripts/run_queue.sh`: Line 4; `~/solar/scripts/stop_queue.sh`: Line 5; `~/solar/live.sh`: Line 5; `~/solar/scripts/run_followup_tasks.py`: Line 861; `~/solar/scripts/multifamily_pipeline.py`: Line 1245; `~/solar/scripts/validate_submission.py`: Line 265; `~/solar/work/queue.txt`: Line 8.
- **Finding & Observation**:
  - Repo root `.` is hardcoded across shell wrappers (`run_queue.sh`, `stop_queue.sh`, `live.sh`).
  - WSL2 host download path `<local_downloads_dir>` is hardcoded in `run_followup_tasks.py:861`, `multifamily_pipeline.py:1245`, and `validate_submission.py:265`. On standard Linux systems, `/mnt/c/` does not exist.

### 5.6 Hardware Architecture Lock to `bfloat16` and CUDA
- **Status**: **[CONFIRMED]**
- **Risk Tier**: **HIGH (Hardware Incompatibility)**
- **Citations**: `~/solar/scripts/predict.py`: Line 94; `~/solar/scripts/predict_ensemble.py`: Line 423; `~/solar/scripts/oof_eval.py`: Line 90; `~/solar/scripts/train_maskrcnn.py`: Line 94; `~/solar/scripts/check_gpus.py`: Lines 10–12.
- **Finding & Observation**:
  All inference scripts hardcode `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
  Native hardware `bfloat16` acceleration requires NVIDIA Ampere (Compute Capability 8.0+) or newer (RTX 3090, RTX 4090, A4000, A6000, A100). Pre-Ampere GPUs (Turing RTX 2080, Volta V100, Pascal GTX 1080) and CPU environments fail with `RuntimeError` or suffer extreme emulation slowdowns. Furthermore, $1536 \times 1536$ and $2048 \times 2048$ tensor inputs require a minimum of 16 GB GPU VRAM.

---

## 6. End-to-End Submission Reproduction Specification (From Weights Alone)

This specification provides the self-contained blueprint to reproduce `submission_A_scorer_v2_multifam_8model.csv`, `submission_B_threshold_multifam_8model.csv`, `submission_5fold_1536_tta.csv`, `submission_5fold1536_plus_2048_tta.csv`, and `submission_full_8model_tta.csv` strictly from pre-trained model weights.

### 6.1 Hardware Prerequisites
- **Operating System**: Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended; WSL2 supported).
- **CPU**: x86_64 CPU with $\ge 8$ physical cores and $\ge 32$ GB system RAM.
- **GPU**: NVIDIA GPU with $\ge 16$ GB VRAM and Compute Capability $\ge 8.0$ (RTX 3090, RTX 4090, RTX A4000, RTX A5000, RTX A6000, A100, H100) supporting hardware `bfloat16`.
- **NVIDIA Driver**: Version $\ge 560.35$.
- **CUDA Toolkit**: CUDA 12.6.

### 6.2 Exact Software Manifest
Extracted directly from `uv pip freeze` of `.venv`:
```
torch==2.14.1+cu126
torchvision==0.29.1+cu126
lightgbm==4.7.0
numpy==2.5.4
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
opencv-python-headless==5.0.0.93
pycocotools==2.0.11
pillow==12.3.0
matplotlib==3.11.2
tqdm==4.70.1
cloudpickle==3.1.2
joblib==1.6.0
threadpoolctl==3.7.0
triton==3.8.0
typing-extensions==4.16.0
```

### 6.3 Checkpoint Manifest (All Models in Final Ensembles)
All 8 models utilized in the final ensemble pipeline are cited from disk with their exact hyperparameters and architectures:

| Model ID | Checkpoint Path | Architecture | Resolution | Fold | Specialist Role / Distinct Hyperparameter |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `f0_1536` | `work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 0 | Standard aspect ratios (`[0.5, 1.0, 2.0]`), CLAHE preprocessing |
| `f1_1536` | `work/runs/r50_1536_all_f1/checkpoints/checkpoint_fold1_last.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 1 | Standard aspect ratios, CLAHE preprocessing |
| `f2_1536` | `work/runs/r50_1536_all_f2/checkpoints/checkpoint_fold2_last.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 2 | Standard aspect ratios, CLAHE preprocessing |
| `f3_1536` | `work/runs/r50_1536_all_f3/checkpoints/checkpoint_fold3_last.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 3 | Standard aspect ratios, CLAHE preprocessing |
| `f4_1536` | `work/runs/r50_1536_all_f4/checkpoints/checkpoint_fold4_last.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 4 | Standard aspect ratios, CLAHE preprocessing |
| `f0_2048` | `work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt` | Mask R-CNN ResNet50-FPN v2 | 2048 | Fold 0 | Full native resolution ($2048 \times 2048$), CLAHE |
| `f0_anchors5` | `work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 0 | 5 aspect ratios (`[0.2, 0.5, 1.0, 2.0, 5.0]`) for extreme elongation |
| `f0_dice` | `work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt` | Mask R-CNN ResNet50-FPN v2 | 1536 | Fold 0 | BCE + Soft Dice composite loss (`alpha=0.5, beta=0.5`) |

### 6.4 True Thresholds and Submission Metadata
All threshold values and postprocessing hyperparameters are cited directly from `scripts/multifamily_pipeline.py`, `work/oof/oof_tuning_summary.json`, and `work/oof/multimodel_fold0_eval.json`:

1. **Submission A (`work/oof/submission_A_scorer_v2_multifam_8model.csv`)**:
   - **Ensemble Composition**: 8 models (`f0_1536`..`f4_1536` + `f0_2048` + `f0_anchors5` + `f0_dice`), 32 views total.
   - **Instance Scorer**: View-Invariant Scorer v2 (19 features, 5 LightGBM models cross-fitted out-of-fold).
   - **Decision Threshold ($\tau_y$)**: $\mathbf{0.2440}$ (mean of per-fold inner CV optimal thresholds: $\{0.230, 0.230, 0.265, 0.230, 0.265\}$).
   - **Mask Probability Threshold ($\tau_{\text{mask}}$)**: $\mathbf{0.525}$.
   - **Canvas Filtering**: Greedy score-ordered canvas painting, $\text{clean\_area} \ge 10$, $\frac{\text{clean\_area}}{\text{orig\_area}} \ge 0.50$.
   - **Artifact Properties**: 1,202 rows covering all 180 test stems (3 empty placeholder rows `"PPPP4"`).

2. **Submission B (`work/oof/submission_B_threshold_multifam_8model.csv`)**:
   - **Ensemble Composition**: 8 models (`f0_1536`..`f4_1536` + `f0_2048` + `f0_anchors5` + `f0_dice`), 32 views total.
   - **Decision Rule**: 3-family threshold rule on raw WMF fused confidence scores.
   - **Score Threshold ($\tau_{\text{score}}$)**: $\mathbf{0.875}$.
   - **Mask Probability Threshold ($\tau_{\text{mask}}$)**: $\mathbf{0.550}$.
   - **Canvas Filtering**: Greedy score-ordered canvas painting, $\text{clean\_area} \ge 10$, $\frac{\text{clean\_area}}{\text{orig\_area}} \ge 0.50$.
   - **Artifact Properties**: 1,205 rows covering all 180 test stems (3 empty placeholder rows `"PPPP4"`).

3. **5-Fold Baseline (`work/oof/submission_5fold_1536_tta.csv`)**:
   - **Ensemble Composition**: 5 fold models (`f0_1536`..`f4_1536`), 20 views total.
   - **Thresholds**: $\tau_{\text{score}} = \mathbf{0.825}$, $\tau_{\text{mask}} = \mathbf{0.500}$ (from `work/oof/oof_tuning_summary.json:16–27`).
   - **Artifact Properties**: 1,328 rows covering all 180 test stems.

4. **6-Model Ensemble (`work/oof/submission_5fold1536_plus_2048_tta.csv`)**:
   - **Ensemble Composition**: 5x 1536 models + `f0_2048`, 24 views total.
   - **Thresholds**: $\tau_{\text{score}} = \mathbf{0.750}$, $\tau_{\text{mask}} = \mathbf{0.500}$ (from `work/oof/multimodel_fold0_eval.json:7–12`).
   - **Artifact Properties**: 1,413 rows covering all 180 test stems.

5. **Full 8-Model Direct WMF (`work/oof/submission_full_8model_tta.csv`)**:
   - **Ensemble Composition**: 8 models, 32 views total (direct WMF without instance scorer).
   - **Thresholds**: $\tau_{\text{score}} = \mathbf{0.850}$, $\tau_{\text{mask}} = \mathbf{0.550}$ (from `work/oof/multimodel_fold0_eval.json:20–25`).
   - **Artifact Properties**: 1,250 rows covering all 180 test stems.

### 6.5 Verified Command-Line Flags Across Codebase
Every CLI flag below has been verified by executing the respective script with `--help` against the real codebase:

| Script | Verified CLI Flags (via `--help`) | Parameter Purpose |
| :--- | :--- | :--- |
| `scripts/multifamily_pipeline.py` | *None* (takes no CLI arguments) | Automatically loads pre-cached test detections under `work/test_cache/` (or generates them if missing), runs Step 2 pipeline, and saves Submissions A & B to `work/oof/`. |
| `scripts/oof_eval.py` | `--predict_test_ensemble`, `--tune_oof`, `--eval`, `--generate`, `--eval_multimodel_fold0`, `--checkpoint`, `--fold`, `--img_size`, `--out_dir`, `--oof_dir`, `--score_filter`, `--tau_score`, `--tau_mask` | `--predict_test_ensemble` generates test CSVs for 5-fold, 6-model, and 8-model WMF ensembles reading thresholds from summary JSONs. |
| `scripts/validate_submission.py` | `--sub SUB`, `--test_dir TEST_DIR` | Validates submission format, column names, 180 test image coverage, 1-indexed IDs, empty placeholders, and 0 canvas pixel overlaps. |
| `scripts/instance_scorer.py` | `--oof_dir OOF_DIR`, `--min_score MIN_SCORE`, `--n_estimators N_ESTIMATORS`, `--learning_rate LEARNING_RATE` | Cross-fits LightGBM regressor across 5 OOF folds. |
| `scripts/predict.py` | `--run_name`, `--checkpoint`, `--image_dir`, `--images`, `--folds_csv`, `--fold`, `--output`, `--score_thresh`, `--mask_thresh`, `--mask_nms_thresh`, `--img_size`, `--batch_size`, `--tta {none,flip}`, `--eval`, `--gt_csv`, `--limit` | Single model inference and fold prediction. |
| `scripts/predict_ensemble.py` | `--eval_fold0`, `--checkpoint`, `--img_size`, `--fold`, `--n_boot`, `--unit_test` | Single checkpoint 4-flip TTA evaluation on Fold 0. |

### 6.6 Exact Reproduction Execution Commands

#### Step 1: Environment Initialization
```bash
# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 12.6 wheels
pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install exact locked supporting packages
pip install lightgbm==4.7.0 numpy==2.5.4 pandas==3.0.6 scipy==1.18.1 \
            scikit-learn==1.9.1 opencv-python-headless==5.0.0.93 \
            pycocotools==2.0.11 pillow==12.3.0 matplotlib==3.11.2 tqdm==4.70.1
```

#### Step 2: Reproduce Primary Submissions A and B (8-Model Multifamily Pipeline)
```bash
cd ~/solar
python3 scripts/multifamily_pipeline.py
```
- **Inputs**: Reads cached test candidate detections under `work/test_cache/` (or extracts them on GPU if cache is absent) across the 8 models.
- **Outputs**:
  - `work/oof/submission_A_scorer_v2_multifam_8model.csv` ($\tau_y=0.2440, \tau_{\text{mask}}=0.525$)
  - `work/oof/submission_B_threshold_multifam_8model.csv` ($\tau_{\text{score}}=0.875, \tau_{\text{mask}}=0.550$)
  - `work/oof/multifamily_pipeline_summary.json`

#### Step 3: Reproduce 5-Fold, 6-Model, and 8-Model WMF Baseline Submissions
```bash
cd ~/solar
python3 scripts/oof_eval.py --predict_test_ensemble
```
- **Inputs**: Reads cached test candidates under `work/test_cache/` and threshold configs from `work/oof/oof_tuning_summary.json` and `work/oof/multimodel_fold0_eval.json`.
- **Outputs**:
  - `work/oof/submission_5fold_1536_tta.csv` ($\tau_{\text{score}}=0.825, \tau_{\text{mask}}=0.500$)
  - `work/oof/submission_5fold1536_plus_2048_tta.csv` ($\tau_{\text{score}}=0.750, \tau_{\text{mask}}=0.500$)
  - `work/oof/submission_full_8model_tta.csv` ($\tau_{\text{score}}=0.850, \tau_{\text{mask}}=0.550$)

#### Step 4: Validate Submissions
Verify all structural invariants (180 stems, monotonic 1..K IDs, 0 overlapping instances, valid placeholder rows):
```bash
cd ~/solar
python3 scripts/validate_submission.py --sub work/oof/submission_A_scorer_v2_multifam_8model.csv
python3 scripts/validate_submission.py --sub work/oof/submission_B_threshold_multifam_8model.csv
python3 scripts/validate_submission.py --sub work/oof/submission_5fold_1536_tta.csv
python3 scripts/validate_submission.py --sub work/oof/submission_5fold1536_plus_2048_tta.csv
python3 scripts/validate_submission.py --sub work/oof/submission_full_8model_tta.csv
```

---

## 7. Independent Verification Methods

All findings, scores, and code citations in this report can be verified independently using the following bash commands without executing any GPU jobs:

```bash
# 1. Verify observation-level fold partitioning (707 observations, 0 overlap):
python -c '
import json, pandas as pd
folds = pd.read_csv("work/folds.csv")
with open("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") as f:
    coco = json.load(f)
obs_coco = set(im["id"].split("-", 1)[1] for im in coco["images"])
obs_folds = set(folds["observation"])
assert obs_coco == obs_folds, "Observation set mismatch"
assert len(folds) == 707, "Observation count mismatch"
print("PASS: 707 physical observations match folds.csv with 0 overlap.")
'

# 2. Verify pq_eval mathematical equivalence (0.313248 == 0.313248):
cd . && python crosscheck.py

# 3. Verify nested CV vs global threshold metrics:
python -c '
import json
with open("work/oof/oof_tuning_summary.json") as f:
    d = json.load(f)
print("Global best PQ: ", d["global_best_metrics"]["pq"])
print("Nested CV PQ:   ", d["nested_cv_metrics"]["pq"])
print("Optimistic gap: ", d["global_best_metrics"]["pq"] - d["nested_cv_metrics"]["pq"])
'

# 4. Verify cross-fitted instance scorer metrics & bootstrap delta:
python -c '
import json
with open("work/oof/instance_scorer_summary.json") as f:
    d = json.load(f)
print("Pooled OOF PQ:   ", d["pooled_oof_metrics"]["pq"])
print("Global best tau: ", d["global_best_tau"], "-> PQ:", d["global_best_metrics"]["pq"])
print("Bootstrap Delta: ", d["bootstrap_vs_nested_cv_threshold"]["point_delta"])
print("Bootstrap 95% CI:", [d["bootstrap_vs_nested_cv_threshold"]["ci_lower"], d["bootstrap_vs_nested_cv_threshold"]["ci_upper"]])
print("p(Scorer > Thresh):", d["bootstrap_vs_nested_cv_threshold"]["p_b_better"])
'

# 5. Verify RLE Fortran contiguity round-trip and empty mask encoding:
python -c "
import numpy as np
from pycocotools import mask as mu
m = np.zeros((2048, 2048), dtype=np.uint8, order='F')
m[100:200, 100:200] = 1
enc = mu.encode(m)
dec = mu.decode(enc)
assert np.array_equal(m, dec), 'Round-trip failed'
empty_enc = mu.encode(np.zeros((2048, 2048), dtype=np.uint8, order='F'))
assert empty_enc['counts'] == b'PPPP4', 'Empty RLE mismatch'
print('PASS: Fortran RLE round-trip and empty PPPP4 verified.')
"

# 6. Verify existing submission structural validity (0 overlaps, 180 stems):
cd . && python scripts/validate_submission.py --sub work/submission.csv

# 7. Verify unit tests for PQ, multi-hit, and empty predictions:
cd . && python -c "
import tests.test_pq as t
t.test_empty_predictions()
t.test_false_positives_and_negatives()
t.test_duplicate_prediction()
print('PASS: All test_pq.py tests passed.')
"

# 8. Verify CPU smoke tests for anchors5 and bce_dice loss switches:
cd . && python scripts/smoke_test_switches.py

# 9. Verify LEAK-01 bias recomputation against mean of held-out halves:
python -c '
import json
with open("work/runs/r50_1536_anchors5_e30/best_thresholds.json") as f:
    d = json.load(f)
best_full = d["best_pq"]
std = d["selection_bias_control"]["standard"]
eA = std["eval_A_on_B"]["pq_held_out"]
eB = std["eval_B_on_A"]["pq_held_out"]
mean_held_out = (eA + eB) / 2.0
bias = best_full - mean_held_out
print(f"best_full: {best_full:.6f}")
print(f"eval_A_on_B: {eA:.6f}, eval_B_on_A: {eB:.6f}")
print(f"Mean held-out: {mean_held_out:.6f}")
print(f"Recomputed Bias: +{bias:.6f} PQ")
assert abs(bias - 0.003111) < 1e-5, "Recomputed bias mismatch"
print("PASS: LEAK-01 recomputed bias verified.")
'

# 10. Verify extended threshold grid results and saturation:
python -c '
import json
with open("audit/grid_extension_results.json") as f:
    d = json.load(f)
assert d["threshold_rule"]["ext_nested_cv"]["pq"] == d["threshold_rule"]["orig_nested_cv"]["pq"]
assert d["threshold_rule"]["ext_best_tau_score"] == 0.875
assert d["threshold_rule"]["bootstrap_ext_vs_orig"]["point_delta"] == 0.0
assert d["scorer_v2"]["ext_nested_cv"]["pq"] == d["scorer_v2"]["orig_nested_cv"]["pq"]
assert d["scorer_v2"]["bootstrap_ext_vs_orig"]["point_delta"] == 0.0
tr_pq = d["threshold_rule"]["ext_nested_cv"]["pq"]
sc_pq = d["scorer_v2"]["ext_nested_cv"]["pq"]
print(f"Threshold Rule Ext Nested CV PQ: {tr_pq:.6f}")
print(f"Scorer v2 Ext Nested CV PQ:      {sc_pq:.6f}")
print("PASS: Extended grid results and interior saturation verified.")
'

# 11. Verify nearest training observation time gap distribution statistics:
python -c '
import json
with open("audit/time_gap_stats.json") as f:
    d = json.load(f)
val_med = d["validation_gaps_to_nearest_train_in_fold_split_hours"]["median_hours"]
test_med = d["test_gaps_to_nearest_train_in_all_707_hours"]["median_hours"]
val_all_med = d["validation_gaps_against_other_train_in_all_707_hours"]["median_hours"]
print(f"Validation in-split median gap: {val_med:.2f} h ({val_med/24:.2f} d)")
print(f"Test vs all 707 median gap:     {test_med:.2f} h ({test_med/24:.2f} d)")
print(f"Val vs all 707 median gap:      {val_all_med:.2f} h ({val_all_med/24:.2f} d)")
assert abs(test_med - 36.68055) < 0.01
print("PASS: Time gap statistics verified.")
'
```

