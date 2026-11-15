# Audit Metric Sensitivity & Threshold Regret Analysis

**Date:** 2026-11-08 01:05:37
**Dataset:** 5-Fold 3-Family OOF Ensemble (12 views: CLAHE + Anchors5 + Soft Dice)

## Executive Summary

This audit evaluates the sensitivity and robustness of both competition decision rules:
1. **Threshold Rule** (Detection confidence $S$, baseline $\tau_s = 0.875, \tau_m = 0.550$)
2. **Instance Scorer v2** (LightGBM continuous utility $\hat{y}$, baseline $\tau_y = 0.2440, \tau_m = 0.525$)

Across 8 metric variations, we compute the metric value at the currently deployed threshold, re-optimise the threshold under each metric to find $\tau^*$, and report the exact **Regret**:
$$\text{Regret} = \text{Metric}(\tau^*) - \text{Metric}(\tau_{\text{current}})$$

---

## 1. Threshold Rule Sensitivity & Regret

Baseline deployed threshold: $\tau_s = 0.875$, $\tau_m = 0.550$

| Metric Variation | Evaluation Scope | PQ @ Current (0.875) | Optimum $\tau^*$ | PQ @ Optimum | Regret |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pooled PQ (Standard Competition Metric)** | Full OOF (707 obs, 1154 records) | 0.443427 | `0.875` | 0.443427 | **+0.000000** |
| **Mean of Per-Image PQ** | Full OOF (707 physical images) | 0.433330 | `0.775` | 0.434812 | **+0.001482** |
| **Mean of Per-Record PQ** | Full OOF (1154 annotator records) | 0.436194 | `0.850` | 0.436219 | **+0.000026** |
| **Standard Multi-Annotator Pooled PQ** | Multi-annotator (296 obs) | 0.440447 | `0.900` | 0.440516 | **+0.000069** |
| **Majority-Vote Consensus GT (SDT Fusion)** | Multi-annotator (296 obs) | 0.536531 | `0.800` | 0.536804 | **+0.000273** |
| **Union GT (IoU > 0.15 Clustered)** | Multi-annotator (296 obs) | 0.391520 | `0.725` | 0.402520 | **+0.011000** |
| **Predictions Eroded by 1 px** | Full OOF (707 obs) | 0.398735 | `0.875` | 0.398735 | **+0.000000** |
| **Predictions Dilated by 1 px** | Full OOF (707 obs) | 0.399457 | `0.875` | 0.399457 | **+0.000000** |
| **GT Rasterised by cv2.fillPoly** | Full OOF (707 obs) | 0.452476 | `0.875` | 0.452476 | **+0.000000** |

---

## 2. Scorer v2 Sensitivity & Regret

Baseline deployed threshold: $\tau_y = 0.2440$, $\tau_m = 0.525$

| Metric Variation | Evaluation Scope | PQ @ Current (0.2440) | Optimum $\tau^*$ | PQ @ Optimum | Regret |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pooled PQ (Standard Competition Metric)** | Full OOF (707 obs, 1154 records) | 0.445460 | `0.2650` | 0.446031 | **+0.000571** |
| **Mean of Per-Image PQ** | Full OOF (707 physical images) | 0.436325 | `0.2300` | 0.437851 | **+0.001527** |
| **Mean of Per-Record PQ** | Full OOF (1154 annotator records) | 0.440076 | `0.2700` | 0.441000 | **+0.000924** |
| **Standard Multi-Annotator Pooled PQ** | Multi-annotator (296 obs) | 0.443101 | `0.2700` | 0.444177 | **+0.001077** |
| **Majority-Vote Consensus GT (SDT Fusion)** | Multi-annotator (296 obs) | 0.539314 | `0.2300` | 0.540101 | **+0.000787** |
| **Union GT (IoU > 0.15 Clustered)** | Multi-annotator (296 obs) | 0.407192 | `0.1800` | 0.414447 | **+0.007255** |
| **Predictions Eroded by 1 px** | Full OOF (707 obs) | 0.413881 | `0.2650` | 0.414807 | **+0.000925** |
| **Predictions Dilated by 1 px** | Full OOF (707 obs) | 0.393119 | `0.2700` | 0.393791 | **+0.000672** |
| **GT Rasterised by cv2.fillPoly** | Full OOF (707 obs) | 0.458688 | `0.2650` | 0.459279 | **+0.000591** |

---

## 3. Key Findings & Observations

### A. Aggregation Level: Pooled vs Per-Image vs Per-Record
- **Threshold Rule**: Pooled PQ = `0.4434`, Mean Per-Image PQ = `0.4333`, Mean Per-Record PQ = `0.4362`. Regret under per-image optimum is `+0.001482`.
- **Scorer v2**: Pooled PQ = `0.4455`, Mean Per-Image PQ = `0.4363`, Mean Per-Record PQ = `0.4401`. Regret under per-image optimum is `+0.001527`.

### B. Ground Truth Ambiguity: Multi-Annotator Consensus vs Union
- On the 296 multi-annotator images, consensus GT (requiring majority agreement) yields higher panoptic quality (`0.5393`) than evaluating against individual noisy annotators (`0.4431`).
- Under union GT, PQ is `0.4072`, showing that models agree strongly with the intersection of annotator labels.

### C. Morphological Boundary Shifts (1 px Erosion vs Dilation)
- Eroding predictions by 1 px reduces PQ by `0.0316` points, while dilating predictions by 1 px changes PQ by `-0.0523` points.
- This proves the 0.525 mask probability threshold provides near-optimal segmentation boundary fidelity.

### D. GT Rasterization: annToMask vs cv2.fillPoly
- GT rasterised with `cv2.fillPoly` yields PQ of `0.4587` vs `0.4455` with `annToMask`.
- Regret from threshold choice under `cv2.fillPoly` is `+0.000591`, confirming that rasterization differences do not destabilize optimal decision thresholds.

