# Experiments Log

| Date | Experiment / Config | Commit | Fold | PQ / SQ / RQ | Threshold | Runtime | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-12 | Mask R-CNN ResNet50-FPN v2 (smoke test: 2 epochs, 50 imgs, batch 4, AdamW, AMP bf16, target=all) | 3edf72a | 0 | PQ: 0.2003 / SQ: 0.5983 / RQ: 0.3348 (tuned: 0.2658 / 0.6059 / 0.4387) | score=0.35, mask=0.5 (tuned: score=0.80, mask=0.50) | ~5.29s/epoch (0.41s/iter train, 0.17s/img infer) | Smoke test passed. End-to-end pipeline (train, tune, predict, score) verified. |
