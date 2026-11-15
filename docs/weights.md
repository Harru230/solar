# Model Weights & Checkpoints Protocol

This document details how model weights are structured, pruned, hosted, and verified for the Solar Filament Detection & Segmentation production repository.

---

## 1. Self-Contained In-Repo Weights (< 100 MB Per File)

Full unpruned training checkpoints contain AdamW optimizer momentum/variance buffers totaling **~16.5 GB** (~550 MB per Mask R-CNN checkpoint across 30 detectors, plus Phase 3B UNets and refiners). To satisfy competition self-containment (**every weight needed to reproduce Rules A, B, and C stored directly in the repository without external files**) AND GitHub's strict **100 MB per-file push limit** without Git LFS:

* Stripping optimizer buffers reduces each Mask R-CNN detector from `~550 MB` to `175.40 MB` (`float32`), and exporting in **`float16`** halves it to **`~92.03 MB`** (`87.76 MiB`).
* Stripping & converting `ResNet50SemanticUNet` to `float16` yields **`65.25 MB`** per fold (`326.26 MB` total across 5 folds).
* Converting `ResNeXt101SemanticUNet` (`183.2 MiB` FP16) into deterministic `<= 96.1 MB` chunks (`resnext101_unet_f{0..4}_fp16.pt.part00`, `.part01`, max chunk `96,064,456` bytes = `91.61 MiB`) keeps **all 81 files under 100 MB** (`< 100,000,000` bytes).
* `weights/` in-repo total: **81 verified files**, **`4,105,994,780` bytes (`4,105.99 MB` / `3.824 GiB`)**:
  - `weights/r50_*_f{0..4}.pt` (30 Stage-1 FP16 checkpoints): `2,760.75 MB`
  - `weights/phase3/semantic_unet_f{0..4}_fp16.pt` (5 files): `326.26 MB`
  - `weights/phase3/resnext101_unet_f{0..4}_fp16.pt.part0{0,1}` (10 shard files): `960.64 MB`
  - `weights/phase3/crop_refiner_f{0..4}_fp16.pt` (5 files): `19.87 MB`
  - `weights/phase3/wide_refiner_v2_f{0..4}_fp16.pt` (5 files): `34.53 MB`
  - `weights/phase3/lgb_s27_yhyb_s{2026..2030}_f{0..4}.txt` + `lgb_metadata.json` (26 files): `3.94 MB`

Verify all 81 committed weights offline:
```bash
(cd weights && sha256sum -c checksums.sha256)
```

### External GitHub Releases Backup (Optional FP32 Archives)
Optional **FP32** backup archives (**7 tarballs**, under 2 GB each) remain mirrored on GitHub Releases (`Harru230/solar`, tag `v1.0.0`):

---

## 2. Packaging Per Family (Under 2 GB Each)

Weights are partitioned into 6 distinct family archives so that each asset is strictly under the 2 GB limit:

| Archive Name | Contents | Models Included | Uncompressed | Compressed | Release URL |
|---|---|---|:---:|:---:|---|
| `weights_family_clahe.tar.gz` | Multi-Scale CLAHE 1536px | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.8 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_clahe.tar.gz) |
| `weights_family_anchors5.tar.gz` | 5 Anchor Aspect Ratios 1536px | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.9 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_anchors5.tar.gz) |
| `weights_family_dice.tar.gz` | BCE + Soft Dice Loss 1536px | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.8 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_dice.tar.gz) |
| `weights_family_2048.tar.gz` | Native 2048px High Resolution | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.5 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_2048.tar.gz) |
| `weights_family_a5dice.tar.gz` | 5 Anchors + Soft Dice 1536px | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.9 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_a5dice.tar.gz) |
| `weights_family_1792.tar.gz` | Multi-Scale CLAHE 1792px | Folds 0, 1, 2, 3, 4 | ~877 MB | **813.7 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_1792.tar.gz) |
| `weights_family_refiner.tar.gz` | Phase 3B Semantic UNets + Refiners + S27 LGB | 20 UNet/Refiner FP16 + 25 LGB | ~1,280 MB | **1,183 MB** | [Download](https://github.com/Harru230/solar/releases/download/v1.0.0/weights_family_refiner.tar.gz) |

> [!NOTE]
> Every individual release archive is strictly under GitHub's 2.0 GB per-file release asset limit.

---

## 3. Automated Weights Download

To automatically download, unpack, and verify model weights into `work/checkpoints/`:

```bash
# Download all 7 release archives from GitHub Releases (30 Mask R-CNN + Phase 3B refiners)
bash scripts/download_weights.sh --repo Harru230/solar --all

# Or download a specific family archive (e.g. baseline CLAHE or Phase 3B refiners)
bash scripts/download_weights.sh --repo Harru230/solar --family clahe
bash scripts/download_weights.sh --repo Harru230/solar --family refiner
```

### Options:
- `--all`: Download and extract all 7 archives (`clahe`, `anchors5`, `dice`, `2048`, `a5dice`, `1792`, `refiner`).
- `--family <name>`: Download a specific family (`clahe`, `anchors5`, `dice`, `2048`, `a5dice`, `1792`, `refiner`).
- `--repo <user/repo>`: GitHub repository target (default: `Harru230/solar`).
- `--tag <tag>`: Release tag (default: `v1.0.0`).
- `--dir <path>`: Download destination directory (default: `work/checkpoints`).
- `--force` (`-f`): Re-download and overwrite existing weights.

The script automatically verifies SHA-256 hashes against `work/checkpoints/checksums.sha256`.

---

## 4. Manual Weights Placement

For air-gapped environments or manual browser downloads:

1. Download the required archive(s) from:
   `https://github.com/Harru230/solar/releases/tag/v1.0.0`
2. Create the target directory:
   ```bash
   mkdir -p work/checkpoints
   ```
3. Extract archive(s) into `work/checkpoints/`:
   ```bash
   tar -xzf weights_family_clahe.tar.gz -C work/checkpoints/
   tar -xzf weights_family_anchors5.tar.gz -C work/checkpoints/
   tar -xzf weights_family_dice.tar.gz -C work/checkpoints/
   tar -xzf weights_family_2048.tar.gz -C work/checkpoints/
   tar -xzf weights_family_a5dice.tar.gz -C work/checkpoints/
   tar -xzf weights_family_1792.tar.gz -C work/checkpoints/
   tar -xzf weights_family_refiner.tar.gz -C work/checkpoints/
   ```
4. Verify checksums:
   ```bash
   cd work/checkpoints && sha256sum -c checksums.sha256
   ```

---

## 5. Complete 30-Model Manifest & Checksums

The authoritative 30 models comprising the ensemble:

### Family 1: Multi-Scale CLAHE 1536px (`weights_family_clahe.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_1536_clahe_f0.pt` | 0 | 1536x1536 | 3 | BCE | `060fdd76ebdef93cdc99f7cc7d68285f08109a35687b7043fe4d8b885ac0e87f` | 175.4 MB |
| `r50_1536_clahe_f1.pt` | 1 | 1536x1536 | 3 | BCE | `2729cee16f340ab18cf56695398db1ba3d84d02e94bb8982e3295bc28b533133` | 175.4 MB |
| `r50_1536_clahe_f2.pt` | 2 | 1536x1536 | 3 | BCE | `e0bed6b916b4d5db2cc0b059aee3890166b66e2fc3e87320cc1dd190c050226a` | 175.4 MB |
| `r50_1536_clahe_f3.pt` | 3 | 1536x1536 | 3 | BCE | `27827a45d5d3adeb43f30f8e454a4cc3ef78b7b6f4d1e7d90492d9811b39b72d` | 175.4 MB |
| `r50_1536_clahe_f4.pt` | 4 | 1536x1536 | 3 | BCE | `771d050d3267d2d4cc23a514eb18cf8f9ecc7d6287f165e14e8d214ea5d587cc` | 175.4 MB |

### Family 2: 5 Anchor Aspect Ratios 1536px (`weights_family_anchors5.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_1536_anchors5_f0.pt` | 0 | 1536x1536 | 5 | BCE | `97bbf20e3ce92e411cd3b90e5afed51b06f2d4e9796201a4124d1bc412a2564f` | 175.4 MB |
| `r50_1536_anchors5_f1.pt` | 1 | 1536x1536 | 5 | BCE | `3efda476597ac106e96fcdca7382609dd8fa567682bfd1bd54da9c46c77a6fa9` | 175.4 MB |
| `r50_1536_anchors5_f2.pt` | 2 | 1536x1536 | 5 | BCE | `6850953e83a6df5cf1d4c994ddf70cd09fca508c9fb9b465c6fc33e6eb91c643` | 175.4 MB |
| `r50_1536_anchors5_f3.pt` | 3 | 1536x1536 | 5 | BCE | `fd5c26ed390dabdcdf009fc58652f85b05347f6c8c92e742e6ab42a6c64a45fe` | 175.4 MB |
| `r50_1536_anchors5_f4.pt` | 4 | 1536x1536 | 5 | BCE | `1dc8e42631f66a3ae8d3831d8331cc65411794bec52f28fa489e4dbb4baa95aa` | 175.4 MB |

### Family 3: BCE + Soft Dice Loss 1536px (`weights_family_dice.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_1536_dice_f0.pt` | 0 | 1536x1536 | 3 | BCE+Dice | `5f1b3f22d193998b2befc8185f2688e0329663f0e53015e410ac886867d1b610` | 175.4 MB |
| `r50_1536_dice_f1.pt` | 1 | 1536x1536 | 3 | BCE+Dice | `508b6c85a9976b5ed943d81d39d9c1d26c122f465e19a6bb7933bc3e95b12d51` | 175.4 MB |
| `r50_1536_dice_f2.pt` | 2 | 1536x1536 | 3 | BCE+Dice | `91a52f4a259761635dc349ae28a6ce7bdc170366c457f6807a5a40f78f9f1512` | 175.4 MB |
| `r50_1536_dice_f3.pt` | 3 | 1536x1536 | 3 | BCE+Dice | `25fd20dd6c52fedd33ff37f174541602556dff95dea37262bf035acec82829cf` | 175.4 MB |
| `r50_1536_dice_f4.pt` | 4 | 1536x1536 | 3 | BCE+Dice | `c597cfe41308aa5786b6172380afd57367b3c7a86fb468a851a66f71f4dbca7b` | 175.4 MB |

### Family 4: Native 2048px Resolution (`weights_family_2048.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_2048_f0.pt` | 0 | 2048x2048 | 3 | BCE | `fff7989cb52ab751ac40613ef1a4b14199b3b2bc35a377212b01e90c8078d78a` | 175.4 MB |
| `r50_2048_f1.pt` | 1 | 2048x2048 | 3 | BCE | `b41388b85712b7f0cef9aa3a6ac52546335a0d269cab65d02ae57d4017fc3a42` | 175.4 MB |
| `r50_2048_f2.pt` | 2 | 2048x2048 | 3 | BCE | `e9da2ae20fefe392cb190ec150729a896dcff97acfe00fe421488ce61a237005` | 175.4 MB |
| `r50_2048_f3.pt` | 3 | 2048x2048 | 3 | BCE | `01913a10f5d224049a04f6815d633dc0adffff315dd7a0a5f448426e32b47fb8` | 175.4 MB |
| `r50_2048_f4.pt` | 4 | 2048x2048 | 3 | BCE | `1380b400ea16014c928baa4f93f1fa609c82886a46a351779d8e496ade82e1b1` | 175.4 MB |

### Family 5: 5 Anchors + Soft Dice 1536px (`weights_family_a5dice.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_1536_a5dice_f0.pt` | 0 | 1536x1536 | 5 | BCE+Dice | `5b6256933aae7d7aec5996c04d08545575c40160fe2e0f48b77990fbaea76256` | 175.4 MB |
| `r50_1536_a5dice_f1.pt` | 1 | 1536x1536 | 5 | BCE+Dice | `7c992a4ea280e559ed91c617048f2a847b25e1cdc8e45cd73cdb21a85e3d4532` | 175.4 MB |
| `r50_1536_a5dice_f2.pt` | 2 | 1536x1536 | 5 | BCE+Dice | `f5df4b386ddd641c0093c2aae7c02ecc3779b807a9a4a9ac7859cdc20e5449d4` | 175.4 MB |
| `r50_1536_a5dice_f3.pt` | 3 | 1536x1536 | 5 | BCE+Dice | `d9b617e2faff64be2cd483468c825c5cc70e12f07036206844c8873225d7b60d` | 175.4 MB |
| `r50_1536_a5dice_f4.pt` | 4 | 1536x1536 | 5 | BCE+Dice | `5ef70f009889c276fc0635f85da71efdaa24c0780cc838ddcbd9b2adfb6b3dd0` | 175.4 MB |

### Family 6: Multi-Scale CLAHE 1792px (`weights_family_1792.tar.gz`)
| Model File | Fold | Resolution | Anchors | Loss | SHA-256 Checksum | Size |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `r50_1792_clahe_f0.pt` | 0 | 1792x1792 | 3 | BCE | `f8f5161311916f89729a1b351d8df96f6c2c1de90d668a1e121d55b72ecea923` | 175.4 MB |
| `r50_1792_clahe_f1.pt` | 1 | 1792x1792 | 3 | BCE | `552a2cb263cae3810204423a5ba0a5ff08797341a30a63c805e7d3c7dc847836` | 175.4 MB |
| `r50_1792_clahe_f2.pt` | 2 | 1792x1792 | 3 | BCE | `ee915ec97105bf50e7fdabf2897d7bbeedf303a1c6a2f4abc7b66e59b908c1c2` | 175.4 MB |
| `r50_1792_clahe_f3.pt` | 3 | 1792x1792 | 3 | BCE | `ade1836dd80645762e38bd0685b17fb406d80b59dad3874eeb4266f0c2ed7509` | 175.4 MB |
| `r50_1792_clahe_f4.pt` | 4 | 1792x1792 | 3 | BCE | `c0712263ac8c08d773a0f244f1a753c87871ee280010186c880fdf7c65dc462f` | 175.4 MB |

### Family 7: Phase 3B Semantic U-Nets, Refiners & LightGBM Ranker (`weights_family_refiner.tar.gz`)
Stored directly in-repo under `weights/phase3/` (**51 files**, `1,345.24 MB`, all `< 96.1 MB` per file) and also bundled unsharded in the GitHub Release backup archive `weights_family_refiner.tar.gz`:

| Model / Bundle | Fold | Resolution / Role | Format | SHA-256 Checksum (or Reference) | Size |
|---|:---:|---|---|---|---:|
| `semantic_unet_f0_fp16.pt` | 0 | `1024x1024`, 2-ch (`fg, core`) | FP16 state dict | `c7633815a7b72bb5349837411b410b50318730e47624b3e5f1a589f0afd3bd4b` | 65.25 MB |
| `semantic_unet_f1..4_fp16.pt` | 1–4 | `1024x1024`, 2-ch (`fg, core`) | FP16 state dict | Verified in `weights/checksums.sha256` (lines 43–46) | 65.25 MB ea |
| `resnext101_unet_f{0..4}_fp16.pt.part0{0,1}` | 0–4 | `1280x1280`, 3-ch (`fg, core, edge`) | 10 FP16 binary shards | Verified in `weights/checksums.sha256` (lines 32–41) | 96.06 MB ea |
| `phase3/crop_refiner_f{0..4}_fp16.pt` | 0–4 | `160x160`, 6-ch Local ROI Refiner | FP16 state dict | `fe1a75f695450bbbdbbd15f34c505fc316133f5d17276fd895c314ca88810b88` (`f0`) | 3.97 MB ea |
| `phase3/wide_refiner_v2_f{0..4}_fp16.pt` | 0–4 | `224x224`, 11-ch Wide Refiner V2 | FP16 state dict | `f753a39aa588cc90dd8d23514d901f1b014c676614f66c4a1fff0886e7ee9132` (`f0`) | 6.91 MB ea |
| `phase3/lgb_s27_yhyb_s{2026..2030}_f{0..4}.txt` + `lgb_metadata.json` | 0–4 | 25 LightGBM boosters (`S27`) + JSON | Text + JSON | Verified in `weights/checksums.sha256` (lines 6–31) | 3.94 MB total |

---

## 6. Optional FP32 Backup Archive Checksums

| Archive File | Size | SHA-256 Checksum |
|---|:---:|---|
| `weights_family_clahe.tar.gz` | 813.8 MB | `2c5e3d1d69dc79aa1eb455ebc7b36f84e503c8e0d84f7d2d3d42ec9a4458d8ff` |
| `weights_family_anchors5.tar.gz` | 813.9 MB | `15910d2f5d73b817a15ec60f935a281971f46c08684a9200de6108546904c556` |
| `weights_family_dice.tar.gz` | 813.8 MB | `353e25aeae47be19839b5aee31bda1a157d96977e56efedca5c1ecbbeaa50d98` |
| `weights_family_2048.tar.gz` | 813.5 MB | `265af42e7c553a9e8bf198edb4754e2684bb2680d4ff58eaea5196cd773e1ce6` |
| `weights_family_a5dice.tar.gz` | 813.9 MB | `b89dc6737f981abcd55c74668d76b03ca0e1a0a3ed4f0cb68e2ccebe7b2e8d57` |
| `weights_family_1792.tar.gz` | 813.7 MB | `f579f5416f00100e00cdcacf4d457026ddb4d9b9b903b5c14f1bdebda9542274` |
| `weights_family_refiner.tar.gz` | 1,183 MB | `f44bc4646e1206acc534739b784eed06bb17e2505730524be5ef1153994ed455` |

---

## 7. In-Repo FP16 Weights Summary (`weights/`, < 100 MB Per File)

All models required to reproduce the **Selected final submission (`submission_phase3b_local.csv`, Rule C, Public LB `0.41`)** and the Kaggle-submitted Stage-1 baselines (**Rule B**, Public LB `0.40`, and **Rule A**, Public LB `0.39`) live directly inside `weights/` under the strict GitHub **100 MB** (`< 100,000,000` bytes) per-file limit:
- **30 Stage-1 Mask R-CNN checkpoints** (`r50_*_f{0..4}.pt` in `float16`): **`~92.03 MB`** (`87.76 MiB`) each, **`2,760.75 MB`** total.
- **51 Phase 3B files under `weights/phase3/`** (`1,345.24 MB` total): `5` `semantic_unet_f{0..4}_fp16.pt` (`65.25 MB` ea), `10` deterministic shards `resnext101_unet_f{0..4}_fp16.pt.part0{0,1}` (`96.06 MB` ea, reconstructed in memory on load), `5` `crop_refiner_f{0..4}_fp16.pt` (`3.97 MB` ea), `5` `wide_refiner_v2_f{0..4}_fp16.pt` (`6.91 MB` ea), `25` `lgb_s27_yhyb_*.txt` boosters, and `lgb_metadata.json`.

### 7.1 Measured Storage & Parity Metrics
* **Checkpoint Size Reduction**: Each pruned Stage-1 checkpoint shrinks from **175.40 MB** (`float32`) to **92.03 MB** (`87.76 MiB`, `float16`, **50.03%** of FP32 size, `-83.2%` vs 523.9 MB raw training checkpoint), strictly under 100 MB per file.
* **Self-Contained Total**: **81 files** totaling **`4,105,994,780` bytes (`4,105.99 MB` / `3.824 GiB`)**, verified by `weights/checksums.sha256`.
* **FP16 Upcast, Stage-1 `bfloat16` Default & Optional Strict `float32`**: `scripts/common.py:extract_state_dict()` upcasts `float16` weight tensors to `float32` on load. To reproduce the submitted Kaggle competition CSVs (`test_cache` was originally generated under CUDA `bfloat16` autocast), Stage-1 inference defaults to `--stage1-precision bf16` (`get_stage1_autocast_context(precision="bf16")`), while Phase 3B Stage 2/3 run in strict `float32` (`get_autocast_context()` defaults to `nullcontext()`). Strict Stage-1 `--stage1-precision fp32` remains available as an option (`+0.003314` Fold-0 PQ gain vs `bf16`).
* **Measured Fold-0 Single-Model Verification (`r50_1536_clahe_f0.pt`, 142 Fold-0 images, 4-flip TTA + WMF at $\tau_s = 0.75, \tau_m = 0.50$)**:
  - **Strict FP32 inference (`--stage1-precision fp32`, no autocast)**: FP32 weights **PQ = 0.437514** (`SQ=0.666565, RQ=0.656371, TP=1275, FP=708, FN=627`) vs FP16 weights **PQ = 0.438118** (`SQ=0.666626, RQ=0.657216, TP=1275, FP=703, FN=627`), $\Delta PQ = +0.000604$.
  - **`bfloat16` autocast inference (`--stage1-precision bf16`, default)**: FP32 weights **PQ = 0.436265** (`SQ=0.664289, RQ=0.656739, TP=1279, FP=714, FN=623`) vs FP16 weights **PQ = 0.432951** (`SQ=0.664573, RQ=0.651472, TP=1272, FP=731, FN=630`), $\Delta PQ = -0.003314$.

### 7.2 Verifying Checksums & Running Strictly Offline Inference
```bash
# Verify all 81 in-repo SHA-256 checksums in weights/
(cd weights && sha256sum -c checksums.sha256)

# Regenerate A, B, and C jointly on GPU strictly from weights/ (default --stage1-precision bf16 matches Kaggle caching)
python scripts/regenerate_submission.py --rule ALL --mode full \
    --strict-weights --stage1-precision bf16 --weights-dir weights \
    --test-dir <TEST_IMAGES_DIR> --output work/reproduced_csvs --validate

# Optional strict float32 Stage-1 pass (+0.0033 Fold-0 PQ vs bf16):
# python scripts/regenerate_submission.py --rule ALL --mode full --strict-weights --stage1-precision fp32 ...

# Run single-model smoke inference directly from weights/
python scripts/run_pipeline.py --mode single \
    --checkpoint weights/r50_1536_clahe_f0.pt \
    --input_dir sample_images
```
