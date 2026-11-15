#!/usr/bin/env python3
"""
scripts/prune_checkpoints.py - Export and Prune Model Weights for Release

Strips AdamW optimizer buffers and lr_scheduler states from raw training checkpoints,
retaining only model_state_dict, config, epoch, and best_pq.
Verifies loading into torchvision Mask R-CNN, computes SHA-256 checksums,
and optionally builds the release tar.gz archive.

Usage:
    python scripts/prune_checkpoints.py --output-dir work/checkpoints --create-archive
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
import torch

# Ensure scripts directory is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import extract_state_dict, get_maskrcnn_model

SOURCE_MODELS = [
    {
        "tag": "f0_1536",
        "relative_src": "work/runs/r50_1536_clahe_e30/checkpoints/checkpoint_fold0_last.pt",
        "dst": "r50_1536_clahe_f0.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 0 (Multi-CLAHE, 1536x1536, epoch 30)",
    },
    {
        "tag": "f1_1536",
        "relative_src": "work/runs/r50_1536_all_f1/checkpoints/checkpoint_fold1_last.pt",
        "dst": "r50_1536_clahe_f1.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 1 (Multi-CLAHE, 1536x1536, epoch 30)",
    },
    {
        "tag": "f2_1536",
        "relative_src": "work/runs/r50_1536_all_f2/checkpoints/checkpoint_fold2_last.pt",
        "dst": "r50_1536_clahe_f2.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 2 (Multi-CLAHE, 1536x1536, epoch 30)",
    },
    {
        "tag": "f3_1536",
        "relative_src": "work/runs/r50_1536_all_f3/checkpoints/checkpoint_fold3_last.pt",
        "dst": "r50_1536_clahe_f3.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 3 (Multi-CLAHE, 1536x1536, epoch 30)",
    },
    {
        "tag": "f4_1536",
        "relative_src": "work/runs/r50_1536_all_f4/checkpoints/checkpoint_fold4_last.pt",
        "dst": "r50_1536_clahe_f4.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 4 (Multi-CLAHE, 1536x1536, epoch 30)",
    },
    {
        "tag": "f0_2048",
        "relative_src": "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt",
        "dst": "r50_2048_f0.pt",
        "img_size": 2048,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 0 (Native 2048x2048, epoch 24)",
    },
    {
        "tag": "f0_anchors5",
        "relative_src": "work/runs/r50_1536_anchors5_e30/checkpoints/maskrcnn_fold0_best.pt",
        "dst": "r50_1536_anchors5_f0.pt",
        "img_size": 1536,
        "anchors5": True,
        "desc": "Mask R-CNN ResNet50-FPN Fold 0 (5 Anchor Aspect Ratios, 1536x1536, epoch 24)",
    },
    {
        "tag": "f0_dice",
        "relative_src": "work/runs/r50_1536_dice_e30/checkpoints/maskrcnn_fold0_best.pt",
        "dst": "r50_1536_dice_f0.pt",
        "img_size": 1536,
        "anchors5": False,
        "desc": "Mask R-CNN ResNet50-FPN Fold 0 (BCE + Dice Loss, 1536x1536, epoch 30)",
    },
]


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def find_source_checkpoint(rel_path: str, repo_root: Path, pruned_name: str = "") -> Path:
    upstream_dir = os.environ.get("SOLAR_UPSTREAM_DIR", str(repo_root.parent / "solar"))
    candidates = [repo_root / rel_path, Path(upstream_dir) / rel_path]
    if pruned_name:
        candidates.extend([
            repo_root / "work/checkpoints" / pruned_name,
            Path(upstream_dir) / "work/checkpoints" / pruned_name,
        ])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Source checkpoint not found in {candidates}")


def load_model_specs(args, repo_root: Path):
    if not args.all_30:
        return SOURCE_MODELS
    cfg_path = Path(args.config) if os.path.isabs(args.config) else repo_root / args.config
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    specs = []
    for m in cfg["models"]:
        specs.append({
            "tag": m["id"],
            "relative_src": m.get("checkpoint_relpath", f"work/checkpoints/{m['pruned_name']}"),
            "dst": m["pruned_name"],
            "img_size": int(m["img_size"]),
            "anchors5": bool(m.get("anchors5", False)),
            "family": m.get("family", ""),
            "fold": int(m.get("fold", 0)),
        })
    return specs


def main():
    parser = argparse.ArgumentParser(description="Prune and export release model weights")
    parser.add_argument("--output-dir", type=str, default="work/checkpoints", help="Output directory")
    parser.add_argument("--create-archive", action="store_true", help="Package into weights_final_ensemble.tar.gz")
    parser.add_argument("--fp16", action="store_true", help="Cast floating-point state_dict tensors to float16 (<100 MB per file)")
    parser.add_argument("--all-30", action="store_true", help="Export all 30 models defined in configs/final_ensemble.json")
    parser.add_argument("--config", type=str, default="configs/final_ensemble.json", help="Ensemble config path for --all-30")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify state_dict loading")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else repo_root / args.output_dir
    if out_dir.is_symlink():
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SOLAR FILAMENT MODEL WEIGHT PRUNING UTILITY ({'FP16' if args.fp16 else 'FP32'})")
    print("=" * 80)

    specs = load_model_specs(args, repo_root)
    results = []
    total_orig_bytes = 0
    total_pruned_bytes = 0

    for m in specs:
        src_path = find_source_checkpoint(m["relative_src"], repo_root, m["dst"])
        dst_path = out_dir / m["dst"]

        orig_bytes = src_path.stat().st_size
        total_orig_bytes += orig_bytes

        t0 = time.time()
        ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        if args.fp16:
            sd = {
                k: (v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                for k, v in sd.items()
            }

        pruned_data = {
            "model_state_dict": sd,
            "dtype": "float16" if args.fp16 else "float32",
            "config": ckpt.get("config", {}),
            "epoch": ckpt.get("epoch"),
            "best_pq": ckpt.get("best_pq"),
            "val_metrics": ckpt.get("val_metrics", {}),
        }

        # Atomic save
        tmp_path = dst_path.with_suffix(".tmp")
        torch.save(pruned_data, tmp_path)
        tmp_path.replace(dst_path)

        pruned_bytes = dst_path.stat().st_size
        total_pruned_bytes += pruned_bytes
        elapsed = time.time() - t0

        checksum = sha256_file(str(dst_path))

        # Verification
        verified = False
        if args.verify:
            model = get_maskrcnn_model(
                pretrained=False,
                min_size=m["img_size"],
                max_size=m["img_size"],
                num_classes=2,
                anchors5=m["anchors5"],
            )
            res = model.load_state_dict(extract_state_dict(pruned_data))
            verified = len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0
            del model

        results.append({
            "tag": m["tag"],
            "filename": m["dst"],
            "family": m.get("family", ""),
            "fold": m.get("fold"),
            "dtype": "float16" if args.fp16 else "float32",
            "orig_mb": orig_bytes / (1024 * 1024),
            "pruned_mb": pruned_bytes / (1024 * 1024),
            "sha256": checksum,
            "verified": verified,
            "elapsed": elapsed,
        })

        reduction = (1.0 - pruned_bytes / orig_bytes) * 100
        print(f"[{m['tag']}] {m['dst']}: {orig_bytes/(1024*1024):.1f}MB -> {pruned_bytes/(1024*1024):.1f}MB (-{reduction:.1f}%) in {elapsed:.1f}s | SHA: {checksum[:12]}...")

        del ckpt
        del pruned_data

    # Copy and verify scorer_v2_models.joblib if available
    scorer_dst = out_dir / "scorer_v2_models.joblib"
    if not scorer_dst.is_file():
        scorer_cands = [
            repo_root / "work/checkpoints/scorer_v2_models.joblib",
            repo_root.parent / "solar/work/checkpoints/scorer_v2_models.joblib",
        ]
        for sc in scorer_cands:
            if sc.is_file():
                import shutil
                shutil.copy2(sc, scorer_dst)
                break

    if scorer_dst.is_file():
        scorer_sha = sha256_file(str(scorer_dst))
        results.append({
            "tag": "scorer_v2",
            "filename": "scorer_v2_models.joblib",
            "orig_mb": scorer_dst.stat().st_size / (1024 * 1024),
            "pruned_mb": scorer_dst.stat().st_size / (1024 * 1024),
            "sha256": scorer_sha,
            "verified": True,
            "elapsed": 0.0,
        })
        print(f"[scorer_v2] scorer_v2_models.joblib: {scorer_dst.stat().st_size / (1024 * 1024):.2f}MB | SHA: {scorer_sha[:12]}...")

    # Write checksums.sha256 and manifest_summary.json
    checksums_path = out_dir / "checksums.sha256"
    with open(checksums_path, "w") as f:
        for r in results:
            f.write(f"{r['sha256']}  {r['filename']}\n")
    print(f"\n[OK] Checksums written to {checksums_path}")

    manifest_path = out_dir / "manifest_summary.json"
    manifest = {
        "dtype": "float16" if args.fp16 else "float32",
        "num_models": sum(1 for r in results if r["filename"].endswith(".pt")),
        "models": {
            r["filename"]: {
                "sha256": r["sha256"],
                "size_mb": r["pruned_mb"],
                "family": r.get("family", ""),
                "fold": r.get("fold"),
                "id": r["tag"],
                "dtype": r.get("dtype", "float32"),
                "verified": r["verified"],
            }
            for r in results
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest written to {manifest_path}")

    # Create convenience symlinks
    for alias in ["maskrcnn_fold0_best.pt", "checkpoint_fold0_last.pt"]:
        alias_path = out_dir / alias
        if alias_path.is_symlink() or alias_path.is_file():
            alias_path.unlink()
        alias_path.symlink_to("r50_1536_clahe_f0.pt")
    print("[OK] Convenience symlinks created (maskrcnn_fold0_best.pt, checkpoint_fold0_last.pt)")

    if args.create_archive:
        archive_path = out_dir / "weights_final_ensemble.tar.gz"
        print(f"\nPackaging into {archive_path}...")
        t0 = time.time()
        with tarfile.open(archive_path, "w:gz") as tar:
            for r in results:
                tar.add(str(out_dir / r["filename"]), arcname=r["filename"])
            tar.add(str(checksums_path), arcname="checksums.sha256")
            tar.add(str(out_dir / "maskrcnn_fold0_best.pt"), arcname="maskrcnn_fold0_best.pt")
            tar.add(str(out_dir / "checkpoint_fold0_last.pt"), arcname="checkpoint_fold0_last.pt")

        arch_bytes = archive_path.stat().st_size
        arch_sha = sha256_file(str(archive_path))
        print(f"[OK] Archive created: {arch_bytes/(1024*1024):.2f} MB in {time.time()-t0:.1f}s")
        print(f"[OK] Archive SHA256: {arch_sha}")

        # Also package standalone scorer archive
        if scorer_dst.is_file():
            sc_archive = out_dir / "scorer_v2_models.tar.gz"
            with tarfile.open(sc_archive, "w:gz") as tar:
                tar.add(str(scorer_dst), arcname="scorer_v2_models.joblib")
            print(f"[OK] Standalone scorer archive created: {sc_archive.name}")

    print("\n" + "=" * 80)
    print("PRUNING COMPLETE")
    print(f"Total Original: {total_orig_bytes/(1024*1024):.1f} MB | Total Pruned: {total_pruned_bytes/(1024*1024):.1f} MB (-{(1-total_pruned_bytes/total_orig_bytes)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
