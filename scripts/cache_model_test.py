#!/usr/bin/env python3
"""
scripts/cache_model_test.py - Generates 4-flip test candidate detections into work/test_cache/<name>/.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import extract_state_dict, find_dataset_paths, get_maskrcnn_model
from scripts.predict import load_image
from scripts.oof_eval import extract_4view_candidates_from_tensor


def drop_os_page_cache(filepath):
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Cache test detections for a single model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--name", type=str, required=True, help="Subdirectory name in test_cache")
    parser.add_argument("--img_size", type=int, default=1536, help="Image resolution")
    parser.add_argument("--cache_dir", type=str, default="work/test_cache", help="Test cache root")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--score_filter", type=float, default=0.25, help="Proposal threshold")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    paths = find_dataset_paths()
    test_dir = paths["test_dir"]
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))

    out_dir = os.path.join(args.cache_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)

    missing = [p for p in test_files if not os.path.isfile(os.path.join(out_dir, f"{Path(p).stem}.pt"))]
    if not missing:
        print(f"[Cache Complete] {args.name}: all {len(test_files)} test stems already cached in {out_dir}")
        return

    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    a5 = cfg.get("anchors5", False) or ("anchors5" in args.name) or ("_a5" in args.checkpoint) or ("_a5dice" in args.checkpoint)
    sz = args.img_size or cfg.get("img_size", 1536)

    model = get_maskrcnn_model(pretrained=False, min_size=sz, max_size=sz, num_classes=2, anchors5=a5)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(device).eval()
    del ckpt
    drop_os_page_cache(args.checkpoint)

    t0 = time.time()
    print(f"Caching 4-flip test candidates for {len(missing)} stems into {out_dir}...")
    with torch.no_grad():
        for idx, img_path in enumerate(missing, start=1):
            stem = Path(img_path).stem
            cache_pt = os.path.join(out_dir, f"{stem}.pt")
            base_t, _ = load_image(img_path, img_size=sz)
            cands = extract_4view_candidates_from_tensor(model, base_t, sz, device, score_filter=args.score_filter)
            tmp_pt = f"{cache_pt}.tmp"
            torch.save({"stem": stem, "model": args.name, "candidates": cands}, tmp_pt)
            os.replace(tmp_pt, cache_pt)
            drop_os_page_cache(cache_pt)
            del cands, base_t
            if idx % 30 == 0 or idx == len(missing):
                print(f"  [{idx:03d}/{len(missing):03d}] {stem} ({time.time() - t0:.1f}s)", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[SUCCESS] Finished caching {args.name} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
