#!/usr/bin/env python3
"""
scripts/run_2048_experiment.py - Benchmark and execution runner for 2048x2048 Mask R-CNN.

Workflow:
1. Runs a 20-iteration dry run at 2048x2048 (fold 0, target 'all').
   Tests batch 4 first; if OOM occurs, falls back to batch 2 with grad accumulation 2.
2. Measures peak GPU memory and s/iter, and prints estimated training ETA.
3. Executes 30 epochs of training with the verified batch configuration.
4. Executes extended threshold sweep (tune_threshold.py).
"""
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (
    SolarDataset,
    collate_fn,
    find_dataset_paths,
    get_autocast_context,
    get_maskrcnn_model,
)


def dry_run_benchmark(batch_size: int, grad_accum: int, img_size: int = 2048, iters: int = 20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    paths = find_dataset_paths()
    dataset = SolarDataset(
        json_path=paths["train_json"],
        image_dir=paths["train_dir"],
        folds_csv=paths["folds_csv"],
        fold=0,
        is_train=True,
        target_mode="all",
        target_size=img_size,
        seed=2026,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    model = get_maskrcnn_model(pretrained=True, min_size=img_size, max_size=img_size, num_classes=2)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    start_time = time.time()
    it = 0
    loader_iter = iter(loader)

    while it < iters:
        images, targets, _ = next(loader_iter)
        images_gpu = [img.to(device) for img in images]
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with get_autocast_context(device):
            loss_dict = model(images_gpu, targets_gpu)
            loss = sum(loss_dict.values()) / grad_accum

        loss.backward()
        it += 1

        if it % grad_accum == 0 or it == iters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    elapsed = time.time() - start_time
    s_per_iter = elapsed / iters
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    del model, optimizer, loader, dataset, images_gpu, targets_gpu, loss_dict, loss
    torch.cuda.empty_cache()

    return s_per_iter, peak_mem_gb


def main():
    argparse.ArgumentParser(
        description="Benchmark and execute 2048x2048 Mask R-CNN training and threshold tuning."
    ).parse_args()
    print("==================================================================")
    print("Starting 2048x2048 Experiment Runner")
    print("Step 1: 20-Iteration Dry Run Benchmark")
    print("==================================================================")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    print(f"Device: {device_name} ({total_mem_gb:.1f} GB total VRAM)")

    chosen_bs = 4
    chosen_accum = 1
    s_per_iter = None
    peak_mem_gb = None

    print("\n--- Testing Batch Size 4 (grad_accum=1) for 20 iterations ---")
    try:
        s_per_iter, peak_mem_gb = dry_run_benchmark(batch_size=4, grad_accum=1, img_size=2048, iters=20)
        print(f"Batch size 4 succeeded: {peak_mem_gb:.2f} GB peak memory, {s_per_iter:.3f} s/iter")
        chosen_bs = 4
        chosen_accum = 1
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"Batch size 4 OOM: {e}")
            print("Switching to Batch Size 2 with Grad Accumulation 2...")
            torch.cuda.empty_cache()
            s_per_iter, peak_mem_gb = dry_run_benchmark(batch_size=2, grad_accum=2, img_size=2048, iters=20)
            print(f"Batch size 2 with grad accum 2 succeeded: {peak_mem_gb:.2f} GB peak memory, {s_per_iter:.3f} s/iter")
            chosen_bs = 2
            chosen_accum = 2
        else:
            raise

    # Calculate training ETA
    # In fold 0, target 'all' has 904 training samples
    train_samples = 904
    batches_per_epoch = math.ceil(train_samples / chosen_bs)
    total_train_iters = 30 * batches_per_epoch
    train_time_sec = total_train_iters * s_per_iter
    val_time_est_sec = 10 * 120.0  # 10 validation runs at val_every=3, ~120s each
    total_eta_sec = train_time_sec + val_time_est_sec
    eta_hours = total_eta_sec / 3600.0

    print("\n==================================================================")
    print("DRY RUN BENCHMARK SUMMARY (2048x2048 Mask R-CNN)")
    print(f"  Chosen Config:       batch_size={chosen_bs}, grad_accum={chosen_accum}")
    print(f"  Peak GPU Memory:     {peak_mem_gb:.2f} GB / {total_mem_gb:.1f} GB ({peak_mem_gb/total_mem_gb*100:.1f}%)")
    print(f"  Measured Speed:      {s_per_iter:.3f} s/iter")
    print(f"  Batches per Epoch:   {batches_per_epoch}")
    print(f"  Total Train Iters:   {total_train_iters} (over 30 epochs)")
    print(f"  Estimated Train ETA: {eta_hours:.2f} hours ({total_eta_sec/60:.1f} minutes)")
    print("==================================================================\n")

    # Step 2: Full training
    train_cmd = [
        sys.executable,
        "scripts/train_maskrcnn.py",
        "--run_name", "r50_2048_all_e30",
        "--fold", "0",
        "--target", "all",
        "--epochs", "30",
        "--batch_size", str(chosen_bs),
        "--grad_accum", str(chosen_accum),
        "--img_size", "2048",
        "--lr", "0.0002",
        "--optimizer", "adamw",
        "--weight_decay", "0.0001",
        "--val_every", "3",
        "--num_workers", "2",
    ]
    print(f"Starting real training: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    # Step 3: Threshold sweep
    tune_cmd = [
        sys.executable,
        "scripts/tune_threshold.py",
        "--run_name", "r50_2048_all_e30",
        "--checkpoint", "work/runs/r50_2048_all_e30/checkpoints/maskrcnn_fold0_best.pt",
        "--img_size", "2048",
        "--rescore",
        "--size_bins",
    ]
    print(f"Starting threshold sweep: {' '.join(tune_cmd)}")
    subprocess.run(tune_cmd, check=True)
    print("2048x2048 experiment finished successfully!")


if __name__ == "__main__":
    main()
