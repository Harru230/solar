#!/usr/bin/env python3
"""
scripts/train_maskrcnn.py - Training pipeline for torchvision Mask R-CNN (ResNet50-FPN v2)
on the Solar Filament Segmentation Challenge 2026.

Features:
- ResNet50-FPN v2 with COCO-pretrained weights (or ImageNet fallback)
- Single foreground class (filament)
- Input resized to --img_size x --img_size (default 1024), predicted masks always upsampled
  back to 2048x2048 before RLE encoding
- 5-fold cross-validation (--fold N, default 0)
- Cosine LR schedule with linear warmup (AdamW or SGD)
- Mixed Precision (AMP bf16) on CUDA
- Topology-preserving augmentations (flips, 90-degree rotations, mild brightness/contrast; NO elastic)
- Target options: --target {all, union}
- Run isolation: all outputs (checkpoints, log, best_thresholds.json) go under
  work/runs/<run_name>/ so runs never overwrite each other (--run_name, default "default")
- Checkpointing: saves checkpoint each epoch and best-by-PQ on held-out fold
- Evaluation on held-out fold via pq_eval.score (NOT Dice); frequency controlled by --val_every N
- Resumable from checkpoints (--resume PATH)
- Tmux-friendly logging to stdout and log file
- Deterministic random seed (--seed 2026)
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pq_eval
from scripts.common import (
    EMPTY_RLE,
    SolarDataset,
    apply_anchors5,
    apply_mask_nms,
    collate_fn,
    find_dataset_paths,
    get_maskrcnn_model,
    get_run_dir,
    make_masks_disjoint,
    patch_maskrcnn_loss_bce_dice,
    rle_encode_mask,
    set_seed,
    setup_logger,
)


def evaluate_pq_on_val(
    model: torch.nn.Module,
    val_dataset: SolarDataset,
    gt_all_csv: str,
    fold: int,
    device: torch.device,
    score_thresh: float = 0.35,
    mask_thresh: float = 0.5,
    mask_nms_thresh: float = 0.5,
    img_size: int = 1024,
    logger=None,
) -> dict:
    """
    Evaluates the model on the held-out validation fold using pq_eval.score.
    Generates full-resolution (2048x2048) masks upsampled from img_size x img_size,
    applies mask NMS, and matches against GT annotations.
    """
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    pred_rows = []
    total_imgs = len(val_dataset)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, _, obs_ids) in enumerate(val_loader):
            images_gpu = [img.to(device) for img in images]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images_gpu)

            for i, output in enumerate(outputs):
                obs = obs_ids[i]
                scores = output["scores"].detach().cpu().numpy()
                masks = output["masks"]  # [K, 1, img_size, img_size]

                # Filter by score threshold
                keep = scores >= score_thresh
                if not np.any(keep):
                    continue

                scores_kept = scores[keep]
                masks_kept = masks[keep]  # [M, 1, img_size, img_size]

                # Bilinear upsample soft mask probabilities to 2048x2048 in chunks to avoid VRAM fragmentation
                if masks_kept.shape[0] > 0:
                    bin_masks_list = []
                    chunk_size = 32
                    for c_start in range(0, masks_kept.shape[0], chunk_size):
                        c_chunk = masks_kept[c_start : c_start + chunk_size]
                        c_up = F.interpolate(c_chunk, size=(2048, 2048), mode="bilinear", align_corners=False)
                        bin_masks_list.append((c_up.squeeze(1) >= mask_thresh).byte().cpu().numpy())
                        del c_up
                    bin_masks = np.concatenate(bin_masks_list, axis=0)
                else:
                    bin_masks = np.zeros((0, 2048, 2048), dtype=np.uint8)

                valid_masks = []
                valid_scores = []
                valid_rles = []
                for m_idx in range(bin_masks.shape[0]):
                    m = bin_masks[m_idx]
                    if m.sum() > 0:
                        valid_masks.append(m)
                        valid_scores.append(float(scores_kept[m_idx]))
                        valid_rles.append(rle_encode_mask(m))

                if not valid_masks:
                    continue

                # Apply Mask NMS
                kept_nms = apply_mask_nms(valid_rles, valid_scores, iou_thresh=mask_nms_thresh)
                if not kept_nms:
                    continue

                nms_masks = [valid_masks[k_idx] for k_idx in kept_nms]
                nms_scores = [valid_scores[k_idx] for k_idx in kept_nms]

                clean_masks, clean_scores = make_masks_disjoint(nms_masks, nms_scores)
                if not clean_masks:
                    continue

                assert np.sum(np.stack(clean_masks, axis=0), axis=0).max() <= 1

                for k_order, m in enumerate(clean_masks, start=1):
                    pred_rows.append({
                        "filament_id": f"{obs}_{k_order}",
                        "segmentation_rle": rle_encode_mask(m),
                    })

    eval_time = time.time() - start_time
    pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])

    # Load GT for validation fold
    gt_df = pd.read_csv(gt_all_csv)
    if "folds_csv" in val_dataset.__dict__ or val_dataset.samples:
        val_obs = set(s["obs"] for s in val_dataset.samples)
        gt_obs = gt_df["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1]
        val_gt_df = gt_df[gt_obs.isin(val_obs)].copy()
    else:
        val_gt_df = gt_df

    if pred_df.empty:
        if logger:
            logger.warning("No predictions above threshold on validation set.")
        res = {
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "tp": 0, "fp": 0, "fn": len(val_gt_df),
            "gt_multi_hit": 0, "pred_multi_hit": 0,
            "annotator_images": val_gt_df["filament_id"].str.split("_").str[0].nunique(),
            "images_without_prediction": len(val_obs) if 'val_obs' in locals() else 0,
        }
    else:
        res = pq_eval.score(val_gt_df, pred_df)

    res["eval_time_sec"] = eval_time
    del val_loader, pred_rows, pred_df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger,
    grad_accum: int = 1,
    log_interval: int = 10,
) -> float:
    """Trains Mask R-CNN for one epoch with AMP bf16."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    start_time = time.time()

    optimizer.zero_grad()
    step_loss_dict = defaultdict(float)

    for batch_idx, (images, targets, _) in enumerate(train_loader):
        batch_len = len(images)
        micro_bs = 2
        step_loss = 0.0

        for c_start in range(0, batch_len, micro_bs):
            sub_images = [img.to(device) for img in images[c_start : c_start + micro_bs]]
            sub_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[c_start : c_start + micro_bs]]
            scale = (len(sub_images) / batch_len) / grad_accum

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sub_loss_dict = model(sub_images, sub_targets)
                sub_loss = sum(l for l in sub_loss_dict.values()) * scale

            if not math.isfinite(sub_loss.item()):
                logger.error(f"Loss is {sub_loss.item()}, stopping training!")
                sys.exit(1)

            sub_loss.backward()
            step_loss += sub_loss.item() * grad_accum
            for k, v in sub_loss_dict.items():
                step_loss_dict[k] += v.item() * (len(sub_images) / batch_len)

            del sub_images, sub_targets, sub_loss_dict, sub_loss

        # Step optimizer every grad_accum batches or at the end of epoch
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += step_loss

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - start_time
            avg_loss = total_loss / (batch_idx + 1)
            loss_details = ", ".join(f"{k}: {v:.3f}" for k, v in step_loss_dict.items())
            logger.info(
                f"Epoch [{epoch:02d}] Step [{batch_idx + 1:03d}/{num_batches:03d}] "
                f"Loss: {avg_loss:.4f} ({loss_details}) "
                f"Time: {elapsed:.1f}s"
            )

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on Solar Filament dataset")
    parser.add_argument("--run_name", type=str, default="default",
                        help="Name for this run; all outputs go under work/runs/<run_name>/ (default: 'default')")
    parser.add_argument("--fold", type=int, default=0, help="Held-out validation fold (0-4)")
    parser.add_argument("--full_data", action="store_true", help="Train on all 707 observations (all 5 folds) without held-out validation")
    parser.add_argument("--target", type=str, choices=["all", "union", "consensus"], default="all",
                        help="Target mode: 'all' (all annotator masks as separate samples), 'union' (merged), or 'consensus'")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Training/inference resolution (images are resized to img_size x img_size; "
                             "predicted masks are always upsampled back to 2048x2048 before RLE). Default: 1024")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw", help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Evaluate PQ on val fold every N epochs (default: 1). "
                             "Set to e.g. 5 to save time on long runs.")
    parser.add_argument("--val_score_thresh", type=float, default=0.35, help="Detection score threshold for val PQ")
    parser.add_argument("--anchors5", action="store_true",
                        help="Reconfigure RPN anchor generator for 5 aspect ratios (0.2, 0.5, 1.0, 2.0, 5.0) per level")
    parser.add_argument("--model_version", type=str, choices=["v1", "v2"], default="v2",
                        help="Torchvision Mask R-CNN version: 'v1' or 'v2' (default: 'v2')")
    parser.add_argument("--mask_loss", type=str, choices=["bce", "bce_dice"], default="bce",
                        help="Mask loss function: 'bce' (default torchvision BCE) or 'bce_dice' (0.5*BCE + 0.5*soft Dice)")
    # Legacy --save_dir and --log_file are kept for backward compatibility but overridden by --run_name
    parser.add_argument("--save_dir", type=str, default=None,
                        help="[Deprecated] Override checkpoint directory (default: work/runs/<run_name>/checkpoints)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="[Deprecated] Override log file path (default: work/runs/<run_name>/train_fold<N>.log)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test (2 epochs, 50 train images)")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit number of train images")
    parser.add_argument("--limit_val", type=int, default=None, help="Limit number of val images")
    args = parser.parse_args()

    # Apply smoke test overrides if requested
    if args.smoke_test:
        args.epochs = 2
        args.limit_train = 50
        args.limit_val = 15
        args.val_every = 1

    # ── Resolve run directory ──────────────────────────────────────────────────
    paths = find_dataset_paths()
    repo_root = paths["repo_root"]
    run_dir = get_run_dir(repo_root, args.run_name)
    ckpt_dir = args.save_dir if args.save_dir else os.path.join(run_dir, "checkpoints")
    log_file = args.log_file if args.log_file else os.path.join(run_dir, f"train_fold{args.fold}.log")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logger("train_maskrcnn", log_file)
    logger.info("==================================================")
    logger.info(f"Starting Mask R-CNN Training - Fold {args.fold}")
    logger.info(f"Run name:   {args.run_name}")
    logger.info(f"Run dir:    {run_dir}")
    logger.info(f"Img size:   {args.img_size}x{args.img_size}")
    logger.info(f"Val every:  {args.val_every} epoch(s)")
    logger.info(f"Config: {vars(args)}")
    logger.info("==================================================")

    # Deterministic seeding
    set_seed(args.seed)

    # Locate paths
    if not paths["train_dir"] or not paths["train_json"] or not paths["folds_csv"] or not paths["gt_all_csv"]:
        logger.error(f"Missing required data files! Paths found: {paths}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets and Loaders
    dataset_fold = None if args.full_data else args.fold
    logger.info(f"Creating training dataset (fold={'ALL' if args.full_data else f'!= {args.fold}'}, target mode: {args.target})...")
    train_dataset = SolarDataset(
        json_path=paths["train_json"],
        image_dir=paths["train_dir"],
        folds_csv=paths["folds_csv"],
        fold=dataset_fold,
        is_train=True,
        target_mode=args.target,
        target_size=args.img_size,
        limit=args.limit_train,
        seed=args.seed,
    )

    if args.full_data:
        val_dataset = None
        logger.info(f"Train samples (FULL DATA): {len(train_dataset)}, Val samples: 0 (disabled)")
    else:
        logger.info(f"Creating validation dataset (fold == {args.fold}, target mode: union/eval)...")
        val_dataset = SolarDataset(
            json_path=paths["train_json"],
            image_dir=paths["train_dir"],
            folds_csv=paths["folds_csv"],
            fold=args.fold,
            is_train=False,
            target_mode="union",  # Validation evaluates per physical observation
            target_size=args.img_size,
            limit=args.limit_val,
            seed=args.seed,
        )
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    # Patch maskrcnn_loss if requested
    if args.mask_loss == "bce_dice":
        patch_maskrcnn_loss_bce_dice(logger=logger)

    # Instantiate Model (anchors5 is applied before .to(device) and before optimizer)
    model = get_maskrcnn_model(
        pretrained=True,
        min_size=args.img_size,
        max_size=args.img_size,
        num_classes=2,
        anchors5=args.anchors5,
        model_version=args.model_version,
        logger=logger,
    )
    model.to(device)

    # Log GPU memory after model load
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU memory after model load: {mem_alloc:.2f} GB allocated, "
                    f"{mem_reserved:.2f} GB reserved, {total_mem:.1f} GB total")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Learning rate schedule: Linear warmup for 1 epoch, then Cosine decay
    warmup_epochs = 1
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_pq = -1.0
    if args.resume:
        resume_file = args.resume
        if args.resume.lower() == "auto":
            auto_ckpt = os.path.join(ckpt_dir, f"checkpoint_fold{args.fold}_last.pt")
            resume_file = auto_ckpt if os.path.isfile(auto_ckpt) else None
            if not resume_file:
                logger.info("Auto-resume requested but no previous checkpoint found. Starting from scratch.")

        if resume_file:
            if os.path.isfile(resume_file):
                logger.info(f"Loading checkpoint from '{resume_file}'...")
                checkpoint = torch.load(resume_file, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_pq = checkpoint.get("best_pq", -1.0)
                logger.info(f"Resumed from epoch {checkpoint['epoch']} (Best PQ so far: {best_pq:.4f})")
            else:
                logger.error(f"Checkpoint not found at '{resume_file}'")
                sys.exit(1)

    epoch_runtimes = []

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} (LR: {current_lr:.6f}) ---")

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            logger=logger,
            grad_accum=args.grad_accum,
        )
        lr_scheduler.step()
        epoch_time = time.time() - t_epoch_start
        epoch_runtimes.append(epoch_time)

        # Log GPU peak memory per epoch
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            torch.cuda.reset_peak_memory_stats(device)
            logger.info(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}, "
                        f"Runtime: {epoch_time:.2f}s, Peak GPU: {peak_mem:.2f} GB")
        else:
            logger.info(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}, Runtime: {epoch_time:.2f}s")

        # Save regular checkpoint atomically (temp file + os.replace)
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_pq": best_pq,
            "config": vars(args),
        }
        last_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_fold{args.fold}_last.pt")
        last_ckpt_tmp = f"{last_ckpt_path}.tmp"
        torch.save(checkpoint_data, last_ckpt_tmp)
        os.replace(last_ckpt_tmp, last_ckpt_path)
        try:
            fd = os.open(last_ckpt_path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass
        del checkpoint_data
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate on validation fold every val_every epochs (and always on last epoch)
        if not args.full_data and (epoch % args.val_every == 0 or epoch == args.epochs):
            logger.info(f"Evaluating Panoptic Quality on validation fold {args.fold}...")
            val_res = evaluate_pq_on_val(
                model=model,
                val_dataset=val_dataset,
                gt_all_csv=paths["gt_all_csv"],
                fold=args.fold,
                device=device,
                score_thresh=args.val_score_thresh,
                img_size=args.img_size,
                logger=logger,
            )
            pq = val_res["pq"]
            sq = val_res["sq"]
            rq = val_res["rq"]
            tp = val_res["tp"]
            fp = val_res["fp"]
            fn = val_res["fn"]
            gm = val_res["gt_multi_hit"]
            pm = val_res["pred_multi_hit"]
            logger.info(
                f"[Validation Fold {args.fold}] PQ: {pq:.4f} | SQ: {sq:.4f} | RQ: {rq:.4f} | "
                f"TP: {tp} | FP: {fp} | FN: {fn} | gt_multi: {gm} | pred_multi: {pm} | "
                f"Eval Time: {val_res['eval_time_sec']:.1f}s"
            )

            # Save best checkpoint atomically
            if pq > best_pq:
                best_pq = pq
                best_ckpt_path = os.path.join(ckpt_dir, f"maskrcnn_fold{args.fold}_best.pt")
                best_ckpt_tmp = f"{best_ckpt_path}.tmp"
                best_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_pq": best_pq,
                    "val_metrics": val_res,
                    "config": vars(args),
                }
                torch.save(best_data, best_ckpt_tmp)
                os.replace(best_ckpt_tmp, best_ckpt_path)
                # Also update last_ckpt_path atomically with new best_pq
                torch.save(best_data, last_ckpt_tmp)
                os.replace(last_ckpt_tmp, last_ckpt_path)
                del best_data
                logger.info(f"*** NEW BEST PQ: {best_pq:.4f}! Saved to {best_ckpt_path} ***")

        # Check for clean stop signal at epoch boundary
        stop_flag_path = os.path.join(REPO_ROOT, "work", "STOP_QUEUE")
        if os.path.exists(stop_flag_path) and epoch < args.epochs:
            logger.info(
                f"Clean stop requested via {stop_flag_path} after Epoch {epoch}/{args.epochs}. "
                f"Atomic checkpoint saved to {last_ckpt_path}. Exiting cleanly."
            )
            sys.exit(130)

    avg_epoch_runtime = sum(epoch_runtimes) / max(1, len(epoch_runtimes))
    logger.info("==================================================")
    logger.info("Training Finished!")
    logger.info(f"Average runtime per epoch: {avg_epoch_runtime:.2f}s ({avg_epoch_runtime/60:.2f} min)")
    logger.info(f"Best Validation PQ: {best_pq:.4f}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
