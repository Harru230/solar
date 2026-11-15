"""
scripts/check_gpus.py - Verify CUDA GPU availability before training runs.

Inputs:
  --min-gpus: Minimum number of visible CUDA devices required (default: 2).

Outputs:
  Exits with status 0 if at least --min-gpus CUDA devices are available, else 1.
"""

import argparse
import sys

import torch


def main() -> None:
    """Parse CLI arguments and check CUDA device availability."""
    parser = argparse.ArgumentParser(description="Verify CUDA GPU availability for training.")
    parser.add_argument("--min-gpus", type=int, default=2, help="Minimum required CUDA devices (default: 2)")
    args = parser.parse_args()

    try:
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA devices.")
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        if num_gpus < args.min_gpus:
            print(f"FAIL: Expected at least {args.min_gpus} GPUs.")
            sys.exit(1)
        print("PASS: GPUs check passed.")
    except Exception as e:
        print(f"FAIL: Could not check GPUs (is torch installed correctly?): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
