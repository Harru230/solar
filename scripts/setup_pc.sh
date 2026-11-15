#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Setting up virtual environment in .venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=========================================================="
echo "REMINDER: PyTorch must be installed manually on the PC."
echo "Check https://pytorch.org/get-started/locally/ for your CUDA version."
echo "Official command (Linux, Pip, CUDA 12.6):"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
echo "=========================================================="
echo ""

echo "Running GPU check (this will fail if torch is not installed yet, which is expected on first run)..."
python scripts/check_gpus.py || true
