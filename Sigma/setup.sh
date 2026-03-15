#!/bin/bash
# Sigma UAVScenes Setup Script
# Usage: bash setup.sh
# Requirements: CUDA 11.8, Python 3.10, A100 GPU

set -e  # Exit on error

echo "========================================="
echo "  Sigma UAVScenes Environment Setup"
echo "========================================="

# Step 1: PyTorch cu118
echo "[1/5] Installing PyTorch (CUDA 11.8)..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall -q

# Step 2: NumPy < 2 and OpenCV (must come after torch)
echo "[2/5] Installing NumPy and OpenCV..."
pip install "numpy<2" opencv-python --force-reinstall -q

# Step 3: causal-conv1d (must use --no-build-isolation)
echo "[3/5] Building causal-conv1d (this takes ~10 min)..."
pip install causal-conv1d --no-build-isolation -q

# Step 4: mamba-ssm (must use --no-build-isolation)
echo "[4/5] Installing mamba-ssm..."
pip install mamba-ssm --no-build-isolation -q

# Step 5: Other dependencies
echo "[5/5] Installing remaining dependencies..."
pip install timm tensorboardX easydict scipy tqdm Pillow \
    PyYAML fvcore tabulate termcolor packaging einops -q

# Step 6: Build selective_scan CUDA kernel
echo "[6/6] Building selective_scan CUDA kernel (this takes ~5 min)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/models/encoders/selective_scan"
python setup.py build_ext --inplace
cp selective_scan_cuda_core*.so "$SCRIPT_DIR/"
cd "$SCRIPT_DIR"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Run: torchrun --nproc_per_node=4 --master_port=16005 train.py --dataset_name uavscenes --devices 0,1,2,3"
echo "========================================="
