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
cd "$(dirname "$0")/models/encoders/selective_scan"
python setup.py build_ext --inplace
cd -

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Run: python train.py --dataset_name uavscenes --devices 0,1"
echo "========================================="
