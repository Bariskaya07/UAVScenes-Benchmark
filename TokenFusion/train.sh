#!/bin/bash
# Training script for TokenFusion on UAVScenes dataset
# RGB + HAG Multi-Modal Semantic Segmentation

# Configuration
CONFIG="configs/uavscenes_rgb_hag.yaml"
GPUS=1

# Training settings
export CUDA_VISIBLE_DEVICES=0

# Download pretrained weights if not exists
if [ ! -f "pretrained/mit_b2.pth" ]; then
    echo "Downloading MiT-B2 pretrained weights..."
    mkdir -p pretrained
    wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth -O pretrained/mit_b2.pth
fi

# Create output directories
mkdir -p logs
mkdir -p checkpoints

# Run training
echo "Starting TokenFusion training on UAVScenes..."
echo "Config: ${CONFIG}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python main.py \
    --config ${CONFIG} \
    --gpus ${GPUS} \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo "Training completed!"
