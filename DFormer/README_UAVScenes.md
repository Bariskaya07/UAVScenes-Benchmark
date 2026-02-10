# DFormerv2 for UAVScenes Dataset

DFormerv2-Base implementation for UAVScenes RGB + HAG (Height Above Ground) multi-modal semantic segmentation.

## Benchmark Target
- **CMNeXt UAVScenes 768x768**: ~76.94% mIoU
- **Goal**: Achieve comparable or better performance with DFormerv2

## Setup

### 1. Install Dependencies
```bash
pip install torch torchvision
pip install timm mmengine mmcv
pip install tensorboardX tabulate tqdm easydict opencv-python
```

### 2. Download Pretrained Weights
Pretrained weights are already in `checkpoints/pretrained/DFormerv2_Base_pretrained.pth`

### 3. Dataset Structure
UAVScenes data should be at `/home/bariskaya/Projelerim/UAV/UAVScenesData` with structure:
```
UAVScenesData/
├── interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
├── interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
└── interval5_HAG/{scene}/{timestamp}.png
```

## Training

### Single GPU Training
```bash
python utils/train.py \
    --config local_configs.UAVScenes.DFormerv2_B \
    --gpus 1 \
    --no-syncbn \
    --amp \
    --mst
```

### Multi-GPU Training (Recommended for Google Cloud)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    utils/train.py \
    --config local_configs.UAVScenes.DFormerv2_B \
    --gpus 4 \
    --amp \
    --mst
```

## Configuration Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | DFormerv2-Base | 53.9M parameters |
| Image Size | 768x768 | Fair comparison with CMNeXt |
| Batch Size | 4 | Adjust based on GPU memory |
| Learning Rate | 6e-5 | AdamW optimizer |
| Epochs | 100 | ~19k training images |
| HAG Normalization | 50m | Same as CMNeXt |
| Augmentation | Scale 0.5-1.75, Flip | DFormerv2 paper settings |

## Key Files

- `utils/dataloader/UAVScenesDataset.py` - UAVScenes dataset class
- `local_configs/UAVScenes/DFormerv2_B.py` - Training configuration
- `local_configs/_base_/datasets/UAVScenes.py` - Dataset configuration

## HAG (Height Above Ground) Encoding

- **Raw format**: 16-bit PNG, encoded as `(HAG_meters * 1000) + 20000`
- **Decoding**: `HAG_meters = (raw - 20000) / 1000.0`
- **Normalization**: Divide by 50m, clip to [0, 1]
- **Invalid pixels**: raw=0 (decoded to -20m) are set to 0

## Dataset Statistics

- **Training**: ~18,400 images (16 scenes)
- **Testing**: ~5,700 images (4 scenes)
- **Classes**: 19 (17 static + 2 dynamic)
- **Resolution**: 2048x2448 (resized/cropped to 768x768)

## Google Cloud VM Recommendations

- **GPU**: A100 40GB (recommended) or V100 32GB
- **Memory**: 64GB RAM
- **Storage**: 100GB SSD (for dataset + checkpoints)
- **Batch Size**:
  - A100 40GB: batch_size=4-8
  - V100 32GB: batch_size=2-4

## Testing Pipeline

```bash
# Test data loading
python test_dataloader.py

# Test model (requires GPU with sufficient memory)
python test_training.py
```

## Expected Results

Training for 100 epochs should yield:
- mIoU comparable to CMNeXt (~76-77%)
- Training time: ~24-48 hours on A100

## Troubleshooting

1. **OOM Error**: Reduce batch_size in config
2. **HAG all zeros**: Check HAG decoding formula (should be `(raw - 20000) / 1000.0`)
3. **Module not found**: Install missing dependencies with pip
