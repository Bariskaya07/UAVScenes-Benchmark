# UAVScenes Multi-Modal Semantic Segmentation Benchmark

This repository contains implementations of 6 state-of-the-art multi-modal semantic segmentation models adapted for the UAVScenes dataset.

## Models

| Model | Paper | Venue | Backbone |
|-------|-------|-------|----------|
| CMNeXt | Delivering Arbitrary-Modal Semantic Segmentation | CVPR 2023 | MiT-B2 + PPX |
| DFormer | Rethinking RGBD Representation Learning | ICLR 2024 | DFormerv2 |
| TokenFusion | Multimodal Token Fusion for Vision Transformers | CVPR 2022 | MiT-B2 (shared) |
| GeminiFusion | GeminiFusion: Efficient Pixel-wise Multimodal Fusion | ICML 2024 | MiT-B2 (shared) |
| Sigma | Siamese Mamba Network for Multi-Modal Segmentation | WACV 2025 | Sigma (dual) |
| Mul-VMamba | Multi-Modal Visual Mamba | arXiv 2024 | VMamba-T |

## Dataset

- **UAVScenes**: Aerial semantic segmentation with RGB + HAG (Height Above Ground)
- **19 classes** (remapped from original 26)
- **Resolution**: 2448 x 2048 (native), 768 x 768 (training)

## Training Configuration (Fair Comparison)

All models use standardized settings for fair comparison:

```yaml
TRAIN:
  BATCH_SIZE: 8
  EPOCHS: 60
  IMAGE_SIZE: [768, 768]

OPTIMIZER:
  NAME: AdamW
  LR: 6e-5
  WEIGHT_DECAY: 0.01

SCHEDULER:
  NAME: PolyLR
  POWER: 0.9
  WARMUP_EPOCHS: 3
  WARMUP_RATIO: 0.1

LOSS:
  NAME: CrossEntropy

EVAL:
  MODE: slide
  STRIDE: [512, 512]
  CROP_SIZE: [768, 768]
```

## Project Structure

```
UAVScenes-Benchmark/
├── CMNeXt/
├── DFormer/
├── TokenFusion/
├── GeminiFusion/
├── Sigma/
├── Mul_VMamba/
├── docs/
│   ├── INPUT_PROJECTION_LAYER_OZET.md
│   └── NewSplit.md
└── README.md
```

## Usage

### Training

```bash
# CMNeXt
cd CMNeXt
python train.py --config configs/uavscenes_rgb_hag.yaml

# DFormer
cd DFormer
python train.py -p local_configs/UAVScenes/DFormerv2_B.py

# TokenFusion
cd TokenFusion
python train.py --config configs/uavscenes_rgb_hag.yaml

# GeminiFusion
cd GeminiFusion
python train.py --config configs/uavscenes_rgb_hag.yaml

# Sigma
cd Sigma
python train.py -p configs/config_UAVScenes.py

# Mul_VMamba
cd Mul_VMamba
python train.py --config configs/uavscenes_rgbhagmulmamba.yaml
```

### Evaluation

```bash
python eval.py --config configs/uavscenes_rgb_hag.yaml --checkpoint best.pth
```

## Multi-Modal Approach

All models use a standardized approach for auxiliary modality (HAG):
- **1-channel HAG** is stacked to **3 channels** for compatibility with ImageNet pretrained weights
- This follows the original CMNeXt protocol

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- See individual model folders for specific requirements

## Citation

If you use this benchmark, please cite the original papers and our work.

## License

MIT License
