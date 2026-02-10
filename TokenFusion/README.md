# TokenFusion for UAVScenes

TokenFusion implementation for RGB + HAG multi-modal semantic segmentation on UAVScenes dataset.

## Overview

This repository reproduces the TokenFusion paper ([Wang et al., CVPR 2022](https://arxiv.org/abs/2204.08721)) for the UAVScenes dataset with fair comparison settings matching CMNeXt and DFormerv2.

### Key Features

- **MiT-B2 backbone** for fair comparison with CMNeXt
- **TokenFusion mechanism** for multi-modal feature fusion
- **RGB + HAG (Height Above Ground)** input modalities
- **768x768 training resolution** matching CMNeXt/DFormerv2
- **Sliding window evaluation** (768x768, stride 512)

## Fair Comparison Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| Image Size | 768x768 | CMNeXt config |
| Batch Size | 8 | CMNeXt config |
| HAG Max Height | 50m | Both implementations |
| Classes | 19 | UAVScenes label mapping |
| Train/Test Split | 16/4 scenes | Same as CMNeXt/DFormerv2 |
| Backbone | MiT-B2 | CMNeXt uses MiT-B2 |
| Evaluation | Sliding window 768x768, stride 512 | CMNeXt config |

## Installation

```bash
# Create conda environment
conda create -n tokenfusion python=3.10
conda activate tokenfusion

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
mkdir -p pretrained
wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth -O pretrained/mit_b2.pth
```

## Dataset Structure

The UAVScenes dataset should be organized as:

```
UAVScenesData/
├── interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg  (RGB)
├── interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png  (Label)
└── interval5_HAG/{scene}/{timestamp}.png  (HAG - 16-bit PNG)
```

## Training

```bash
# Start training
./train.sh

# Or manually:
python main.py --config configs/uavscenes_rgb_hag.yaml
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pth --config configs/uavscenes_rgb_hag.yaml
```

## Model Architecture

### TokenFusion Mechanism

1. **MixVisionTransformer**: Parallel processing for 2 modalities (RGB + HAG)
2. **PredictorLG**: Token importance scoring network
3. **TokenExchange**: Swaps uninformative tokens between modalities (threshold=0.02)
4. **LayerNormParallel**: Individual LayerNorm per modality
5. **SegFormerHead**: MLP-based decoder

### Loss Function

```
Loss = NLLLoss(output, target) + λ * L1_loss(masks)
```

Where λ = 1e-4 for L1 sparsity on token masks.

## Classes (19)

| ID | Class | Type |
|----|-------|------|
| 0 | background | Static |
| 1 | roof | Static |
| 2 | dirt_road | Static |
| 3 | paved_road | Static |
| 4 | river | Static |
| 5 | pool | Static |
| 6 | bridge | Static |
| 7 | container | Static |
| 8 | airstrip | Static |
| 9 | traffic_barrier | Static |
| 10 | green_field | Static |
| 11 | wild_field | Static |
| 12 | solar_panel | Static |
| 13 | umbrella | Static |
| 14 | transparent_roof | Static |
| 15 | car_park | Static |
| 16 | paved_walk | Static |
| 17 | sedan | Dynamic |
| 18 | truck | Dynamic |

## Metrics

- **mIoU**: Mean Intersection over Union (all 19 classes)
- **Static mIoU**: mIoU for static classes (0-16)
- **Dynamic mIoU**: mIoU for dynamic classes (17-18: sedan, truck)
- **Pixel Accuracy**: Overall pixel classification accuracy

## Project Structure

```
TokenFusion_UAVScenes/
├── configs/
│   └── uavscenes_rgb_hag.yaml    # Training configuration
├── datasets/
│   ├── __init__.py
│   └── uavscenes.py              # UAVScenes dataset loader
├── models/
│   ├── __init__.py
│   ├── modules.py                # TokenExchange, ModuleParallel
│   ├── mix_transformer.py        # MixVisionTransformer backbone
│   └── segformer.py              # WeTr model with SegFormerHead
├── utils/
│   ├── __init__.py
│   ├── transforms.py             # Data augmentation
│   ├── optimizer.py              # PolyWarmupAdamW
│   ├── metrics.py                # mIoU computation
│   └── helpers.py                # Logging, checkpointing
├── pretrained/                   # MiT-B2 pretrained weights
├── main.py                       # Training script
├── evaluate.py                   # Evaluation script
├── train.sh                      # Training shell script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## References

- [TokenFusion Paper](https://arxiv.org/abs/2204.08721)
- [TokenFusion GitHub](https://github.com/yikaiw/TokenFusion)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [CMNeXt Paper](https://arxiv.org/abs/2302.10035)
- [DFormerv2 Paper](https://arxiv.org/abs/2309.04466)

## Citation

```bibtex
@inproceedings{wang2022multimodal,
  title={Multimodal Token Fusion for Vision Transformers},
  author={Wang, Yikai and Chen, Xinghao and Cao, Lele and Huang, Wenbing and Sun, Fuchun and Wang, Yunhe},
  booktitle={CVPR},
  year={2022}
}
```
