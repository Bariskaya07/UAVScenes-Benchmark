# UAVScenes Benchmark

This repository collects our UAVScenes benchmark implementations for multi-modal semantic segmentation. The current harmonized benchmark protocol is set up for these 6 models:

- `CMNeXt`
- `CMX`
- `TokenFusion`
- `GeminiFusion`
- `Sigma`
- `Mul_VMamba`

`DFormer` and `HRFuser` are still present in the repository, but they are not part of the current fully aligned UAVScenes benchmark recipe.

## Benchmark Scope

All 6 benchmarked models are trained on `UAVScenes` with `RGB + HAG` and a shared training recipe where possible:

- `19` semantic classes
- train crop size `768 x 768`
- batch size `8`
- `60` epochs
- `AdamW`, base LR `6e-5`, weight decay `0.01`
- polynomial LR decay with power `0.9`
- warmup `3` epochs, warmup ratio `0.1`
- mixed precision training
- fast `whole` validation during training
- sliding-window inference for final test evaluation

Important fairness rule:

- We standardize the training recipe.
- We do **not** force all models to share the same architecture-specific initialization policy, loss design, or fusion design.
- In other words, method-specific components are preserved when they are part of the original method design.

## Models

| Model | Paper | Backbone used in this benchmark | Notes |
| --- | --- | --- | --- |
| `CMNeXt` | Delivering Arbitrary-Modal Semantic Segmentation | `MiT-B2 + PPX` | UAVScenes RGB+HAG benchmark recipe aligned |
| `CMX` | Cross-Modal Fusion for RGB-X Semantic Segmentation | `MiT-B2` | Added as `CMX/` under this repo |
| `TokenFusion` | Multimodal Token Fusion for Vision Transformers | `MiT-B2` | Shared/parallel encoder setup preserved |
| `GeminiFusion` | GeminiFusion: Efficient Pixel-wise Multimodal Fusion | `MiT-B2` | Shared/parallel encoder setup preserved |
| `Sigma` | Siamese Mamba Network for Multi-Modal Semantic Segmentation | `Sigma-Tiny / VMamba-T` | UAVScenes RGB+HAG benchmark recipe aligned |
| `Mul_VMamba` | Multi-Modal Visual Mamba | `MulMamba-T / VMamba-T` | UAVScenes RGB+HAG benchmark recipe aligned |

## Dataset

`UAVScenes` is used with:

- `RGB`
- `HAG` (Height Above Ground)

Expected dataset root contents:

```text
UAVScenesData/
├── interval5_CAM_LIDAR/
├── interval5_CAM_label/
├── interval5_HAG_CSF/
├── interval5_LIDAR_label/
└── terra_3dmap_pointcloud_mesh/
```

For our bucket layout, the real dataset root is the **inner** `UAVScenesData` directory:

```text
thesis-uavscenes/uavscenes007/uavscenes-cmnext/UAVScenesData/UAVScenesData/
```

When you download it onto a VM, your local dataset root should be the directory that directly contains `interval5_CAM_LIDAR`, `interval5_CAM_label`, `interval5_HAG_CSF`, `interval5_LIDAR_label`, and `terra_3dmap_pointcloud_mesh`.

## Pretrained Weights

The benchmark currently expects these shared pretrained files:

- `mit_b2.pth`
- `hrt_tiny.pth`
- `Vmamba-T.pth`

Bucket location:

```text
thesis-uavscenes/uavscenes007/pretrained/
```

## Recommended VM Layout

On a fresh VM, keep the large files outside the repos and link the repos to them.

Example:

```text
<workspace>/
├── UAVScenes-Benchmark/
├── UAVScenesData/
└── pretrained/
    ├── mit_b2.pth
    ├── hrt_tiny.pth
    └── Vmamba-T.pth
```

Inside each repo, local links such as `data/UAVScenes` or `datasets/UAVScenes` can point to the same shared dataset root. The same idea applies to pretrained weights.

We intentionally do not commit machine-specific dataset symlinks, because absolute local paths would break on other VMs.

## Training Commands

Run these from the corresponding model directory.

### CMNeXt

```bash
python tools/train_mm.py --cfg configs/uavscenes_rgb_hag.yaml
```

### CMX

```bash
python train.py
```

### TokenFusion

```bash
python main.py --config configs/uavscenes_rgb_hag.yaml
```

### GeminiFusion

```bash
python main.py
```

### Sigma

```bash
python train.py --dataset_name uavscenes
```

### Mul_VMamba

```bash
python tools/train_mm.py --cfg configs/uavscenes_rgbhagmulmamba.yaml
```

## Evaluation Notes

- During training, validation is kept fast with `whole` inference.
- For final reporting on the test split, use the model-specific evaluation script with sliding-window inference.
- `Sigma/eval_slide.py` is included as a helper script for standalone sliding-window evaluation.

## Repository Layout

```text
UAVScenes-Benchmark/
├── CMNeXt/
├── CMX/
├── DFormer/
├── GeminiFusion/
├── HRFuser/
├── Mul_VMamba/
├── Sigma/
├── TokenFusion/
├── docs/
└── README.md
```

## Status

The main focus of this repository is now the aligned UAVScenes benchmark for:

- `CMNeXt`
- `CMX`
- `TokenFusion`
- `GeminiFusion`
- `Sigma`
- `Mul_VMamba`

If you extend the benchmark with additional methods, keep the shared recipe fixed and document any method-specific design choices that remain intentionally unchanged.
