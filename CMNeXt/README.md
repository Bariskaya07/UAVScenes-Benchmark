# CMNeXt UAVScenes Implementation

Paper-compliant implementation of **CMNeXt** (Delivering Arbitrary-Modal Semantic Segmentation) for **UAVScenes** dataset with RGB + HAG (Height Above Ground) modalities.

## 📋 Architecture Overview

Following the paper exactly:
- **Hub Branch (RGB):** MiT-B2 backbone with ImageNet pretrained weights
- **Auxiliary Branch (HAG):** PPX (Parallel Pooling Mixer) encoder with random initialization
- **Fusion:** Hub2Fuse mechanism with spatial reduction attention
- **Decoder:** SegFormer-style MLP head

### Key Paper Compliance
- ✅ PPX blocks for auxiliary modalities (not transformer)
- ✅ Only RGB branch uses pretrained weights
- ✅ HAG branch: random initialization
- ✅ 768×768 input size for memory efficiency

---

## 📂 Project Structure

```
CMNeXt_UAVScenes/
├── configs/
│   └── uavscenes_rgb_hag.yaml       # Training config (768×768, batch=8)
│
├── semseg/
│   ├── datasets/
│   │   └── uavscenes.py             # UAVScenes dataset loader (RGB + HAG)
│   │
│   ├── models/
│   │   ├── backbones/
│   │   │   └── mit.py               # MiT (Mix Transformer) B0-B5 backbones
│   │   │
│   │   ├── cmnext.py                # CMNeXt main model (Hub2Fuse + decoder)
│   │   └── ppx.py                   # PPX encoder (Parallel Pooling Mixer)
│   │
│   ├── augment.py                   # Data augmentation (random crop, flip, etc.)
│   ├── losses.py                    # Loss functions (CrossEntropy + optional Dice)
│   ├── metrics.py                   # mIoU, IoU per class, confusion matrix
│   ├── optimizers.py                # AdamW optimizer setup
│   └── scheduler.py                 # Polynomial LR scheduler
│
└── tools/
    ├── train_mm.py                  # Multi-modal training script
    └── val_mm.py                    # Multi-modal validation script
```

---

## 📄 File Descriptions

### Configuration
| File | Description |
|------|-------------|
| `configs/uavscenes_rgb_hag.yaml` | Training config: 768×768 input, batch=8, 40 epochs, paper settings |

### Dataset
| File | Description |
|------|-------------|
| `semseg/datasets/uavscenes.py` | UAVScenes dataloader with RGB + HAG support, label remapping (26→19 classes) |

### Models
| File | Description |
|------|-------------|
| `semseg/models/cmnext.py` | **Main model:** Hub2Fuse, CrossModalFusion, FeatureRectifyModule, SegFormerHead |
| `semseg/models/ppx.py` | **PPX Encoder:** PPXBlock, PPXStage, multi-scale pooling (paper Section 3.3) |
| `semseg/models/backbones/mit.py` | MiT transformer backbones (B0-B5) for RGB branch |

### Training Utilities
| File | Description |
|------|-------------|
| `semseg/augment.py` | RandomCrop, HorizontalFlip, ColorJitter, Normalize |
| `semseg/losses.py` | CrossEntropyLoss with label smoothing, Dice loss |
| `semseg/metrics.py` | mIoU calculation, per-class IoU, static/dynamic split |
| `semseg/optimizers.py` | AdamW with layer-wise learning rates |
| `semseg/scheduler.py` | PolynomialLR scheduler with warmup |

### Scripts
| File | Description |
|------|-------------|
| `tools/train_mm.py` | Training loop with gradient accumulation, AMP, checkpointing |
| `tools/val_mm.py` | Sliding window inference for full-resolution evaluation |

---

## 🏗️ Model Architecture Details

### CMNeXt (`semseg/models/cmnext.py`)

**Components:**
1. **Hub Backbone:** MiT-B2 for RGB (ImageNet pretrained)
2. **Auxiliary Encoder:** PPXEncoder for HAG (random init)
3. **Hub2Fuse Blocks:** Cross-modal fusion at 4 scales
4. **Decoder:** SegFormer MLP head

**Key Classes:**
- `CMNeXt`: Main model
- `Hub2FuseBlock`: Fusion block (rectify + cross-attention + enhance)
- `CrossModalFusion`: SRA-based cross-modal attention
- `FeatureRectifyModule`: Channel & spatial attention for alignment
- `SegFormerHead`: Multi-scale decoder

### PPX Encoder (`semseg/models/ppx.py`)

**Paper Section 3.3 Implementation:**
- `PPXBlock`: DW-Conv 7×7 → Parallel Pooling (3,7,11) → Gating → FFN+SE
- `PPXStage`: Patch embedding + N × PPXBlock
- `PPXEncoder`: 4-stage encoder matching MiT-B2 output channels [64, 128, 320, 512]

**Output:** Same spatial resolution as MiT-B2 for each stage

---

## 🎯 UAVScenes Dataset

**19 Classes (remapped from original 26):**
- **17 Static:** background, roof, roads, river, pool, bridge, container, airstrip, traffic_barrier, green_field, wild_field, solar_panel, umbrella, transparent_roof, car_park, paved_walk
- **2 Dynamic:** sedan, truck

**Train/Test Split:**
- Train: 16 scenes (~19k images)
- Test: 4 scenes (~5k images)

**Modalities:**
- **RGB:** Standard camera images
- **HAG:** Height Above Ground (16-bit PNG → normalized to [0,1])

---

## 🚀 Usage

### Training
```bash
python tools/train_mm.py --config configs/uavscenes_rgb_hag.yaml
```

### Validation
```bash
python tools/val_mm.py --config configs/uavscenes_rgb_hag.yaml
```

---

## 📊 Configuration Highlights

**From `configs/uavscenes_rgb_hag.yaml`:**
- Input size: 768×768 (reduced from 1024 for memory)
- Batch size: 8 (direct, no gradient accumulation)
- Optimizer: AdamW (lr=6e-5 for decoder, 6e-6 for backbone)
- Scheduler: Polynomial with 1500-step warmup
- Epochs: 40
- Evaluation: Sliding window (768 crop, 512 stride)

---

## 🔬 Key Implementation Details

1. **Paper Compliance:**
   - HAG branch uses PPX (not MiT-B2)
   - No pretrained weights for HAG encoder
   - Spatial reduction attention (SRA) for memory efficiency

2. **Memory Optimization:**
   - 768×768 input instead of 1024
   - SRA ratios: [8, 4, 2, 1] for stages 1-4
   - Mixed precision training (AMP)

3. **Label Processing:**
   - Original 26 classes → 19 used classes
   - Fast NumPy LUT-based remapping
   - Ignore label: 255

---

## 📝 Dependencies

```txt
torch>=1.10.0
torchvision>=0.11.0
numpy
opencv-python
pyyaml
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📖 References

- **CMNeXt Paper:** "Delivering Arbitrary-Modal Semantic Segmentation"
- **UAVScenes Dataset:** ICCV 2025
- **MiT Backbone:** SegFormer (NVIDIA)

---

## 👤 Author

Implementation by Barış Kaya
