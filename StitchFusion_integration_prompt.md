# StitchFusion UAVScenes Benchmark Integration — GPT Prompt

You are integrating **StitchFusion** into the UAVScenes RGB+HAG multimodal semantic segmentation benchmark. The benchmark already contains 7 models (CMX, CMNeXt, TokenFusion, GeminiFusion, HRFuser, Sigma, Mul-VMamba), all using **identical training parameters** for fair comparison. StitchFusion must use the exact same parameters — no exceptions.

**Source repo:** https://github.com/LiBingyu01/StitchFusion
**Paper:** "StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation" (ECCV 2024)

---

## 1. Target Directory Structure

Create `UAVScenes-Benchmark/StitchFusion/` following the existing benchmark pattern:

```
UAVScenes-Benchmark/
├── StitchFusion/
│   ├── config.py                    # Training configuration (copy pattern from CMX/config.py)
│   ├── train.py                     # Training script (copy pattern from CMX/train.py)
│   ├── eval.py                      # Evaluation script
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stitchfusion.py          # Main model (adapted from repo)
│   │   ├── builder.py               # Model builder with activation checkpoint config
│   │   └── encoders/
│   │       └── mix_transformer.py   # MiT-B2 backbone (shared SegFormer encoder)
│   ├── datasets/
│   │   └── UAVScenesDataset.py      # Copy from CMX/dataloader/UAVScenesDataset.py
│   ├── dataloader/
│   │   └── dataloader.py            # Copy from CMX/dataloader/dataloader.py
│   ├── utils/
│   │   ├── lr_policy.py             # Copy from CMX/utils/lr_policy.py (WarmUpPolyLR)
│   │   ├── pyt_utils.py             # Copy from CMX/utils/pyt_utils.py
│   │   └── metric.py                # Copy from CMX/utils/metric.py
│   ├── engine/
│   │   └── engine.py                # Copy from CMX/engine/engine.py
│   └── datasets/
│       └── UAVScenes -> ../../CMX/datasets/UAVScenes  (symlink to shared data)
```

---

## 2. Exact Training Parameters (MANDATORY — no changes allowed)

These are the standardized benchmark parameters. Copy them exactly into `config.py`:

```python
import os
import os.path as osp
import sys
import time
from pathlib import Path
import numpy as np
from easydict import EasyDict as edict

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared_paths import resolve_pretrained_path

C = edict()
config = C
cfg = C

C.seed = 42

C.root_dir = osp.dirname(osp.abspath(__file__))
C.abs_dir = C.root_dir

# ── Dataset ──
C.dataset_name = 'UAVScenes'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'UAVScenes')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, 'HAG')
C.x_format = '.png'
C.x_is_single_channel = False
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = None
C.num_eval_imgs = None
C.num_classes = 19
C.class_names = [
    'background', 'roof', 'dirt_road', 'paved_road', 'river',
    'pool', 'bridge', 'container', 'airstrip', 'traffic_barrier',
    'green_field', 'wild_field', 'solar_panel', 'umbrella',
    'transparent_roof', 'car_park', 'paved_walk', 'sedan', 'truck'
]

# ── Image ──
C.background = 255
C.image_height = 768
C.image_width = 768
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# ── Network ──
C.backbone = 'mit_b2'
C.pretrained_model = resolve_pretrained_path('pretrained/mit_b2.pth', C.root_dir)
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 768
C.optimizer = 'AdamW'

# ── StitchFusion-specific ──
C.moa_type = 'obMoA'       # bi-directional Modality Adapter (paper default)
C.moa_r = 8                # MoA hidden dimension (paper Table 3)
C.freeze_backbone = True    # StitchFusion freezes shared backbone, trains MoA + decoder

# ── Training ──
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 30
C.niters_per_epoch = None
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
C.warm_up_epoch = 2
C.warmup_ratio = 0.1
C.use_photometric = True
C.use_gaussian_blur = True
C.gaussian_blur_prob = 0.2
C.gaussian_blur_kernel = 3
C.freeze_bn = True
C.amp_dtype = 'bf16'
C.activation_checkpoint = True

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# ── Eval ──
C.eval_iter = 5
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [768, 768]

# ── Checkpoint ──
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

# ── Paths ──
C.log_dir = osp.join(C.root_dir, 'log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.join(C.log_dir, "tb")
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.root_dir, "stitchfusion_checkpoints")

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'
```

---

## 3. Loss Function (MANDATORY)

```python
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
```

This is the ONLY loss function. Do NOT add auxiliary losses, deep supervision, or any other loss term.

---

## 4. LR Scheduler (MANDATORY)

Use `WarmUpPolyLR` from `utils/lr_policy.py`:

```python
class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps, warmup_ratio=0.0):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

    def get_lr(self, cur_iter):
        if self.warmup_steps > 0 and cur_iter < self.warmup_steps:
            alpha = cur_iter / self.warmup_steps
            return self.start_lr * (self.warmup_ratio + (1 - self.warmup_ratio) * alpha)
        else:
            if self.warmup_steps > 0:
                progress = (cur_iter - self.warmup_steps) / (self.total_iters - self.warmup_steps)
            else:
                progress = float(cur_iter) / self.total_iters
            return self.start_lr * ((1 - progress) ** self.lr_power)
```

Instantiation:
```python
niters_per_epoch = len(train_loader)
total_iters = config.nepochs * niters_per_epoch
warmup_iters = config.warm_up_epoch * niters_per_epoch

lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iters, warmup_iters, config.warmup_ratio)
```

LR is set BEFORE each forward pass (no 1-iter lag):
```python
current_idx = (epoch - 1) * niters_per_epoch + idx
lr = lr_policy.get_lr(current_idx)
for pg in optimizer.param_groups:
    pg['lr'] = lr
```

---

## 5. Optimizer (MANDATORY)

```python
optimizer = torch.optim.AdamW(
    params,                    # only trainable params (MoA + decoder if backbone is frozen)
    lr=config.lr,              # 6e-5
    betas=(0.9, 0.999),
    weight_decay=config.weight_decay,  # 0.01
)
```

For StitchFusion specifically, since the backbone is frozen:
```python
# Freeze backbone parameters
for param in model.backbone.parameters():
    param.requires_grad = False

# Only optimize MoA adapters + decoder
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
```

---

## 6. BF16 Mixed Precision (MANDATORY)

```python
def get_amp_dtype():
    amp_dtype = str(getattr(config, 'amp_dtype', 'bf16')).lower()
    if amp_dtype == 'bf16':
        return torch.bfloat16
    return torch.float16

amp_dtype = get_amp_dtype()
# GradScaler is ONLY used with fp16, NOT with bf16 (bf16 doesn't need loss scaling)
use_grad_scaler = torch.cuda.is_available() and amp_dtype == torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
```

Forward pass:
```python
with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=amp_dtype):
    loss = model(imgs, modal_xs, gts)
```

Backward pass (scaler-aware):
```python
optimizer.zero_grad(set_to_none=True)
if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
else:
    loss.backward()

# Gradient clipping AFTER unscale
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if scaler.is_enabled():
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()
```

---

## 7. NaN Guards (MANDATORY)

Two-level NaN protection, matching CMX/CMNeXt exactly:

### 7a. Loss-level guard (after forward pass, before backward):
```python
if torch.isnan(loss) or torch.isinf(loss):
    rgb_finite = bool(torch.isfinite(imgs).all().item())
    modal_finite = bool(torch.isfinite(modal_xs).all().item())
    logger.warning(
        f'NaN/Inf loss at epoch {epoch} iter {idx}, skipping... '
        f'(valid_pixels={valid_pixels}, rgb_finite={rgb_finite}, modal_finite={modal_finite})'
    )
    optimizer.zero_grad(set_to_none=True)
    del loss, imgs, gts, modal_xs, minibatch
    torch.cuda.empty_cache()
    continue
```

### 7b. Gradient-level guard (after clip_grad_norm_, before optimizer.step):
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    logger.warning(
        f'NaN/Inf gradient at epoch {epoch} iter {idx}, skipping step... '
        f'(valid_pixels={valid_pixels})'
    )
    optimizer.zero_grad(set_to_none=True)
    if scaler.is_enabled():
        scaler.update()
    del grad_norm, loss, imgs, gts, modal_xs, minibatch
    torch.cuda.empty_cache()
    continue
```

### 7c. All-ignore batch guard (before forward pass):
```python
valid_pixels = int((gts != config.background).sum().item())
if valid_pixels == 0:
    logger.warning(f'All-ignore batch at epoch {epoch} iter {idx}, skipping...')
    optimizer.zero_grad(set_to_none=True)
    del imgs, gts, modal_xs, minibatch
    torch.cuda.empty_cache()
    continue
```

---

## 8. Activation Checkpointing

StitchFusion has two checkpointable components:

### 8a. MiT-B2 Encoder Blocks (if backbone is NOT frozen)
Even if backbone is frozen, if gradients flow through it (which they do for MoA adapters), checkpoint the transformer blocks:

```python
from torch.utils.checkpoint import checkpoint

def _should_checkpoint(self, *tensors):
    if not (self.use_checkpoint and self.training and torch.is_grad_enabled()):
        return False
    return any(t.requires_grad for t in tensors if torch.is_tensor(t))

def _run_block(self, block, x, H, W):
    if not self._should_checkpoint(x):
        return block(x, H, W)
    def custom_forward(inp, module=block, height=H, width=W):
        return module(inp, height, width)
    return checkpoint(custom_forward, x, use_reentrant=False)
```

### 8b. MoA (Modality Adapter) Blocks
The MoA adapters are lightweight (LoRA-inspired: down-project → ReLU → up-project), but if they're within a checkpointed encoder block, they get checkpointed automatically. If MoA is applied between stages, wrap it:

```python
def _run_moa(self, moa_module, rgb_feat, aux_feat):
    if not self._should_checkpoint(rgb_feat, aux_feat):
        return moa_module(rgb_feat, aux_feat)
    def custom_forward(r, a, module=moa_module):
        return module(r, a)
    return checkpoint(custom_forward, rgb_feat, aux_feat, use_reentrant=False)
```

### 8c. Safety Rules
- **NEVER** use `inplace=True` in any ReLU or operation inside a checkpointed path. Use `nn.ReLU(inplace=False)` or `F.relu(x)` (without inplace).
- Always use `use_reentrant=False` in `checkpoint()`.
- Add BN safety guard if module contains BatchNorm:
```python
def _bn_checkpoint_safe(module):
    """Check if all BN layers in module are in eval mode or have track_running_stats=False."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.training and m.track_running_stats:
                return False
    return True
```
- Config flag: `activation_checkpoint = True` in config.py, passed to model via `getattr(cfg, 'activation_checkpoint', False)`.

---

## 9. StitchFusion Architecture Adaptation

### 9a. Paper Architecture (Section 3)
StitchFusion uses a **shared SegFormer (MiT-B2) backbone** for both modalities, with **Modality Adapters (MoA)** inserted between backbone stages. The architecture:

1. **Shared Encoder**: Single MiT-B2 processes both RGB and auxiliary (HAG) modality sequentially or in parallel
2. **MoA Adapters**: Lightweight LoRA-inspired adapters that adapt shared features for each modality
   - `obMoA` (bi-directional): adapts features in both directions (RGB↔Aux)
   - Hidden dimension `r=8`
   - Applied between each encoder stage
3. **Decoder**: SegFormer-style MLP decoder (same as CMX/CMNeXt)

### 9b. Key Modifications for UAVScenes
- Input: RGB (3ch) + HAG (3ch, stacked from 1ch normalized height-above-ground)
- Output: 19 classes
- Backbone: MiT-B2 pretrained weights from `pretrained/mit_b2.pth`
- The backbone is frozen; only MoA adapters and decoder are trained
- Feature dimensions from MiT-B2: [64, 128, 320, 512] at stages 1-4

### 9c. Model Forward Pass
```python
def forward(self, rgb, modal_x, label=None):
    # rgb: [B, 3, H, W], modal_x: [B, 3, H, W]

    # Extract multi-scale features with MoA fusion
    fused_features = self.encoder_with_moa(rgb, modal_x)  # list of 4 feature maps

    # Decode
    logits = self.decode_head(fused_features)  # [B, num_classes, H/4, W/4]
    logits = F.interpolate(logits, size=rgb.shape[2:], mode='bilinear', align_corners=False)

    if label is not None:
        loss = self.criterion(logits, label)
        return loss
    return logits
```

---

## 10. Checkpoint Saving System (MANDATORY)

Import from the shared benchmark utility:
```python
import sys, os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from checkpoint_ops import epoch_checkpoint_name, promote_best_checkpoint, maybe_sync_checkpoint_dir
```

### 10a. Checkpoint naming
- Epoch checkpoints: `stitchfusion_epoch_{N}.pth` (model_slug = `'stitchfusion'`)
- Best checkpoint: `stitchfusion_epoch_{N}_best.pth`
- Symlinks: `last.pth`, `best.pth`

### 10b. engine.py modification
In `save_and_link_checkpoint`, the checkpoint filename should use model slug:
```python
current_epoch_checkpoint = osp.join(
    checkpoint_dir, 'stitchfusion_epoch_{}.pth'.format(self.state.epoch)
)
```

### 10c. Saving best checkpoint
```python
if mean_IoU > best_miou:
    best_miou = mean_IoU
    best_epoch = epoch
    epoch_ckpt = osp.join(config.checkpoint_dir, epoch_checkpoint_name('stitchfusion', epoch))
    if osp.exists(epoch_ckpt):
        best_epoch_path = promote_best_checkpoint(epoch_ckpt, 'stitchfusion', epoch)
        link_file(best_epoch_path, osp.join(config.checkpoint_dir, 'best.pth'))
```

### 10d. End-of-training bucket sync
```python
# At the very end of training
maybe_sync_checkpoint_dir(config.checkpoint_dir, logger=logger.info)
```

---

## 11. UAVScenes Dataset

Copy `CMX/dataloader/UAVScenesDataset.py` as-is. Key details:

- **19 classes** with label remapping from original 26-class UAVScenes annotations
- **HAG loading**: 16-bit PNG → `(pixel - 20000) / 1000.0` → meters → clip to [0, 50m] → normalize [0, 1] → stack to 3 channels
- **Scene-based splits**:
  - Train: 13 scenes (AMtown01, AMvalley01, HKairport01-03, HKairport_GNSS02-03, HKairport_GNSS_Evening, HKisland01-03, HKisland_GNSS02-03)
  - Val: 3 scenes (AMtown02, AMvalley02, HKisland_GNSS01)
  - Test: 4 scenes (AMtown03, AMvalley03, HKairport_GNSS01, HKisland_GNSS_Evening)
- **Returns dict**: `{data: rgb_tensor, label: gt_tensor, modal_x: hag_tensor, fn: str, n: int}`

Dataset symlink:
```bash
ln -s ../../CMX/datasets/UAVScenes StitchFusion/datasets/UAVScenes
```

---

## 12. Data Augmentation Pipeline (MANDATORY — exact order)

```
RandomResize (continuous uniform from train_scale_array=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
→ RandomCrop(768×768, pad_value=0 for RGB/HAG, pad_value=255 for label)
→ RandomHorizontalFlip(p=0.5)
→ PhotometricDistortion (RGB only — brightness, contrast, saturation, hue)
→ GaussianBlur (RGB only, p=0.2, kernel=3)
→ Normalize RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
→ Normalize HAG: (x - 0.5) / 0.5
→ Transpose to CHW
```

Copy `CMX/dataloader/dataloader.py` (TrainPre class) exactly.

---

## 13. Seed and Reproducibility

```python
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(config.seed)  # seed = 42
```

---

## 14. Freeze BatchNorm (MANDATORY)

```python
def apply_freeze_bn_if_needed(net):
    if not getattr(config, 'freeze_bn', False):
        return
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
```

Call after `model.train()` and after validation:
```python
model.train()
apply_freeze_bn_if_needed(model)
```

---

## 15. VRAM Memory Logging

```python
def get_cuda_memory_stats_mb(device):
    if not torch.cuda.is_available():
        return None
    scale = 1024 ** 2
    return {
        'allocated': torch.cuda.memory_allocated(device) / scale,
        'reserved': torch.cuda.memory_reserved(device) / scale,
        'peak_allocated': torch.cuda.max_memory_allocated(device) / scale,
        'peak_reserved': torch.cuda.max_memory_reserved(device) / scale,
    }
```

Log at iter 1 and every 100 iters:
```python
if (idx + 1) == 1 or (idx + 1) % 100 == 0:
    logger.info('Epoch %d Iter %d memory: alloc=%.0fMiB reserved=%.0fMiB peak_alloc=%.0fMiB peak_reserved=%.0fMiB', ...)
```

Also reset peak stats at epoch start:
```python
torch.cuda.reset_peak_memory_stats(device)
```

---

## 16. Validation

Use whole-image batched validation (same as CMX):
```python
@torch.no_grad()
def validate_batched(model, val_dataset, device, num_classes, batch_size=8):
    model.eval()
    # Compute confusion matrix, mIoU, pixel accuracy
    # See CMX/train.py validate_batched for exact implementation
```

Validate at: `epoch >= checkpoint_start_epoch(5) and epoch % checkpoint_step(5) == 0`, and at the final epoch.

---

## 17. Shared Pretrained Path Resolution

StitchFusion uses the shared `shared_paths.py` at the benchmark root. Import it in config.py:
```python
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
from shared_paths import resolve_pretrained_path

C.pretrained_model = resolve_pretrained_path('pretrained/mit_b2.pth', C.root_dir)
```

This resolves MiT-B2 weights from multiple possible locations (project-local, benchmark-root, home directory, env vars).

---

## 18. Summary of What to Implement

1. **Clone/adapt** StitchFusion model code from GitHub repo into `UAVScenes-Benchmark/StitchFusion/`
2. **Adapt the model** for 2-modality RGB+HAG input (both 3-channel) with MiT-B2 backbone, obMoA (r=8), SegFormer MLP decoder (embed_dim=768), 19-class output
3. **Copy** dataset, dataloader, augmentation pipeline, LR scheduler, engine from CMX
4. **Write config.py** with exact parameters listed above
5. **Write train.py** with:
   - BF16 AMP (with conditional GradScaler for fp16 fallback)
   - NaN guards (loss-level + gradient-level + all-ignore-batch)
   - Gradient clipping (max_norm=1.0)
   - WarmUpPolyLR scheduling
   - Activation checkpointing on encoder blocks and MoA adapters
   - Frozen backbone with trainable MoA + decoder
   - Frozen BatchNorm
   - VRAM memory logging
   - Checkpoint saving via `checkpoint_ops.py` (model_slug='stitchfusion')
   - Bucket sync at end of training
   - Seed=42 with deterministic mode
6. **Ensure no `inplace=True`** in any checkpointed code path
7. **Test** that the model runs for at least 1 epoch without errors

---

## 19. Things NOT To Do

- Do NOT change any training hyperparameter (lr, batch_size, epochs, weight_decay, etc.)
- Do NOT add auxiliary losses, deep supervision, or extra loss terms
- Do NOT change the augmentation pipeline
- Do NOT use a different optimizer or scheduler
- Do NOT use fp16 — use bf16 (config.amp_dtype = 'bf16')
- Do NOT skip NaN guards or gradient clipping
- Do NOT use `use_reentrant=True` in checkpoint calls
- Do NOT use `inplace=True` in any ReLU/operation in a checkpointed path
- Do NOT add new evaluation metrics or change the validation procedure
- Do NOT modify the dataset class, label remapping, or HAG loading

---

## 20. Reference Files to Study

Before implementing, read these files from the benchmark for reference patterns:
- `CMX/config.py` — config pattern
- `CMX/train.py` — complete training loop with all guards
- `CMX/models/builder.py` — how activation_checkpoint flows from config to model
- `CMX/models/encoders/dual_segformer.py` — MiT-B2 checkpoint pattern
- `CMX/dataloader/UAVScenesDataset.py` — dataset class
- `CMX/dataloader/dataloader.py` — TrainPre augmentation + dataloader factory
- `CMX/utils/lr_policy.py` — WarmUpPolyLR
- `CMX/engine/engine.py` — Engine class with save_and_link_checkpoint
- `checkpoint_ops.py` — shared checkpoint naming/sync utilities
- `shared_paths.py` — pretrained weight path resolution
