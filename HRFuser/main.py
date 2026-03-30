"""
HRFuser Training Script for UAVScenes Dataset

Training protocol matches TokenFusion UAVScenes for fair comparison benchmark.
Uses RGB + HAG (Height Above Ground) multimodal input with HRFuser backbone.

Fair Comparison Settings (same as TokenFusion, CMNeXt, DFormerv2, GeminiFusion):
- Image Size: 768x768
- Batch Size: 8
- HAG Max Height: 50m
- Classes: 19
- Backbone: HRFuser-T (Tiny)
- Optimizer: AdamW, lr=6e-5, PolyLR power=0.9
- Epochs: 60
- Seed: 42

Usage:
    python main.py --config configs/uavscenes_rgb_hag.yaml
    python main.py --config configs/uavscenes_rgb_hag.yaml --resume checkpoints/checkpoint_epoch_10.pth
    python main.py --config configs/uavscenes_rgb_hag.yaml --eval-only --resume checkpoints/best.pth
"""

import os
import sys
import argparse
import yaml
import time
import random
import shutil
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from models.hrfuser_segformer import HRFuserSegFormer
from datasets.uavscenes import UAVScenesDataset, CLASS_NAMES
from utils.optimizer import PolyWarmupAdamW, get_fair_param_groups
from utils.augmentations_mm import (
    RandomColorJitter, RandomHorizontalFlip, RandomGaussianBlur,
    RandomResizedCrop, Normalize, Resize
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_paths import resolve_pretrained_path
from checkpoint_ops import materialize_epoch_checkpoint, promote_best_checkpoint, maybe_sync_checkpoint_dir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """Configuration class to hold YAML config as attributes."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def resolve_amp_dtype(dtype_name):
    """Map config AMP dtype string to torch dtype."""
    dtype_name = str(dtype_name).lower()
    if dtype_name == 'bf16':
        return torch.bfloat16
    if dtype_name == 'fp16':
        return torch.float16
    raise ValueError(f"Unsupported training.amp_dtype='{dtype_name}'. Expected 'bf16' or 'fp16'.")


def get_amp_dtype(cfg):
    """Return configured AMP autocast dtype."""
    return resolve_amp_dtype(getattr(cfg.training, 'amp_dtype', 'bf16'))


def parse_args():
    parser = argparse.ArgumentParser(description='HRFuser UAVScenes Training')
    parser.add_argument('--config', type=str, default='configs/uavscenes_rgb_hag.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    repo_root = config_path.parent.parent

    def resolve_path(path_value):
        path_obj = Path(path_value)
        if path_obj.is_absolute():
            return str(path_obj)
        return str((repo_root / path_obj).resolve())

    config_dict['dataset']['data_path'] = resolve_path(config_dict['dataset']['data_path'])
    config_dict['model']['pretrained'] = resolve_pretrained_path(config_dict['model']['pretrained'], repo_root)
    config_dict['logging']['log_dir'] = resolve_path(config_dict['logging']['log_dir'])
    config_dict['logging']['checkpoint_dir'] = resolve_path(config_dict['logging']['checkpoint_dir'])
    return Config(config_dict)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir=None, name='hrfuser', is_main=True):
    """Setup logger for training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if not is_main:
        logger.addHandler(logging.NullHandler())
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Log file: {log_file}')

    return logger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        # Validation runs only on rank0, so other ranks can wait at barrier for a long time.
        # Increase process group timeout to avoid NCCL watchdog timeouts during validation.
        timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "60"))
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=timeout_minutes),
        )
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_complexity(model, cfg, logger, device, is_main=True):
    """Log parameter counts and FLOPs at training start."""
    if not is_main:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameters: {total_params / 1e6:.2f}M total, "
        f"{trainable_params / 1e6:.2f}M trainable"
    )

    if not FVCORE_AVAILABLE:
        logger.info("FLOPs: fvcore not installed (pip install -r requirements.txt)")
        return

    # Use benchmark input size for reported complexity.
    h = int(cfg.training.image_size)
    w = int(cfg.training.image_size)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            rgb = torch.zeros(1, 3, h, w, device=device)
            hag = torch.zeros(1, 3, h, w, device=device)
            flops = FlopCountAnalysis(model, (rgb, hag))
            total_flops = flops.total()
        logger.info(f"FLOPs (1x3x{h}x{w} RGB + 1x3x{h}x{w} HAG): {total_flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        logger.info(f"FLOPs: could not calculate ({e})")
    finally:
        if was_training:
            model.train()


def save_checkpoint(state, filename='checkpoint.pth', is_best=False, best_filename='best.pth'):
    """Save training checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_torch_checkpoint(path, map_location):
    """Load checkpoint with PyTorch>=2.6 and older-version compatibility."""
    try:
        # We save full training checkpoints (model/optimizer/metadata), so
        # weights_only must be False on PyTorch 2.6+.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch<2.6 does not have weights_only argument.
        return torch.load(path, map_location=map_location)


def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """Load checkpoint."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = load_torch_checkpoint(filename, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'segmenter' in checkpoint:
        state_dict = checkpoint['segmenter']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)

    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    info = {
        'epoch': checkpoint.get('epoch', checkpoint.get('epoch_start', 0)),
        'best_miou': checkpoint.get('best_miou', checkpoint.get('best_val', 0)),
    }

    return info


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, distributed=False, rank=0, world_size=1):
    """Build training and validation dataloaders."""
    from torchvision import transforms
    from utils.transforms import ToTensor

    crop_size = [cfg.training.crop_size, cfg.training.crop_size]

    # Training transforms
    composed_trn = transforms.Compose([
        ToTensor(),
        RandomColorJitter(p=cfg.augmentation.color_jitter_p),
        RandomHorizontalFlip(p=cfg.augmentation.flip_prob),
        RandomGaussianBlur(cfg.augmentation.gaussian_blur_kernel,
                           p=cfg.augmentation.gaussian_blur_p),
        RandomResizedCrop(crop_size,
                          scale=tuple(cfg.augmentation.scale_range),
                          seg_fill=cfg.dataset.ignore_label),
        Normalize(cfg.normalization.rgb_mean, cfg.normalization.rgb_std),
    ])

    # Validation transforms
    composed_val = transforms.Compose([
        ToTensor(),
        Resize(crop_size),
        Normalize(cfg.normalization.rgb_mean, cfg.normalization.rgb_std),
    ])

    # Create datasets
    train_dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path,
        split='train',
        transform=composed_trn,
        hag_max_height=cfg.hag.max_meters
    )

    val_dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path,
        split='val',
        transform=composed_val,
        hag_max_height=cfg.hag.max_meters
    )

    # Create dataloaders
    global_batch_size = cfg.training.batch_size
    if distributed:
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"Global batch_size ({global_batch_size}) must be divisible by WORLD_SIZE ({world_size}). "
                f"Use a divisible value, e.g. {world_size}, {2 * world_size}, ..."
            )
        per_gpu_batch_size = global_batch_size // world_size
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    else:
        per_gpu_batch_size = global_batch_size
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_batch_size = int(getattr(cfg.evaluation, 'batch_size', 8))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_sampler, per_gpu_batch_size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    """Build HRFuser segmentation model."""
    model = HRFuserSegFormer(
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        drop_path_rate=cfg.model.drop_path_rate,
        num_fused_modalities=cfg.model.num_fused_modalities,
        mod_in_channels=cfg.model.mod_in_channels
    )

    # Load pretrained weights
    pretrained = cfg.model.pretrained
    if pretrained and os.path.isfile(pretrained):
        model.load_pretrained(pretrained)
    elif pretrained:
        print(f"Warning: pretrained weights not found at {pretrained}")

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def slide_inference(model, rgb, hag, num_classes, crop_size, stride):
    """Sliding window inference."""
    B, _, H, W = rgb.shape
    crop_h, crop_w = crop_size, crop_size
    stride_h, stride_w = stride, stride

    h_grids = max((H - crop_h + stride_h - 1) // stride_h + 1, 1)
    w_grids = max((W - crop_w + stride_w - 1) // stride_w + 1, 1)

    preds = rgb.new_zeros((B, num_classes, H, W))
    count_mat = rgb.new_zeros((B, 1, H, W))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride_h
            x1 = w_idx * stride_w
            y2 = min(y1 + crop_h, H)
            x2 = min(x1 + crop_w, W)
            y1 = max(y2 - crop_h, 0)
            x1 = max(x2 - crop_w, 0)

            crop_rgb = rgb[:, :, y1:y2, x1:x2]
            crop_hag = hag[:, :, y1:y2, x1:x2]

            crop_pred = model(crop_rgb, crop_hag)

            if crop_pred.shape[2:] != (y2 - y1, x2 - x1):
                crop_pred = F.interpolate(crop_pred, size=(y2 - y1, x2 - x1),
                                          mode='bilinear', align_corners=False)

            preds[:, :, y1:y2, x1:x2] += crop_pred
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat > 0).all()
    preds = preds / count_mat
    return preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, criterion, cfg,
                    epoch, device, scaler, logger, distributed=False, world_size=1, is_main=True):
    """Train for one epoch."""
    model.train()

    # Freeze BN
    if cfg.training.freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    amp_enabled = cfg.training.amp
    amp_dtype = get_amp_dtype(cfg)

    end = time.time()
    num_iters = len(train_loader)

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        rgb = sample['rgb'].to(device).float()
        hag = sample['depth'].to(device).float()  # HAG stored as 'depth' key
        target = sample['mask'].to(device).long()

        # Forward pass with AMP
        with autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            output = model(rgb, hag)

            # Upsample to target size
            output = F.interpolate(output, size=target.shape[1:],
                                   mode='bilinear', align_corners=False)

            loss = criterion(output, target)

        if not torch.isfinite(loss):
            logger.warning(
                f'Non-finite loss at epoch {epoch} iter {i}, skipping step... '
                f'(rgb_finite={bool(torch.isfinite(rgb).all().item())}, '
                f'hag_finite={bool(torch.isfinite(hag).all().item())})'
            )
            optimizer.zero_grad()
            del loss, output, rgb, hag, target
            torch.cuda.empty_cache()
            continue

        # Backward pass
        optimizer.zero_grad()
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            logger.warning(f'Non-finite gradient at epoch {epoch} iter {i}, skipping step...')
            optimizer.zero_grad()
            if amp_enabled:
                scaler.update()
            del grad_norm, loss, output, rgb, hag, target
            torch.cuda.empty_cache()
            continue

        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update meters
        loss_meter.update(loss.item(), rgb.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if is_main and (i + 1) % cfg.logging.print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{i + 1}/{num_iters}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'LR: {current_lr:.6f} '
                f'Data: {data_time.avg:.3f}s '
                f'Batch: {batch_time.avg:.3f}s'
            )

    epoch_loss = torch.tensor(loss_meter.avg, dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss /= world_size
    return float(epoch_loss.item())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, cfg, device, logger):
    """Validate model with whole-image or sliding-window inference."""
    model.eval()

    num_classes = cfg.dataset.num_classes
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    eval_mode = getattr(cfg.evaluation, 'val_mode', getattr(cfg.evaluation, 'mode', 'whole'))
    eval_mode = str(eval_mode).lower()
    if eval_mode not in {'whole', 'slide'}:
        logger.warning(f"Unknown validation mode '{eval_mode}', falling back to 'whole'")
        eval_mode = 'whole'

    mode_str = "whole image" if eval_mode == 'whole' else "sliding window"
    logger.info(f"Running validation with {mode_str} inference...")
    start_time = time.time()
    inference_time = 0.0
    inference_samples = 0

    for i, sample in enumerate(val_loader):
        rgb = sample['rgb'].to(device).float()
        hag = sample['depth'].to(device).float()
        target = sample['mask']

        infer_start = time.time()
        if eval_mode == 'whole':
            output = model(rgb, hag)
            if output.shape[2:] != tuple(target.size()[1:]):
                output = F.interpolate(
                    output, size=target.size()[1:],
                    mode='bilinear', align_corners=False)
        else:
            # Sliding window inference
            output = slide_inference(
                model, rgb, hag, num_classes,
                cfg.evaluation.slide_size, cfg.evaluation.slide_stride)
        inference_time += time.time() - infer_start
        inference_samples += target.shape[0]

        # Convert to prediction and update confusion matrix for all samples in batch
        batch_size = target.shape[0]
        for b in range(batch_size):
            gt = target[b].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes

            pred = cv2.resize(
                output[b, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                target.size()[1:][::-1],
                interpolation=cv2.INTER_CUBIC
            ).argmax(axis=2).astype(np.uint8)

            mask = np.ones_like(gt[gt_idx]) == 1
            k = (gt[gt_idx] >= 0) & (pred[gt_idx] < num_classes) & mask
            conf_mat += np.bincount(
                num_classes * gt[gt_idx][k].astype(int) + pred[gt_idx][k],
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

        if (i + 1) % 100 == 0:
            logger.info(f'Validated {i + 1}/{len(val_loader)} samples')

    elapsed = time.time() - start_time
    logger.info(f'Validation completed in {elapsed:.2f}s')

    # Compute metrics
    if conf_mat.sum() == 0:
        return {
            'miou': 0.0,
            'static_miou': 0.0,
            'dynamic_miou': 0.0,
            'pixel_acc': 0.0,
            'mean_acc': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1': 0.0,
            'latency_ms': 0.0,
            'fps': 0.0,
            'class_iou': np.zeros(num_classes),
            'class_acc': np.zeros(num_classes),
            'class_precision': np.zeros(num_classes),
            'class_recall': np.zeros(num_classes),
            'class_f1': np.zeros(num_classes),
            'class_support': np.zeros(num_classes, dtype=np.int64),
        }

    with np.errstate(divide='ignore', invalid='ignore'):
        # Per-class IoU
        tp = np.diag(conf_mat)
        fp = conf_mat.sum(axis=0) - tp
        fn = conf_mat.sum(axis=1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / union, 0)

        # Pixel accuracy
        pixel_acc = tp.sum() / conf_mat.sum() * 100.0

        # Mean accuracy
        class_total = conf_mat.sum(axis=1)
        class_acc = np.where(class_total > 0, tp / class_total, 0)
        valid_mask = class_total > 0
        mean_acc = np.nanmean(class_acc[valid_mask]) * 100.0

        # mIoU
        miou = np.nanmean(iou[valid_mask]) * 100.0

        # Per-class precision/recall/F1
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0
        )
        support = class_total.astype(np.int64)
        mean_precision = np.nanmean(precision[valid_mask]) * 100.0
        mean_recall = np.nanmean(recall[valid_mask]) * 100.0
        mean_f1 = np.nanmean(f1[valid_mask]) * 100.0

        # Static mIoU (classes 0-16)
        static_mask = np.zeros(num_classes, dtype=bool)
        static_mask[:17] = True
        static_valid = valid_mask & static_mask
        static_miou = np.nanmean(iou[static_valid]) * 100.0 if static_valid.any() else 0.0

        # Dynamic mIoU (classes 17-18: sedan, truck)
        dynamic_mask = np.zeros(num_classes, dtype=bool)
        if num_classes > 17:
            dynamic_mask[17:] = True
        dynamic_valid = valid_mask & dynamic_mask
        dynamic_miou = np.nanmean(iou[dynamic_valid]) * 100.0 if dynamic_valid.any() else 0.0

    latency = inference_time / max(inference_samples, 1)
    fps = 1.0 / latency if latency > 0 else 0.0

    # Print results
    logger.info(
        f"\nmIoU: {miou:.2f}% | Static: {static_miou:.2f}% | "
        f"Dynamic: {dynamic_miou:.2f}% | Acc: {pixel_acc:.2f}%"
    )
    logger.info("\n" + "=" * 100)
    logger.info(
        f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} "
        f"{'F1':>8} {'Acc':>8} {'Support':>12}"
    )
    logger.info("-" * 100)
    for cls_idx, cls_name in enumerate(CLASS_NAMES[:num_classes]):
        if support[cls_idx] <= 0:
            continue
        cls_type = "[D]" if cls_idx >= 17 else "[S]"
        logger.info(
            f"  {cls_type} {cls_name:<15} "
            f"{iou[cls_idx] * 100:>7.2f}% "
            f"{precision[cls_idx] * 100:>7.2f}% "
            f"{recall[cls_idx] * 100:>7.2f}% "
            f"{f1[cls_idx] * 100:>7.2f}% "
            f"{class_acc[cls_idx] * 100:>7.2f}% "
            f"{support[cls_idx]:>11,}"
        )
    logger.info("-" * 100)
    logger.info(f"  {'mIoU':<18} {miou:>7.2f}%")
    logger.info(f"  {'Static mIoU':<18} {static_miou:>7.2f}%")
    logger.info(f"  {'Dynamic mIoU':<18} {dynamic_miou:>7.2f}%")
    logger.info(f"  {'Pixel Accuracy':<18} {pixel_acc:>7.2f}%")
    logger.info(f"  {'Mean Accuracy':<18} {mean_acc:>7.2f}%")
    logger.info(f"  {'Mean Precision':<18} {mean_precision:>7.2f}%")
    logger.info(f"  {'Mean Recall':<18} {mean_recall:>7.2f}%")
    logger.info(f"  {'Mean F1':<18} {mean_f1:>7.2f}%")
    logger.info("=" * 100)
    logger.info("\nInference speed:")
    logger.info(f"  Average time per image: {latency * 1000:.1f}ms")
    logger.info(f"  FPS: {fps:.2f}")

    metrics = {
        'miou': miou,
        'static_miou': static_miou,
        'dynamic_miou': dynamic_miou,
        'pixel_acc': pixel_acc,
        'mean_acc': mean_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'latency_ms': latency * 1000.0,
        'fps': fps,
        'class_iou': iou,
        'class_acc': class_acc,
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'class_support': support,
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)

    # Load config
    cfg = load_config(args.config)

    # Setup logging
    if is_main:
        os.makedirs(cfg.logging.log_dir, exist_ok=True)
        os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    logger = setup_logger(cfg.logging.log_dir, is_main=is_main)

    logger.info(f"Config: {args.config}")
    logger.info(f"Backbone: {cfg.model.backbone}")
    logger.info(f"Image size: {cfg.training.image_size}")
    logger.info(f"Global batch size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    logger.info(f"LR: {cfg.optimizer.lr}")
    logger.info(f"AMP: {cfg.training.amp} (dtype={getattr(cfg.training, 'amp_dtype', 'bf16')})")

    # Set seed (rank-shifted for DDP workers)
    set_seed(cfg.training.seed + rank)

    logger.info(
        f"Using device: {device} | distributed={distributed} rank={rank} "
        f"world_size={world_size} local_rank={local_rank}"
    )

    # Build model
    model = build_model(cfg, device)
    log_model_complexity(model, cfg, logger, device, is_main=is_main)
    raw_model = model

    # Build dataloaders
    train_loader, val_loader, train_sampler, per_gpu_batch_size = build_dataloaders(
        cfg, distributed=distributed, rank=rank, world_size=world_size)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Per-GPU batch size: {per_gpu_batch_size}")

    # Calculate total iterations for PolyLR
    iters_per_epoch = len(train_loader)
    total_iters = cfg.training.epochs * iters_per_epoch
    warmup_iters = cfg.scheduler.warmup_epochs * iters_per_epoch

    logger.info(f"Iters/epoch: {iters_per_epoch}")
    logger.info(f"Total iters: {total_iters}")
    logger.info(f"Warmup iters: {warmup_iters}")
    logger.info(
        "[LR] epochs=%d warmup_epochs=%d warmup_iters=%d power=%.3f warmup_ratio=%.3f warmup=%s",
        cfg.training.epochs,
        cfg.scheduler.warmup_epochs,
        warmup_iters,
        cfg.scheduler.power,
        cfg.scheduler.warmup_ratio,
        cfg.scheduler.warmup_type,
    )

    # Build optimizer with fair benchmark param groups
    lr = cfg.optimizer.lr

    optimizer = PolyWarmupAdamW(
        params=get_fair_param_groups(raw_model, lr=lr, weight_decay=cfg.optimizer.weight_decay),
        lr=lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=warmup_iters,
        max_iter=total_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_label).to(device)

    # AMP scaler
    amp_dtype = get_amp_dtype(cfg)
    scaler = GradScaler('cuda', enabled=cfg.training.amp and amp_dtype == torch.float16)

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        info = load_checkpoint(args.resume, raw_model, optimizer, device)
        start_epoch = info['epoch']
        best_miou = info.get('best_miou', 0)
        # Keep LR schedule continuous on resume (avoid warmup restart).
        if iters_per_epoch > 0:
            resumed_global_step = min(start_epoch * iters_per_epoch, max(0, total_iters - 1))
            optimizer.global_step = max(0, resumed_global_step)
            logger.info(
                f"[LR-Resume] epoch={start_epoch} -> global_step={optimizer.global_step} "
                f"(iters/epoch={iters_per_epoch})"
            )
        logger.info(f"Resumed at epoch {start_epoch}, best mIoU: {best_miou:.2f}%")

    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # Evaluation only
    if args.eval_only:
        if is_main:
            _ = validate(raw_model, val_loader, cfg, device, logger)
        if distributed:
            dist.barrier()
            cleanup_distributed(distributed)
        return

    # Training loop
    logger.info("Starting training...")
    training_start = time.time()

    for epoch in range(start_epoch, cfg.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg,
            epoch + 1, device, scaler, logger,
            distributed=distributed, world_size=world_size, is_main=is_main
        )

        logger.info(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")

        epoch_ckpt_path = os.path.join(
            cfg.logging.checkpoint_dir,
            f'checkpoint_epoch_{epoch + 1}.pth'
        )
        if is_main:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                },
                filename=epoch_ckpt_path
            )
            materialize_epoch_checkpoint(epoch_ckpt_path, 'hrfuser', epoch + 1)

        val_this_epoch = (epoch + 1) % cfg.training.val_every == 0 or epoch == cfg.training.epochs - 1

        # Validate
        if val_this_epoch:
            if is_main:
                metrics = validate(raw_model, val_loader, cfg, device, logger)
                current_miou = metrics['miou']

                # Save best model
                is_best = current_miou > best_miou
                if is_best:
                    best_miou = current_miou
                    logger.info(f"New best mIoU: {best_miou:.2f}%")

                if is_best:
                    checkpoint = torch.load(epoch_ckpt_path, map_location='cpu', weights_only=False)
                    checkpoint['metrics'] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in metrics.items()
                    }
                    checkpoint['best_miou'] = best_miou
                    torch.save(checkpoint, os.path.join(cfg.logging.checkpoint_dir, 'best.pth'))
                    promote_best_checkpoint(epoch_ckpt_path, 'hrfuser', epoch + 1)

            if distributed:
                dist.barrier()
    total_time = (time.time() - training_start) / 60.0
    logger.info(f"\nTraining completed in {total_time:.1f} minutes!")
    if is_main:
        maybe_sync_checkpoint_dir(cfg.logging.checkpoint_dir, logger.info)
    logger.info(f"Best mIoU: {best_miou:.2f}%")
    cleanup_distributed(distributed)


if __name__ == '__main__':
    main()
