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
from datetime import datetime

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from models.hrfuser_segformer import HRFuserSegFormer
from datasets.uavscenes import UAVScenesDataset, CLASS_NAMES
from utils.optimizer import PolyWarmupAdamW, get_fair_param_groups
from utils.augmentations_mm import (
    RandomColorJitter, RandomHorizontalFlip, RandomGaussianBlur,
    RandomResizedCrop, Normalize, Resize
)


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
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir=None, name='hrfuser'):
    """Setup logger for training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename='checkpoint.pth', is_best=False, best_filename='best.pth'):
    """Save training checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """Load checkpoint."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

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

def build_dataloaders(cfg):
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


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
                    epoch, device, scaler, logger):
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

    end = time.time()
    num_iters = len(train_loader)

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        rgb = sample['rgb'].to(device).float()
        hag = sample['depth'].to(device).float()  # HAG stored as 'depth' key
        target = sample['mask'].to(device).long()

        # Forward pass with AMP
        with autocast('cuda', enabled=cfg.training.amp):
            output = model(rgb, hag)

            # Upsample to target size
            output = F.interpolate(output, size=target.shape[1:],
                                   mode='bilinear', align_corners=False)

            loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        if cfg.training.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update meters
        loss_meter.update(loss.item(), rgb.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if (i + 1) % cfg.logging.print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{i + 1}/{num_iters}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'LR: {current_lr:.6f} '
                f'Data: {data_time.avg:.3f}s '
                f'Batch: {batch_time.avg:.3f}s'
            )

    return loss_meter.avg


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

    for i, sample in enumerate(val_loader):
        rgb = sample['rgb'].to(device).float()
        hag = sample['depth'].to(device).float()
        target = sample['mask']
        gt = target[0].data.cpu().numpy().astype(np.uint8)
        gt_idx = gt < num_classes

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

        # Convert to prediction
        pred = cv2.resize(
            output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
            target.size()[1:][::-1],
            interpolation=cv2.INTER_CUBIC
        ).argmax(axis=2).astype(np.uint8)

        # Update confusion matrix
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
        return {'miou': 0, 'pixel_acc': 0, 'mean_acc': 0, 'class_iou': np.zeros(num_classes)}

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

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"  mIoU:         {miou:.2f}%")
    logger.info(f"  Static mIoU:  {static_miou:.2f}%")
    logger.info(f"  Dynamic mIoU: {dynamic_miou:.2f}%")
    logger.info(f"  Pixel Acc:    {pixel_acc:.2f}%")
    logger.info(f"  Mean Acc:     {mean_acc:.2f}%")

    logger.info(f"\nPer-class IoU:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        marker = " (dynamic)" if cls_idx >= 17 else ""
        logger.info(f"  {cls_idx:2d}. {cls_name:20s}: {iou[cls_idx]*100:5.2f}%{marker}")
    logger.info(f"{'='*60}")

    metrics = {
        'miou': miou,
        'static_miou': static_miou,
        'dynamic_miou': dynamic_miou,
        'pixel_acc': pixel_acc,
        'mean_acc': mean_acc,
        'class_iou': iou,
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup logging
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    logger = setup_logger(cfg.logging.log_dir)

    logger.info(f"Config: {args.config}")
    logger.info(f"Backbone: {cfg.model.backbone}")
    logger.info(f"Image size: {cfg.training.image_size}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    logger.info(f"LR: {cfg.optimizer.lr}")
    logger.info(f"AMP: {cfg.training.amp}")

    # Set seed
    set_seed(cfg.training.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(cfg, device)
    logger.info(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Calculate total iterations for PolyLR
    iters_per_epoch = len(train_loader)
    total_iters = cfg.training.epochs * iters_per_epoch
    warmup_iters = cfg.scheduler.warmup_epochs * iters_per_epoch

    logger.info(f"Iters/epoch: {iters_per_epoch}")
    logger.info(f"Total iters: {total_iters}")
    logger.info(f"Warmup iters: {warmup_iters}")

    # Build optimizer with fair benchmark param groups
    lr = cfg.optimizer.lr

    optimizer = PolyWarmupAdamW(
        params=get_fair_param_groups(model, lr=lr, weight_decay=cfg.optimizer.weight_decay),
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
    scaler = GradScaler('cuda', enabled=cfg.training.amp)

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        info = load_checkpoint(args.resume, model, optimizer, device)
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

    # Evaluation only
    if args.eval_only:
        metrics = validate(model, val_loader, cfg, device, logger)
        return

    # Training loop
    logger.info("Starting training...")
    training_start = time.time()

    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg,
            epoch + 1, device, scaler, logger
        )

        logger.info(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")

        # Validate
        if (epoch + 1) % cfg.training.val_every == 0 or epoch == cfg.training.epochs - 1:
            metrics = validate(model, val_loader, cfg, device, logger)
            current_miou = metrics['miou']

            # Save best model
            is_best = current_miou > best_miou
            if is_best:
                best_miou = current_miou
                logger.info(f"New best mIoU: {best_miou:.2f}%")

            # Save checkpoint
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in metrics.items()},
                },
                filename=os.path.join(cfg.logging.checkpoint_dir,
                                      f'checkpoint_epoch_{epoch + 1}.pth'),
                is_best=is_best,
                best_filename=os.path.join(cfg.logging.checkpoint_dir, 'best.pth')
            )

        # Regular checkpoint save
        elif (epoch + 1) % cfg.training.save_every == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                },
                filename=os.path.join(cfg.logging.checkpoint_dir,
                                      f'checkpoint_epoch_{epoch + 1}.pth')
            )

    total_time = (time.time() - training_start) / 60.0
    logger.info(f"\nTraining completed in {total_time:.1f} minutes!")
    logger.info(f"Best mIoU: {best_miou:.2f}%")


if __name__ == '__main__':
    main()
