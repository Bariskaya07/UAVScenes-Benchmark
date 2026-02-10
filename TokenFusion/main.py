"""
TokenFusion Training Script for UAVScenes RGB+HAG Segmentation

Usage:
    python main.py --config configs/uavscenes_rgb_hag.yaml

Features:
    - MiT-B2 backbone with TokenFusion
    - PolyWarmupAdamW optimizer
    - Mixed precision training (AMP)
    - Sliding window validation
    - Fair comparison settings matching CMNeXt/DFormerv2
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Local imports
from models import WeTr
from datasets import UAVScenesDataset
from utils.transforms import TrainTransform, ValTransform
from utils.optimizer import PolyWarmupAdamW
from utils.metrics import ConfusionMatrix, print_metrics, UAVSCENES_CLASSES
from utils.helpers import (
    setup_logger, save_checkpoint, load_checkpoint,
    AverageMeter, set_seed, count_parameters, sliding_window_inference
)


class Config:
    """Configuration class to hold YAML config as attributes."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='TokenFusion UAVScenes Training')
    parser.add_argument('--config', type=str, default='configs/uavscenes_rgb_hag.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def build_dataloaders(cfg):
    """Build training and validation dataloaders."""
    # Training transform
    train_transform = TrainTransform(
        crop_size=cfg.training.crop_size,
        scale_range=tuple(cfg.augmentation.scale_range),
        flip_prob=cfg.augmentation.flip_prob,
        rgb_mean=cfg.normalization.rgb_mean,
        rgb_std=cfg.normalization.rgb_std,
        hag_mean=cfg.normalization.hag_mean,
        hag_std=cfg.normalization.hag_std,
        ignore_label=cfg.dataset.ignore_label
    )

    # Validation transform
    val_transform = ValTransform(
        rgb_mean=cfg.normalization.rgb_mean,
        rgb_std=cfg.normalization.rgb_std,
        hag_mean=cfg.normalization.hag_mean,
        hag_std=cfg.normalization.hag_std
    )

    # Create datasets
    train_dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path,
        split='train',
        transform=train_transform,
        hag_max_height=cfg.hag.max_meters
    )

    val_dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path,
        split='test',
        transform=val_transform,
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
        batch_size=1,  # Use batch size 1 for sliding window inference
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def build_model(cfg, device):
    """Build WeTr model."""
    pretrained = cfg.model.pretrained if os.path.exists(cfg.model.pretrained) else None

    model = WeTr(
        backbone=cfg.model.backbone,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        pretrained=pretrained
    )

    model = model.to(device)
    return model


def compute_loss(outputs, target, masks, lamda, ignore_label=255):
    """
    Compute TokenFusion loss.

    Loss = NLLLoss(output, target) + lamda * L1_loss(masks)

    Args:
        outputs: List of [rgb_pred, hag_pred, ensemble_pred]
        target: Ground truth labels [B, H, W]
        masks: List of token importance masks
        lamda: L1 sparsity weight
        ignore_label: Label to ignore

    Returns:
        Total loss, segmentation loss, L1 loss
    """
    total_loss = 0
    seg_loss = 0

    # Segmentation loss for each modality output
    for output in outputs[:2]:  # RGB and HAG predictions
        # Upsample to target size
        output = F.interpolate(
            output, size=target.shape[1:],
            mode='bilinear', align_corners=False
        )

        # NLL loss
        soft_output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(soft_output, target, ignore_index=ignore_label)
        total_loss += loss
        seg_loss += loss.item()

    # L1 sparsity loss on masks
    l1_loss = 0
    if lamda > 0 and masks:
        for mask in masks:
            for m in mask:
                l1_loss += torch.abs(m).sum()
        total_loss += lamda * l1_loss
        l1_loss = l1_loss.item()

    return total_loss, seg_loss / 2, l1_loss


def train_one_epoch(model, train_loader, optimizer, cfg, epoch, device, scaler, logger):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    l1_loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    num_iters = len(train_loader)

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Get data
        rgb = sample['rgb'].to(device)
        hag = sample['hag'].to(device)
        label = sample['label'].to(device)

        # Forward pass with AMP
        with autocast(enabled=cfg.training.amp):
            outputs, masks = model([rgb, hag])
            loss, seg_loss, l1_loss = compute_loss(
                outputs, label, masks,
                lamda=cfg.loss.lamda,
                ignore_label=cfg.dataset.ignore_label
            )

        # Backward pass
        optimizer.zero_grad()
        if cfg.training.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer.optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.optimizer.step()

        # Update LR
        optimizer.scheduler.step()
        optimizer.current_iter += 1

        # Update meters
        loss_meter.update(loss.item(), rgb.size(0))
        seg_loss_meter.update(seg_loss, rgb.size(0))
        l1_loss_meter.update(l1_loss, rgb.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if (i + 1) % cfg.logging.print_freq == 0:
            current_lr = optimizer.get_lr()
            logger.info(
                f'Epoch [{epoch}][{i + 1}/{num_iters}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'Seg: {seg_loss_meter.avg:.4f} '
                f'L1: {l1_loss_meter.avg:.6f} '
                f'LR: {current_lr:.6f} '
                f'Time: {batch_time.avg:.3f}s'
            )

    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, cfg, device, logger):
    """Validate model with sliding window inference."""
    model.eval()

    confusion_matrix = ConfusionMatrix(
        num_classes=cfg.dataset.num_classes,
        ignore_label=cfg.dataset.ignore_label
    )

    logger.info("Running validation with sliding window inference...")
    start_time = time.time()

    for i, sample in enumerate(val_loader):
        rgb = sample['rgb'].to(device)
        hag = sample['hag'].to(device)
        label = sample['label']

        # Sliding window inference
        output = sliding_window_inference(
            model, rgb, hag,
            window_size=cfg.evaluation.slide_size,
            stride=cfg.evaluation.slide_stride,
            num_classes=cfg.dataset.num_classes,
            device=device
        )

        # Get predictions
        pred = output.argmax(dim=1).cpu()

        # Update confusion matrix
        confusion_matrix.update(pred, label)

        if (i + 1) % 100 == 0:
            logger.info(f'Validated {i + 1}/{len(val_loader)} samples')

    # Compute metrics
    metrics = confusion_matrix.get_metrics()

    elapsed = time.time() - start_time
    logger.info(f'Validation completed in {elapsed:.2f}s')
    print_metrics(metrics, UAVSCENES_CLASSES)

    return metrics


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

    # Calculate max iterations
    iters_per_epoch = len(train_loader)
    max_iter = cfg.training.epochs * iters_per_epoch

    # Build optimizer
    optimizer = PolyWarmupAdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_iter=cfg.optimizer.warmup_iter,
        max_iter=max_iter,
        power=cfg.optimizer.power
    )

    # AMP scaler
    scaler = GradScaler(enabled=cfg.training.amp)

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        info = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = info['epoch']
        best_miou = info.get('best_miou', 0)

    # Evaluation only
    if args.eval_only:
        metrics = validate(model, val_loader, cfg, device, logger)
        return

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        logger.info(f"{'=' * 60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg,
            epoch + 1, device, scaler, logger
        )

        # Validate
        if (epoch + 1) % cfg.training.val_every == 0 or epoch == cfg.training.epochs - 1:
            metrics = validate(model, val_loader, cfg, device, logger)
            current_miou = metrics['miou']

            # Save best model
            is_best = current_miou > best_miou
            if is_best:
                best_miou = current_miou
                logger.info(f"New best mIoU: {best_miou * 100:.2f}%")

            # Save checkpoint
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'metrics': metrics
                },
                filename=os.path.join(cfg.logging.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'),
                is_best=is_best,
                best_filename=os.path.join(cfg.logging.checkpoint_dir, 'best.pth')
            )

        # Regular checkpoint save
        if (epoch + 1) % cfg.training.save_every == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou
                },
                filename=os.path.join(cfg.logging.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            )

    logger.info(f"\nTraining completed!")
    logger.info(f"Best mIoU: {best_miou * 100:.2f}%")


if __name__ == '__main__':
    main()
