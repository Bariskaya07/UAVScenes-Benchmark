#!/usr/bin/env python3
"""
CMNeXt Multi-Modal Training Script for UAVScenes

Usage:
    python tools/train_mm.py --cfg configs/uavscenes_rgb_hag.yaml

Features:
    - RGB + HAG multi-modal training
    - Mixed precision (AMP)
    - Early stopping
    - Checkpoint saving
    - TensorBoard logging
"""

import os
import sys
import argparse
import yaml
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semseg.datasets import UAVScenes
from semseg.models import CMNeXt, cmnext_b2
from semseg.augment import get_train_transform, get_test_transform
from semseg.losses import get_loss
from semseg.optimizers import get_optimizer
from semseg.scheduler import get_scheduler
from semseg.metrics import UAVScenesMetrics, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='CMNeXt Training')
    parser.add_argument('--cfg', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()


def load_config(cfg_path):
    """Load YAML config file."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(cfg, split='train'):
    """Create DataLoader for training or testing."""
    dataset_cfg = cfg['DATASET']
    train_cfg = cfg['TRAIN']

    # Get transforms
    if split == 'train':
        transform = get_train_transform(cfg)
    else:
        transform = get_test_transform(cfg)

    # Create dataset
    dataset = UAVScenes(
        root=dataset_cfg['ROOT'],
        split=split,
        transform=transform,
        modals=dataset_cfg['MODALS']
    )

    # Create dataloader
    batch_size = train_cfg['BATCH_SIZE'] if split == 'train' else cfg['EVAL']['BATCH_SIZE']
    shuffle = (split == 'train')
    num_workers = train_cfg.get('WORKERS', 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


def create_model(cfg):
    """Create CMNeXt model."""
    model_cfg = cfg['MODEL']

    backbone = model_cfg.get('BACKBONE', 'mit_b2')
    num_classes = model_cfg.get('NUM_CLASSES', 19)
    pretrained = model_cfg.get('PRETRAINED', None)
    modals = cfg['DATASET']['MODALS']

    model = CMNeXt(
        backbone=backbone,
        num_classes=num_classes,
        modals=modals,
        pretrained=pretrained
    )

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch, cfg, writer):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    log_interval = cfg.get('LOGGING', {}).get('LOG_INTERVAL', 50)
    accum_steps = cfg['TRAIN'].get('GRAD_ACCUM', 1)  # Gradient accumulation steps

    start_time = time.time()
    optimizer.zero_grad()  # Zero gradients at start

    for batch_idx, (inputs, target) in enumerate(dataloader):
        # Move to device
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
        target = target.to(device)

        # Forward pass with AMP
        if cfg['TRAIN'].get('AMP', True):
            with autocast():
                logits = model(inputs)
                loss = criterion(logits, target)
                loss = loss / accum_steps  # Scale loss for accumulation

            # NaN/Inf loss kontrolü - bu batch'i atla
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                del logits, loss  # Tensor'ları sil
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # GPU memory'yi temizle
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Step optimizer every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping for numerical stability
                scaler.unscale_(optimizer)
                
                # NaN gradient kontrolü
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping step...")
                    optimizer.zero_grad()
                    scaler.update()
                    torch.cuda.empty_cache()  # GPU memory'yi temizle
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # Update scheduler per effective batch
        else:
            logits = model(inputs)
            loss = criterion(logits, target)
            loss = loss / accum_steps
            
            # NaN/Inf loss kontrolü
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                del logits, loss
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
                
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping for numerical stability
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping step...")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += loss.item() * accum_steps  # Unscale for logging

        # Logging
        if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)

            print(f"Epoch [{epoch}] [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} "
                  f"LR: {current_lr:.2e} "
                  f"ETA: {eta/60:.1f}min")

            # TensorBoard
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR', current_lr, global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, cfg, num_classes=19, eval_mode=None, config_key='EVAL'):
    """Evaluate model on test set."""
    model.eval()
    metrics = UAVScenesMetrics(num_classes=num_classes, ignore_label=255)

    # Use provided eval_mode or read from config
    if eval_mode is None:
        eval_mode = cfg.get(config_key, {}).get('MODE', 'whole')
    print(f"\n[DEBUG] eval_mode = '{eval_mode}' (from cfg['{config_key}']['MODE'])")
    print(f"[DEBUG] cfg['{config_key}'] = {cfg.get(config_key, {})}")
    total_samples = len(dataloader)
    eval_start_time = time.time()

    for idx, (inputs, target) in enumerate(dataloader):
        # Progress bar
        if (idx + 1) % 50 == 0 or idx == 0 or (idx + 1) == total_samples:
            elapsed = time.time() - eval_start_time
            eta = elapsed / (idx + 1) * (total_samples - idx - 1) if idx > 0 else 0
            print(f"\rEvaluating: {idx + 1}/{total_samples} ({100*(idx+1)/total_samples:.1f}%) ETA: {eta/60:.1f}min", end="", flush=True)
        # Move to device
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
        target = target.to(device)

        if eval_mode == 'slide':
            # Sliding window inference
            logits = sliding_window_inference(model, inputs, cfg)
        else:
            # Whole image inference (resize to training size for speed)
            orig_size = target.shape[-2:]  # (H, W)
            eval_size = cfg.get('EVAL', {}).get('IMAGE_SIZE', [768, 768])

            # Resize inputs to eval size
            if isinstance(inputs, (list, tuple)):
                inputs_resized = [F.interpolate(x, size=eval_size, mode='bilinear', align_corners=False) for x in inputs]
            else:
                inputs_resized = F.interpolate(inputs, size=eval_size, mode='bilinear', align_corners=False)

            # Forward pass at reduced resolution
            logits = model(inputs_resized)

            # Upsample back to original size
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)

        # Get predictions
        pred = logits.argmax(dim=1)

        # Update metrics
        metrics.update_batch(pred, target)

    print()  # New line after progress bar
    results = metrics.get_results()
    return results, metrics


def sliding_window_inference(model, inputs, cfg):
    """Sliding window inference for full resolution evaluation.

    Args:
        model: Model
        inputs: List of input tensors [rgb, hag]
        cfg: Config dictionary

    Returns:
        Aggregated logits [B, C, H, W]
    """
    eval_cfg = cfg.get('EVAL', {})
    crop_size = eval_cfg.get('CROP_SIZE', [1024, 1024])
    stride = eval_cfg.get('STRIDE', [768, 768])
    num_classes = cfg['MODEL']['NUM_CLASSES']

    # Get input dimensions
    if isinstance(inputs, (list, tuple)):
        B, C, H, W = inputs[0].shape
    else:
        B, C, H, W = inputs.shape

    crop_h, crop_w = crop_size
    stride_h, stride_w = stride

    # Initialize output
    device = inputs[0].device if isinstance(inputs, (list, tuple)) else inputs.device
    count_mat = torch.zeros((B, 1, H, W), device=device)
    logits_sum = torch.zeros((B, num_classes, H, W), device=device)

    # Sliding window
    h_start = 0
    while h_start < H:
        h_end = min(h_start + crop_h, H)
        if h_end - h_start < crop_h and h_start > 0:
            h_start = H - crop_h
            h_end = H

        w_start = 0
        while w_start < W:
            w_end = min(w_start + crop_w, W)
            if w_end - w_start < crop_w and w_start > 0:
                w_start = W - crop_w
                w_end = W

            # Extract crop
            if isinstance(inputs, (list, tuple)):
                crop_inputs = [x[:, :, h_start:h_end, w_start:w_end] for x in inputs]
            else:
                crop_inputs = inputs[:, :, h_start:h_end, w_start:w_end]

            # Forward pass
            with autocast():
                crop_logits = model(crop_inputs)

            # Resize if needed (shouldn't be necessary if crop matches model input)
            if crop_logits.shape[2:] != (h_end - h_start, w_end - w_start):
                crop_logits = F.interpolate(
                    crop_logits,
                    size=(h_end - h_start, w_end - w_start),
                    mode='bilinear',
                    align_corners=False
                )

            # Accumulate
            logits_sum[:, :, h_start:h_end, w_start:w_end] += crop_logits
            count_mat[:, :, h_start:h_end, w_start:w_end] += 1

            # Move window
            if w_end == W:
                break
            w_start += stride_w

        if h_end == H:
            break
        h_start += stride_h

    # Average overlapping regions
    logits = logits_sum / count_mat

    return logits


def save_checkpoint(model, optimizer, scheduler, epoch, miou, cfg, is_best=False):
    """Save model checkpoint."""
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'miou': miou,
        'config': cfg
    }

    # Save latest
    torch.save(checkpoint, save_dir / 'latest.pth')

    # Save best
    if is_best:
        torch.save(checkpoint, save_dir / 'best.pth')
        print(f"Saved best model with mIoU: {miou*100:.2f}%")

    # Save periodic
    save_interval = cfg.get('LOGGING', {}).get('SAVE_INTERVAL', 10)
    if (epoch + 1) % save_interval == 0:
        torch.save(checkpoint, save_dir / f'epoch_{epoch+1}.pth')


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.cfg)

    # Set seed
    seed = args.seed or cfg.get('SEED', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Device
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    print(f"Device: {device}")

    # Create save directory
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    # TensorBoard
    log_dir = save_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(cfg, split='train')
    val_loader = create_dataloader(cfg, split='val')      # For periodic evaluation during training
    test_loader = create_dataloader(cfg, split='test')    # For final evaluation only
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Create model
    print("\nCreating model...")
    model = create_model(cfg)
    model = model.to(device)
    print(f"Model: {cfg['MODEL']['NAME']} with {cfg['MODEL']['BACKBONE']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Calculate FLOPs
    if FVCORE_AVAILABLE:
        try:
            num_modals = len(cfg['DATASET'].get('MODALS', ['rgb', 'hag']))
            dummy_input = [torch.zeros(1, 3, 768, 768).to(device) for _ in range(num_modals)]
            flops = FlopCountAnalysis(model, dummy_input)
            print(f"FLOPs: {flops.total() / 1e9:.2f}G")
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
    else:
        print("FLOPs: fvcore not installed (pip install fvcore)")

    # Create loss, optimizer, scheduler
    criterion = get_loss(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    # AMP scaler
    scaler = GradScaler() if cfg['TRAIN'].get('AMP', True) else None

    # Early stopping
    early_stop_cfg = cfg['TRAIN'].get('EARLY_STOP', {})
    early_stopper = None
    if early_stop_cfg.get('ENABLE', False):
        early_stopper = EarlyStopping(
            patience=early_stop_cfg.get('PATIENCE', 5),
            min_delta=early_stop_cfg.get('MIN_DELTA', 0.001),
            mode='max'
        )

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('miou', 0.0)
        
        # Override LR and scheduler params from config (for step decay / epoch change)
        new_lr = cfg['OPTIMIZER']['LR']
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # Scheduler'ın base_lrs'ini de güncelle (her step'te bunu kullanıyor)
        scheduler.base_lrs = [new_lr for _ in scheduler.base_lrs]
        
        # Recalculate max_iters for new epoch count
        iters_per_epoch = len(train_loader)
        grad_accum = cfg['TRAIN'].get('GRAD_ACCUM', 1)
        effective_iters = iters_per_epoch // grad_accum
        new_max_iters = cfg['TRAIN']['EPOCHS'] * effective_iters
        scheduler.max_iters = new_max_iters
        
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")
        print(f"LR overridden to: {new_lr:.2e}, max_iters updated to {new_max_iters}")
        print(f"Training will end at epoch {cfg['TRAIN']['EPOCHS']}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    train_cfg = cfg['TRAIN']
    epochs = train_cfg['EPOCHS']
    eval_start = train_cfg.get('EVAL_START', 5)
    eval_interval = train_cfg.get('EVAL_INTERVAL', 5)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, cfg, writer
        )
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)

        # Evaluate on VALIDATION set (not test set!)
        if epoch >= eval_start and (epoch - eval_start) % eval_interval == 0:
            print("\nEvaluating on validation set...")
            results, metrics = evaluate(model, val_loader, device, cfg,
                                        num_classes=cfg['MODEL']['NUM_CLASSES'])

            miou = results['mIoU']
            static_miou = results['static_mIoU']
            dynamic_miou = results['dynamic_mIoU']
            pixel_acc = results['pixel_accuracy']

            print(f"Val mIoU: {miou*100:.2f}% | Static: {static_miou*100:.2f}% | "
                  f"Dynamic: {dynamic_miou*100:.2f}% | Acc: {pixel_acc*100:.2f}%")

            # Per-class IoU (compact format)
            print("\nPer-class IoU:")
            per_class_iou = results['per_class_iou']
            for i, cls_name in enumerate(UAVScenes.CLASSES):
                marker = "[D]" if i >= 17 else "[S]"  # Dynamic: sedan(17), truck(18)
                print(f"  {marker} {cls_name:<18} {per_class_iou[i]*100:>6.2f}%")
            print()

            # TensorBoard (validation metrics)
            writer.add_scalar('Val/mIoU', miou, epoch)
            writer.add_scalar('Val/Static_mIoU', static_miou, epoch)
            writer.add_scalar('Val/Dynamic_mIoU', dynamic_miou, epoch)
            writer.add_scalar('Val/Pixel_Accuracy', pixel_acc, epoch)

            # Per-class IoU (validation)
            for i, iou in enumerate(results['per_class_iou']):
                writer.add_scalar(f'Val_Class_IoU/{UAVScenes.CLASSES[i]}', iou, epoch)

            # Save checkpoint
            is_best = miou > best_miou
            if is_best:
                best_miou = miou
            save_checkpoint(model, optimizer, scheduler, epoch, miou, cfg, is_best)

            # Early stopping
            if early_stopper is not None:
                if early_stopper(miou):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best mIoU: {best_miou*100:.2f}%")
                    break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Load best model and evaluate
    best_path = save_dir / 'best.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    print("\nFinal evaluation on test set:")
    test_mode = cfg.get('TEST', {}).get('MODE', 'slide')
    results, metrics = evaluate(model, test_loader, device, cfg,
                                num_classes=cfg['MODEL']['NUM_CLASSES'],
                                eval_mode=test_mode, config_key='TEST')
    metrics.print_results()

    writer.close()
    print(f"\nLogs saved to: {log_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == '__main__':
    main()
