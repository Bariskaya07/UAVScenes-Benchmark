#!/usr/bin/env python3
"""
CMNeXt Multi-Modal Validation Script for UAVScenes

Usage:
    python tools/val_mm.py --cfg configs/uavscenes_rgb_hag.yaml --checkpoint output/UAVScenes_CMNeXt_B2/best.pth

Features:
    - Sliding window inference
    - Per-class IoU reporting
    - Static vs Dynamic mIoU
    - Visualization output
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semseg.datasets import UAVScenes
from semseg.models import CMNeXt
from semseg.augment import get_test_transform
from semseg.metrics import UAVScenesMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='CMNeXt Validation')
    parser.add_argument('--cfg', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--save-vis', action='store_true', help='Save visualization')
    parser.add_argument('--vis-dir', type=str, default='output/vis', help='Visualization output dir')
    return parser.parse_args()


def load_config(cfg_path):
    """Load YAML config file."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def create_dataloader(cfg):
    """Create test DataLoader."""
    dataset_cfg = cfg['DATASET']
    transform = get_test_transform(cfg)

    dataset = UAVScenes(
        root=dataset_cfg['ROOT'],
        split='test',
        transform=transform,
        modals=dataset_cfg['MODALS']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['TRAIN'].get('WORKERS', 8),
        pin_memory=True
    )

    return dataloader, dataset


def create_model(cfg, checkpoint_path, device):
    """Create and load CMNeXt model."""
    model_cfg = cfg['MODEL']

    model = CMNeXt(
        backbone=model_cfg.get('BACKBONE', 'mit_b2'),
        num_classes=model_cfg.get('NUM_CLASSES', 19),
        modals=cfg['DATASET']['MODALS'],
        pretrained=None  # Will load from checkpoint
    )

    # Load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}, "
                  f"mIoU: {checkpoint.get('miou', 0)*100:.2f}%")
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def sliding_window_inference(model, inputs, cfg):
    """Sliding window inference."""
    eval_cfg = cfg.get('EVAL', {})
    crop_size = eval_cfg.get('CROP_SIZE', [1024, 1024])
    stride = eval_cfg.get('STRIDE', [768, 768])
    num_classes = cfg['MODEL']['NUM_CLASSES']

    if isinstance(inputs, (list, tuple)):
        B, C, H, W = inputs[0].shape
        device = inputs[0].device
    else:
        B, C, H, W = inputs.shape
        device = inputs.device

    crop_h, crop_w = crop_size
    stride_h, stride_w = stride

    count_mat = torch.zeros((B, 1, H, W), device=device)
    logits_sum = torch.zeros((B, num_classes, H, W), device=device)

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

            if isinstance(inputs, (list, tuple)):
                crop_inputs = [x[:, :, h_start:h_end, w_start:w_end] for x in inputs]
            else:
                crop_inputs = inputs[:, :, h_start:h_end, w_start:w_end]

            with autocast():
                crop_logits = model(crop_inputs)

            if crop_logits.shape[2:] != (h_end - h_start, w_end - w_start):
                crop_logits = F.interpolate(
                    crop_logits,
                    size=(h_end - h_start, w_end - w_start),
                    mode='bilinear',
                    align_corners=False
                )

            logits_sum[:, :, h_start:h_end, w_start:w_end] += crop_logits
            count_mat[:, :, h_start:h_end, w_start:w_end] += 1

            if w_end == W:
                break
            w_start += stride_w

        if h_end == H:
            break
        h_start += stride_h

    return logits_sum / count_mat


def save_visualization(rgb, pred, target, save_path, dataset):
    """Save visualization of prediction vs ground truth."""
    # Denormalize RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_vis = rgb.cpu().numpy().transpose(1, 2, 0)
    rgb_vis = (rgb_vis * std + mean) * 255
    rgb_vis = rgb_vis.clip(0, 255).astype(np.uint8)

    # Decode segmentation maps
    pred_vis = dataset.decode_segmap(pred.cpu().numpy())
    target_vis = dataset.decode_segmap(target.cpu().numpy())

    # Concatenate
    vis = np.concatenate([rgb_vis, pred_vis, target_vis], axis=1)

    # Save
    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate(model, dataloader, dataset, device, cfg, save_vis=False, vis_dir=None):
    """Evaluate model."""
    metrics = UAVScenesMetrics(num_classes=cfg['MODEL']['NUM_CLASSES'], ignore_label=255)
    eval_mode = cfg.get('EVAL', {}).get('MODE', 'slide')

    if save_vis and vis_dir:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0
    num_samples = 0

    for batch_idx, (inputs, target) in enumerate(dataloader):
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
        target = target.to(device)

        start_time = time.time()

        if eval_mode == 'slide':
            logits = sliding_window_inference(model, inputs, cfg)
        else:
            with autocast():
                logits = model(inputs)

        elapsed = time.time() - start_time
        total_time += elapsed
        num_samples += target.shape[0]

        pred = logits.argmax(dim=1)
        metrics.update_batch(pred, target)

        # Save visualization
        if save_vis and vis_dir and batch_idx < 20:  # Save first 20
            for i in range(target.shape[0]):
                rgb = inputs[0][i] if isinstance(inputs, (list, tuple)) else inputs[i]
                save_path = vis_dir / f'sample_{batch_idx}_{i}.png'
                save_visualization(rgb, pred[i], target[i], save_path, dataset)

        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches...")

    avg_time = total_time / num_samples
    fps = 1.0 / avg_time

    return metrics, avg_time, fps


def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    device = torch.device(cfg.get('DEVICE', 'cuda'))
    print(f"Device: {device}")

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        save_dir = Path(cfg['SAVE_DIR'])
        best_path = save_dir / 'best.pth'
        if best_path.exists():
            checkpoint_path = str(best_path)
        else:
            raise ValueError("No checkpoint specified and no best.pth found")

    # Create dataloader
    print("\nCreating dataloader...")
    dataloader, dataset = create_dataloader(cfg)
    print(f"Test set: {len(dataset)} samples")

    # Create model
    print("\nLoading model...")
    model = create_model(cfg, checkpoint_path, device)

    # Evaluate
    print("\nEvaluating...")
    print(f"Mode: {cfg.get('EVAL', {}).get('MODE', 'slide')}")

    metrics, avg_time, fps = evaluate(
        model, dataloader, dataset, device, cfg,
        save_vis=args.save_vis, vis_dir=args.vis_dir
    )

    # Print results
    print("\n")
    metrics.print_results()

    print(f"\nInference speed:")
    print(f"  Average time per image: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.2f}")

    if args.save_vis:
        print(f"\nVisualizations saved to: {args.vis_dir}")


if __name__ == '__main__':
    main()
