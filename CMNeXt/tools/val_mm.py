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
import io
import json
import yaml
import time
from pathlib import Path
from contextlib import redirect_stdout

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
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override evaluation batch size')
    parser.add_argument('--save-vis', action='store_true', help='Save visualization')
    parser.add_argument('--vis-dir', type=str, default='output/vis', help='Visualization output dir')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save JSON/TXT evaluation summaries')
    return parser.parse_args()


def load_config(cfg_path):
    """Load YAML config file."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def save_results(metrics, output_dir, avg_time, fps, split, checkpoint_path, num_images):
    """Save evaluation summary as JSON and human-readable TXT."""
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = metrics.get_results()
    payload = _to_serializable(results)
    payload.update({
        'split': split,
        'checkpoint': checkpoint_path,
        'num_images': int(num_images),
        'avg_time_ms': float(avg_time * 1000.0),
        'fps': float(fps),
        'class_names': UAVScenesMetrics.CLASS_NAMES,
    })

    json_path = results_dir / 'CMNeXt_results.json'
    txt_path = results_dir / 'CMNeXt_results.txt'

    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        metrics.print_results()
    metrics_text = buffer.getvalue().strip()

    with open(txt_path, 'w') as f:
        f.write("CMNeXt UAVScenes Evaluation Results\n")
        f.write("=" * 100 + "\n")
        f.write(f"Split: {split}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Images: {num_images}\n\n")
        f.write(metrics_text + "\n\n")
        f.write("Inference speed:\n")
        f.write(f"  Average time per image: {avg_time * 1000.0:.1f}ms\n")
        f.write(f"  FPS: {fps:.2f}\n")

    print(f"\nSaved results to: {json_path}")
    print(f"Saved summary to: {txt_path}")


def create_dataloader(cfg, split='test', batch_size_override=None):
    """Create DataLoader for evaluation."""
    dataset_cfg = cfg['DATASET']
    transform = get_test_transform(cfg)

    dataset = UAVScenes(
        root=dataset_cfg['ROOT'],
        split=split,
        transform=transform,
        modals=dataset_cfg['MODALS'],
        aux_channels=dataset_cfg.get('AUX_CHANNELS', 3),
        hag_max_meters=dataset_cfg.get('HAG_MAX_METERS', 50.0),
    )

    if batch_size_override is not None:
        batch_size = batch_size_override
    elif split == 'test':
        batch_size = cfg.get('TEST', {}).get('BATCH_SIZE', cfg['EVAL']['BATCH_SIZE'])
    else:
        batch_size = cfg['EVAL']['BATCH_SIZE']

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
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
        pretrained=None,  # Will load from checkpoint
        embed_dim=model_cfg.get('EMBED_DIM', 256),
        aux_in_chans=model_cfg.get('AUX_IN_CHANS', 3),
        activation_checkpoint=model_cfg.get('ACTIVATION_CHECKPOINT', False),
    )

    # Load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    """Sliding window inference with fixed 768x768 crop and 512 stride for fair benchmarking."""
    eval_cfg = cfg.get('EVAL', {})
    crop_size = eval_cfg.get('CROP_SIZE', [768, 768])
    stride = eval_cfg.get('STRIDE', [512, 512])
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
def evaluate(model, dataloader, dataset, device, cfg, eval_mode='slide', save_vis=False, vis_dir=None):
    """Evaluate model."""
    metrics = UAVScenesMetrics(num_classes=cfg['MODEL']['NUM_CLASSES'], ignore_label=255)

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
        if num_samples % 100 == 0 or num_samples == len(dataset):
            print(f"Processed {num_samples}/{len(dataset)} images...")

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
    print(f"\nCreating dataloader for '{args.split}' split...")
    dataloader, dataset = create_dataloader(cfg, split=args.split, batch_size_override=args.batch_size)
    print(f"{args.split.capitalize()} set: {len(dataset)} samples")

    # Create model
    print("\nLoading model...")
    model = create_model(cfg, checkpoint_path, device)

    # Determine eval mode based on split
    if args.split == 'test':
        eval_mode = cfg.get('TEST', {}).get('MODE', 'slide')
    else:
        eval_mode = cfg.get('EVAL', {}).get('MODE', 'whole')

    # Evaluate
    print("\nEvaluating...")
    print(f"Mode: {eval_mode} (from cfg['{args.split.upper()}']['MODE'])")

    metrics, avg_time, fps = evaluate(
        model, dataloader, dataset, device, cfg, eval_mode=eval_mode,
        save_vis=args.save_vis, vis_dir=args.vis_dir
    )

    # Print results
    print("\n")
    metrics.print_results()

    print(f"\nInference speed:")
    print(f"  Average time per image: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.2f}")

    if args.output_dir:
        save_results(
            metrics,
            args.output_dir,
            avg_time,
            fps,
            args.split,
            checkpoint_path,
            len(dataset),
        )

    if args.save_vis:
        print(f"\nVisualizations saved to: {args.vis_dir}")


if __name__ == '__main__':
    main()
