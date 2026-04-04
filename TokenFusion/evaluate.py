"""
TokenFusion Evaluation Script for UAVScenes

Performs sliding window evaluation matching CMNeXt/DFormerv2 settings:
- Window size: 768x768
- Stride: 512

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth --config configs/uavscenes_rgb_hag.yaml
"""

import os
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Local imports
from models import WeTr
from datasets import UAVScenesDataset
from utils.transforms import ValTransform, TestTransform
from utils.metrics import UAVScenesMetrics, UAVSCENES_CLASSES
from utils.helpers import setup_logger, sliding_window_inference


class Config:
    """Configuration class to hold YAML config as attributes."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='TokenFusion UAVScenes Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/uavscenes_rgb_hag.yaml',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--eval-mode', type=str, default='slide', choices=['slide', 'whole'],
                        help='Inference mode: slide for native benchmark, whole for fast full-image eval')
    parser.add_argument('--legacy-resize-eval', action='store_true',
                        help='Use legacy resize-based transform before evaluation')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Limit evaluation to the first N images for quick speed checks')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--output', type=str, default=str(Path(__file__).resolve().parent / 'results2'),
                        help='Output directory for predictions')
    parser.add_argument('--save-pred', action='store_true',
                        help='Save prediction images')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def build_model(cfg, checkpoint_path, device):
    """Build and load model from checkpoint."""
    model = WeTr(
        backbone=cfg.model.backbone,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        pretrained=None
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def build_dataloader(cfg, split='test', batch_size_override=None, legacy_resize_eval=False, eval_mode='slide',
                     num_images=None):
    """Build test dataloader.

    Note:
        For fair comparison with CMNeXt, test evaluation should NOT resize
        inputs/labels before sliding-window inference.
    """
    if legacy_resize_eval:
        transform = ValTransform(
            rgb_mean=cfg.normalization.rgb_mean,
            rgb_std=cfg.normalization.rgb_std,
            hag_mean=cfg.normalization.hag_mean,
            hag_std=cfg.normalization.hag_std,
            size=cfg.training.image_size,
        )
    else:
        transform = TestTransform(
            rgb_mean=cfg.normalization.rgb_mean,
            rgb_std=cfg.normalization.rgb_std,
            hag_mean=cfg.normalization.hag_mean,
            hag_std=cfg.normalization.hag_std
        )

    test_dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path,
        split=split,
        transform=transform,
        hag_max_height=cfg.hag.max_meters
    )

    if num_images is not None:
        test_dataset.samples = test_dataset.samples[:num_images]

    if batch_size_override is not None:
        batch_size = batch_size_override
    elif eval_mode == 'whole' and legacy_resize_eval:
        batch_size = getattr(cfg.evaluation, 'val_batch_size', 1)
    else:
        batch_size = 1

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return val_loader


def build_result_name(split, eval_mode, legacy_resize_eval):
    """Return result prefix without overwriting the default benchmark outputs."""
    if split == 'test' and eval_mode == 'slide' and not legacy_resize_eval:
        return 'TokenFusion'

    suffixes = [split, eval_mode]
    if legacy_resize_eval:
        suffixes.append('legacyresize')
    return 'TokenFusion_' + '_'.join(suffixes)


@torch.no_grad()
def evaluate(model, val_loader, cfg, device, eval_mode='slide', save_pred=False, output_dir=None,
             split='test', legacy_resize_eval=False):
    """
    Evaluate model with sliding window inference.

    Args:
        model: TokenFusion model
        val_loader: Validation dataloader
        cfg: Configuration
        device: Device
        save_pred: Whether to save predictions
        output_dir: Output directory for predictions
    """
    model.eval()

    metrics_obj = UAVScenesMetrics(
        num_classes=cfg.dataset.num_classes,
        ignore_label=cfg.dataset.ignore_label
    )

    print("\nRunning evaluation with sliding window inference...")
    print(f"  Split: {split}")
    print(f"  Eval mode: {eval_mode}")
    print(f"  Legacy resize eval: {'enabled' if legacy_resize_eval else 'disabled'}")
    if eval_mode == 'slide':
        print(f"  Window size: {cfg.evaluation.slide_size}")
        print(f"  Stride: {cfg.evaluation.slide_stride}")
    print(f"  Samples: {len(val_loader.dataset)}")

    start_time = time.time()

    for i, sample in enumerate(tqdm(val_loader, desc="Evaluating")):
        rgb = sample['rgb'].to(device)
        hag = sample['hag'].to(device)
        label = sample['label']

        if eval_mode == 'slide':
            output = sliding_window_inference(
                model, rgb, hag,
                window_size=cfg.evaluation.slide_size,
                stride=cfg.evaluation.slide_stride,
                num_classes=cfg.dataset.num_classes,
                device=device
            )
        else:
            outputs, _ = model([rgb, hag])
            output = outputs[-1]
            label_size = label.shape[1:]
            output = F.interpolate(output, size=label_size, mode='bilinear', align_corners=False)

        # Get predictions
        pred = output.argmax(dim=1).cpu()

        # Update confusion matrix
        metrics_obj.update(pred, label)

        # Save predictions if requested
        if save_pred and output_dir:
            batch_base = i * rgb.shape[0]
            for b in range(pred.shape[0]):
                save_prediction(pred[b].numpy(), batch_base + b, output_dir)

    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / len(val_loader.dataset)) * 1000.0
    fps = len(val_loader.dataset) / elapsed if elapsed > 0 else 0.0
    print(f'\nEvaluation completed in {elapsed:.2f}s ({elapsed / len(val_loader.dataset):.3f}s per image)')
    metrics_obj.print_results()
    print("Inference speed:")
    print(f"  Average time per image: {avg_time_ms:.1f}ms")
    print(f"  FPS: {fps:.2f}")
    result_name = build_result_name(split, eval_mode, legacy_resize_eval)
    metrics_obj.save_results(
        output_dir or str(Path(__file__).resolve().parent / 'results2'),
        result_name,
        avg_time_ms,
        fps,
        len(val_loader.dataset),
    )

    return metrics_obj.get_results()


def save_prediction(pred, idx, output_dir):
    """Save prediction as colored image."""
    import cv2

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Color palette for visualization
    palette = get_palette(19)

    # Convert prediction to colored image
    colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label_id in range(19):
        colored[pred == label_id] = palette[label_id]

    # Save
    output_path = os.path.join(output_dir, f'pred_{idx:05d}.png')
    cv2.imwrite(output_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))


def get_palette(num_classes):
    """Generate color palette for visualization."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    # Define colors for UAVScenes classes
    colors = [
        [128, 128, 128],  # background
        [180, 120, 120],  # roof
        [139, 119, 101],  # dirt_road
        [128, 128, 128],  # paved_road
        [0, 0, 255],      # river
        [0, 255, 255],    # pool
        [100, 100, 100],  # bridge
        [255, 128, 0],    # container
        [200, 200, 200],  # airstrip
        [255, 0, 255],    # traffic_barrier
        [0, 255, 0],      # green_field
        [144, 238, 144],  # wild_field
        [0, 0, 128],      # solar_panel
        [255, 192, 203],  # umbrella
        [176, 224, 230],  # transparent_roof
        [64, 64, 64],     # car_park
        [211, 211, 211],  # paved_walk
        [255, 0, 0],      # sedan (dynamic)
        [0, 128, 0],      # truck (dynamic)
    ]

    for i in range(min(num_classes, len(colors))):
        palette[i] = colors[i]

    return palette


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_model(cfg, args.checkpoint, device)
    print(f"Model: {cfg.model.backbone}")
    print(f"Classes: {cfg.dataset.num_classes}")

    # Build dataloader
    val_loader = build_dataloader(
        cfg,
        split=args.split,
        batch_size_override=args.batch_size,
        legacy_resize_eval=args.legacy_resize_eval,
        eval_mode=args.eval_mode,
        num_images=args.num_images,
    )

    # Evaluate
    evaluate(
        model, val_loader, cfg, device,
        eval_mode=args.eval_mode,
        save_pred=args.save_pred,
        output_dir=args.output,
        split=args.split,
        legacy_resize_eval=args.legacy_resize_eval,
    )


if __name__ == '__main__':
    main()
