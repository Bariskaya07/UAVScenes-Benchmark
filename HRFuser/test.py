"""
HRFuser Test Script for UAVScenes Dataset

Standalone evaluation script with sliding window inference.
Produces per-class IoU, mIoU (overall, static, dynamic), pixel accuracy, and mean accuracy.

Usage:
    python test.py --config configs/uavscenes_rgb_hag.yaml --ckpt-path checkpoints/best.pth
    python test.py --config configs/uavscenes_rgb_hag.yaml --ckpt-path checkpoints/best.pth --save-image -1
    python test.py --config configs/uavscenes_rgb_hag.yaml --ckpt-path checkpoints/best.pth --flip-test
"""

import os
import argparse
import yaml
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.hrfuser_segformer import HRFuserSegFormer
from datasets.uavscenes import UAVScenesDataset, CLASS_NAMES


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


def get_arguments():
    parser = argparse.ArgumentParser(description="HRFuser UAVScenes Test")
    parser.add_argument("--config", type=str, default="configs/uavscenes_rgb_hag.yaml",
                        help="Path to config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"])
    parser.add_argument("--save-image", type=int, default=0,
                        help="Number of images to save (-1 for all)")
    parser.add_argument("--save-dir", type=str, default="test_results")
    parser.add_argument("--flip-test", action="store_true", default=False,
                        help="Enable horizontal flip test-time augmentation")
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


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
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_arguments()

    # Load config
    cfg = load_config(args.config)
    num_classes = cfg.dataset.num_classes

    # Create model
    model = HRFuserSegFormer(
        num_classes=num_classes,
        embedding_dim=cfg.model.embedding_dim,
        drop_path_rate=cfg.model.drop_path_rate,
        num_fused_modalities=cfg.model.num_fused_modalities,
        mod_in_channels=cfg.model.mod_in_channels)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

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
    model.cuda()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    # Create dataset
    from torchvision import transforms
    from utils.transforms import ToTensor
    from utils.augmentations_mm import Normalize, Resize

    slide_size = cfg.evaluation.slide_size
    composed_test = transforms.Compose([
        ToTensor(),
        Resize([slide_size, slide_size]),
        Normalize(cfg.normalization.rgb_mean, cfg.normalization.rgb_std),
    ])

    dataset = UAVScenesDataset(
        data_root=cfg.dataset.data_path, split=args.split,
        transform=composed_test, hag_max_height=cfg.hag.max_meters)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True)

    print(f"Testing on {len(dataset)} images ({args.split} split)")

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Test
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_times = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            rgb = sample["rgb"].float().cuda()
            hag = sample["depth"].float().cuda()
            target = sample["mask"]
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes

            start_time = time.time()

            # Sliding window inference
            output = slide_inference(
                model, rgb, hag, num_classes,
                cfg.evaluation.slide_size, cfg.evaluation.slide_stride)

            # Optional flip augmentation
            if args.flip_test:
                rgb_flip = torch.flip(rgb, dims=[3])
                hag_flip = torch.flip(hag, dims=[3])
                output_flip = slide_inference(
                    model, rgb_flip, hag_flip, num_classes,
                    cfg.evaluation.slide_size, cfg.evaluation.slide_stride)
                output_flip = torch.flip(output_flip, dims=[3])
                output = (output + output_flip) / 2.0

            end_time = time.time()
            all_times += end_time - start_time

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

            # Save prediction images
            if args.save_image != 0 and (i < args.save_image or args.save_image == -1):
                pred_path = os.path.join(args.save_dir, f"pred_{i:04d}.png")
                cv2.imwrite(pred_path, pred)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")

    # Compute metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(conf_mat)
        fp = conf_mat.sum(axis=0) - tp
        fn = conf_mat.sum(axis=1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / union, 0)

        pixel_acc = tp.sum() / conf_mat.sum() * 100.0
        class_total = conf_mat.sum(axis=1)
        class_acc = np.where(class_total > 0, tp / class_total, 0)
        valid_mask = class_total > 0
        mean_acc = np.nanmean(class_acc[valid_mask]) * 100.0
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

    # Results
    latency = all_times / len(dataset)
    fps = 1.0 / latency if latency > 0 else 0

    print(f"\n{'='*60}")
    print(f"HRFuser-T UAVScenes Test Results ({args.split} split)")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Total images: {len(dataset)}")
    print(f"Total time: {all_times:.1f}s")
    print(f"Latency: {latency*1000:.1f}ms")
    print(f"FPS: {fps:.1f}")

    print(f"\nmIoU:         {miou:.2f}%")
    print(f"Static mIoU:  {static_miou:.2f}%")
    print(f"Dynamic mIoU: {dynamic_miou:.2f}%")
    print(f"Pixel Acc:    {pixel_acc:.2f}%")
    print(f"Mean Acc:     {mean_acc:.2f}%")

    # Per-class IoU
    print(f"\nPer-class IoU:")
    print(f"{'-'*40}")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        marker = " (dynamic)" if cls_idx >= 17 else ""
        print(f"  {cls_idx:2d}. {cls_name:<20s}: {iou[cls_idx]*100:5.2f}%{marker}")
    print(f"{'-'*40}")
    print(f"  {'mIoU':<24s}: {miou:5.2f}%")

    # Save results to file
    results_path = os.path.join(args.save_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"HRFuser-T UAVScenes Test Results ({args.split} split)\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Config: {args.config}\n\n")
        f.write(f"mIoU: {miou:.2f}%\n")
        f.write(f"Static mIoU: {static_miou:.2f}%\n")
        f.write(f"Dynamic mIoU: {dynamic_miou:.2f}%\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.2f}%\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}%\n\n")
        f.write(f"Per-class IoU:\n")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            marker = " (dynamic)" if cls_idx >= 17 else ""
            f.write(f"  {cls_idx:2d}. {cls_name:<20s}: {iou[cls_idx]*100:5.2f}%{marker}\n")
        f.write(f"\nLatency: {latency*1000:.1f}ms\n")
        f.write(f"FPS: {fps:.1f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
