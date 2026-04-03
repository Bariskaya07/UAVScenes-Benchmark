"""
HRFuser Test Script for UAVScenes Dataset

Standalone evaluation script with sliding window inference.
Produces per-class IoU/Precision/Recall/F1/Accuracy/Support, mIoU
(overall, static, dynamic), and inference speed (latency/FPS).

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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Test batch size")
    parser.add_argument("--save-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2"))
    parser.add_argument("--flip-test", action="store_true", default=False,
                        help="Enable horizontal flip test-time augmentation")
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def load_torch_checkpoint(path, map_location):
    """Load checkpoint with PyTorch>=2.6 and older-version compatibility."""
    try:
        # Checkpoints include more than raw tensor weights.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def top_confused_pairs(conf_mat, class_names, top_k=5):
    """Return top confused class pairs as (true_name, pred_name, count)."""
    cm_copy = conf_mat.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_indices = np.argsort(cm_copy.ravel())[::-1][:top_k]
    pairs = []
    num_classes = cm_copy.shape[0]
    for idx in flat_indices:
        true_cls = idx // num_classes
        pred_cls = idx % num_classes
        count = int(cm_copy[true_cls, pred_cls])
        if count > 0:
            pairs.append((class_names[true_cls], class_names[pred_cls], count))
    return pairs


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
    checkpoint = load_torch_checkpoint(args.ckpt_path, map_location='cpu')

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
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True)

    print(f"Testing on {len(dataset)} images ({args.split} split)")

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Test
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_times = 0
    processed_images = 0
    next_progress = 100

    with torch.no_grad():
        for i, sample in enumerate(loader):
            rgb = sample["rgb"].float().cuda()
            hag = sample["depth"].float().cuda()
            target = sample["mask"]

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

                sample_index = processed_images + b
                if args.save_image != 0 and (sample_index < args.save_image or args.save_image == -1):
                    pred_path = os.path.join(args.save_dir, f"pred_{sample_index:04d}.png")
                    cv2.imwrite(pred_path, pred)

            processed_images += batch_size
            if processed_images >= next_progress or processed_images == len(dataset):
                print(f"Processed {processed_images}/{len(dataset)} images")
                while next_progress <= processed_images:
                    next_progress += 100

    if conf_mat.sum() == 0:
        raise RuntimeError("Confusion matrix is empty. Check labels/predictions in test set.")

    # Compute metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(conf_mat)
        fp = conf_mat.sum(axis=0) - tp
        fn = conf_mat.sum(axis=1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / union, 0)

        class_total = conf_mat.sum(axis=1)
        class_acc = np.where(class_total > 0, tp / class_total, 0)
        valid_mask = class_total > 0
        support = class_total.astype(np.int64)

        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0
        )

        pixel_acc = tp.sum() / conf_mat.sum() * 100.0
        mean_acc = np.nanmean(class_acc[valid_mask]) * 100.0
        miou = np.nanmean(iou[valid_mask]) * 100.0
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

    # Results
    latency = all_times / max(len(dataset), 1)
    fps = 1.0 / latency if latency > 0 else 0.0

    print(f"\nmIoU: {miou:.2f}% | Static: {static_miou:.2f}% | "
          f"Dynamic: {dynamic_miou:.2f}% | Acc: {pixel_acc:.2f}%")
    print("\n" + "=" * 100)
    print(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}")
    print("-" * 100)
    for cls_idx, cls_name in enumerate(CLASS_NAMES[:num_classes]):
        if support[cls_idx] <= 0:
            continue
        cls_type = "[D]" if cls_idx >= 17 else "[S]"
        print(
            f"  {cls_type} {cls_name:<15} "
            f"{iou[cls_idx] * 100:>7.2f}% "
            f"{precision[cls_idx] * 100:>7.2f}% "
            f"{recall[cls_idx] * 100:>7.2f}% "
            f"{f1[cls_idx] * 100:>7.2f}% "
            f"{class_acc[cls_idx] * 100:>7.2f}% "
            f"{support[cls_idx]:>11,}"
        )
    print("-" * 100)
    print(f"  {'mIoU':<18} {miou:>7.2f}%")
    print(f"  {'Static mIoU':<18} {static_miou:>7.2f}%")
    print(f"  {'Dynamic mIoU':<18} {dynamic_miou:>7.2f}%")
    print(f"  {'Pixel Accuracy':<18} {pixel_acc:>7.2f}%")
    print(f"  {'Mean Accuracy':<18} {mean_acc:>7.2f}%")
    print(f"  {'Mean Precision':<18} {mean_precision:>7.2f}%")
    print(f"  {'Mean Recall':<18} {mean_recall:>7.2f}%")
    print(f"  {'Mean F1':<18} {mean_f1:>7.2f}%")
    print("=" * 100)
    print("\nMost Confused Class Pairs (Top 5):")
    print("-" * 60)
    confused_pairs = top_confused_pairs(conf_mat, CLASS_NAMES[:num_classes])
    for true_name, pred_name, count in confused_pairs:
        print(f"  {true_name:<18} -> {pred_name:<18}: {count:>12,} pixels")
    print("=" * 100)
    print("\nInference speed:")
    print(f"  Average time per image: {latency * 1000:.1f}ms")
    print(f"  FPS: {fps:.2f}")

    # Save results to file
    results_path = os.path.join(args.save_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"HRFuser-T UAVScenes Test Results ({args.split} split)\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Total images: {len(dataset)}\n")
        f.write(f"Total time: {all_times:.1f}s\n\n")

        f.write(f"mIoU: {miou:.2f}% | Static: {static_miou:.2f}% | "
                f"Dynamic: {dynamic_miou:.2f}% | Acc: {pixel_acc:.2f}%\n\n")
        f.write(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}\n")
        f.write("-" * 100 + "\n")
        for cls_idx, cls_name in enumerate(CLASS_NAMES[:num_classes]):
            if support[cls_idx] <= 0:
                continue
            cls_type = "[D]" if cls_idx >= 17 else "[S]"
            f.write(
                f"  {cls_type} {cls_name:<15} "
                f"{iou[cls_idx] * 100:>7.2f}% "
                f"{precision[cls_idx] * 100:>7.2f}% "
                f"{recall[cls_idx] * 100:>7.2f}% "
                f"{f1[cls_idx] * 100:>7.2f}% "
                f"{class_acc[cls_idx] * 100:>7.2f}% "
                f"{support[cls_idx]:>11,}\n"
            )
        f.write("-" * 100 + "\n")
        f.write(f"  {'mIoU':<18} {miou:>7.2f}%\n")
        f.write(f"  {'Static mIoU':<18} {static_miou:>7.2f}%\n")
        f.write(f"  {'Dynamic mIoU':<18} {dynamic_miou:>7.2f}%\n")
        f.write(f"  {'Pixel Accuracy':<18} {pixel_acc:>7.2f}%\n")
        f.write(f"  {'Mean Accuracy':<18} {mean_acc:>7.2f}%\n")
        f.write(f"  {'Mean Precision':<18} {mean_precision:>7.2f}%\n")
        f.write(f"  {'Mean Recall':<18} {mean_recall:>7.2f}%\n")
        f.write(f"  {'Mean F1':<18} {mean_f1:>7.2f}%\n")
        f.write("=" * 100 + "\n\n")
        f.write("Most Confused Class Pairs (Top 5):\n")
        f.write("-" * 60 + "\n")
        for true_name, pred_name, count in confused_pairs:
            f.write(f"  {true_name:<18} -> {pred_name:<18}: {count:>12,} pixels\n")
        f.write("=" * 100 + "\n\n")
        f.write("Inference speed:\n")
        f.write(f"  Average time per image: {latency * 1000:.1f}ms\n")
        f.write(f"  FPS: {fps:.2f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
