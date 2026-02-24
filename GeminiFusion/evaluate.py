"""
GeminiFusion Evaluation Script for UAVScenes Dataset

Implements sliding window inference for fair comparison with CMNeXt/DFormerv2.
Saves results to JSON and TXT files automatically.

Usage:
    python evaluate.py --resume ckpt/model-best.pth.tar --backbone mit_b2
"""

import os
import argparse
import json
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from models.segformer import WeTr
from datasets.uavscenes import UAVScenesDataset, CLASS_NAMES
from utils.transforms import ToTensor
from utils.augmentations_mm import Normalize, Resize
from utils.meter import confusion_matrix, getScores


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GeminiFusion UAVScenes Evaluation")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/bariskaya/Projelerim/UAV/UAVScenesData",
        help="Path to UAVScenes dataset",
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mit_b2",
        help="Backbone network",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of cross-attention heads",
    )
    parser.add_argument(
        "--dpr",
        type=float,
        default=0.1,
        help="Drop path rate",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=768,
        help="Sliding window size",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride",
    )
    parser.add_argument(
        "--hag-max-height",
        type=float,
        default=50.0,
        help="Maximum HAG height for normalization",
    )
    parser.add_argument(
        "--save-pred",
        action="store_true",
        help="Save prediction images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for results (defaults to checkpoint directory)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Number of classes",
    )

    return parser.parse_args()


def sliding_window_inference(
    model, rgb, hag, window_size=768, stride=512, num_classes=19
):
    """
    Perform sliding window inference for large images.

    Args:
        model: Segmentation model
        rgb: RGB image tensor (1, 3, H, W)
        hag: HAG image tensor (1, 3, H, W)
        window_size: Sliding window size
        stride: Sliding window stride
        num_classes: Number of output classes

    Returns:
        Segmentation prediction (H, W)
    """
    _, _, H, W = rgb.shape
    device = rgb.device

    # Initialize prediction and count maps
    pred_map = torch.zeros((num_classes, H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # Calculate padding if needed
    pad_h = (window_size - H % window_size) % window_size if H < window_size else 0
    pad_w = (window_size - W % window_size) % window_size if W < window_size else 0

    if H < window_size or W < window_size:
        # Pad image if smaller than window
        rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode='reflect')
        hag = F.pad(hag, (0, pad_w, 0, pad_h), mode='reflect')
        padded_H, padded_W = rgb.shape[2], rgb.shape[3]
        pred_map = torch.zeros((num_classes, padded_H, padded_W), device=device)
        count_map = torch.zeros((padded_H, padded_W), device=device)
    else:
        padded_H, padded_W = H, W

    # Sliding window
    for y in range(0, padded_H - window_size + 1, stride):
        for x in range(0, padded_W - window_size + 1, stride):
            # Extract window
            rgb_crop = rgb[:, :, y:y+window_size, x:x+window_size]
            hag_crop = hag[:, :, y:y+window_size, x:x+window_size]

            # Forward pass
            outputs, _ = model([rgb_crop, hag_crop])
            pred = outputs[-1]  # Ensemble output

            # Upsample to window size if needed
            if pred.shape[2] != window_size or pred.shape[3] != window_size:
                pred = F.interpolate(
                    pred, size=(window_size, window_size),
                    mode='bilinear', align_corners=False
                )

            # Accumulate predictions
            pred_map[:, y:y+window_size, x:x+window_size] += pred[0]
            count_map[y:y+window_size, x:x+window_size] += 1

    # Handle last column and row if not covered
    if (padded_W - window_size) % stride != 0:
        x = padded_W - window_size
        for y in range(0, padded_H - window_size + 1, stride):
            rgb_crop = rgb[:, :, y:y+window_size, x:x+window_size]
            hag_crop = hag[:, :, y:y+window_size, x:x+window_size]
            outputs, _ = model([rgb_crop, hag_crop])
            pred = outputs[-1]
            if pred.shape[2] != window_size:
                pred = F.interpolate(pred, size=(window_size, window_size), mode='bilinear', align_corners=False)
            pred_map[:, y:y+window_size, x:x+window_size] += pred[0]
            count_map[y:y+window_size, x:x+window_size] += 1

    if (padded_H - window_size) % stride != 0:
        y = padded_H - window_size
        for x in range(0, padded_W - window_size + 1, stride):
            rgb_crop = rgb[:, :, y:y+window_size, x:x+window_size]
            hag_crop = hag[:, :, y:y+window_size, x:x+window_size]
            outputs, _ = model([rgb_crop, hag_crop])
            pred = outputs[-1]
            if pred.shape[2] != window_size:
                pred = F.interpolate(pred, size=(window_size, window_size), mode='bilinear', align_corners=False)
            pred_map[:, y:y+window_size, x:x+window_size] += pred[0]
            count_map[y:y+window_size, x:x+window_size] += 1

    # Average predictions
    count_map = torch.clamp(count_map, min=1)
    pred_map = pred_map / count_map.unsqueeze(0)

    # Remove padding
    pred_map = pred_map[:, :H, :W]

    # Get class predictions
    prediction = pred_map.argmax(dim=0).cpu().numpy().astype(np.uint8)

    return prediction


def evaluate(args):
    """Run evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine output directory
    if args.output_dir:
        results_dir = args.output_dir
    else:
        # Default: same directory as checkpoint
        results_dir = os.path.join(os.path.dirname(args.resume), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create model
    print(f"Loading model with backbone: {args.backbone}")
    model = WeTr(
        args.backbone,
        num_classes=args.num_classes,
        n_heads=args.n_heads,
        dpr=args.dpr,
        drop_rate=0.0,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

    # Handle DDP checkpoint
    state_dict = checkpoint["segmenter"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # Create validation transforms (no augmentation)
    val_transform = transforms.Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create dataset
    dataset = UAVScenesDataset(
        data_root=args.data_dir,
        split='test',
        transform=val_transform,
        hag_max_height=args.hag_max_height,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Evaluating on {len(dataset)} samples...")
    print(f"Sliding window: {args.window_size}x{args.window_size}, stride={args.stride}")
    print(f"Results will be saved to: {results_dir}")

    # Initialize confusion matrix
    conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    # Create prediction output directory
    if args.save_pred:
        pred_dir = os.path.join(results_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

    # Evaluation loop with timing
    total_time = 0.0
    num_images = 0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            rgb = sample["rgb"].to(device).float()
            hag = sample["depth"].to(device).float()
            gt = sample["mask"][0].numpy().astype(np.uint8)

            # Inference with timing
            torch.cuda.synchronize()
            start_time = time.time()

            pred = sliding_window_inference(
                model, rgb, hag,
                window_size=args.window_size,
                stride=args.stride,
                num_classes=args.num_classes,
            )

            torch.cuda.synchronize()
            total_time += time.time() - start_time
            num_images += 1

            # Update confusion matrix (ignore label 255)
            valid_mask = gt < args.num_classes
            conf_mat += confusion_matrix(
                gt[valid_mask], pred[valid_mask], args.num_classes
            )

            # Save prediction
            if args.save_pred:
                pred_path = os.path.join(pred_dir, f"pred_{i:04d}.png")
                cv2.imwrite(pred_path, pred)

    # Calculate metrics
    glob_acc, mean_acc, miou = getScores(conf_mat)

    # Calculate per-class IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_iou = np.diag(conf_mat) / (
            conf_mat.sum(1) + conf_mat.sum(0) - np.diag(conf_mat)
        ).astype(np.float32)

    # Calculate static and dynamic mIoU
    static_classes = list(range(17))  # Classes 0-16
    dynamic_classes = [17, 18]  # sedan, truck

    static_iou = np.nanmean(per_class_iou[static_classes]) * 100
    dynamic_iou = np.nanmean(per_class_iou[dynamic_classes]) * 100

    # Inference speed
    avg_time_ms = (total_time / num_images) * 1000
    fps = num_images / total_time

    # GPU info
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Global Accuracy: {glob_acc:.2f}%")
    print(f"Mean Accuracy:   {mean_acc:.2f}%")
    print(f"mIoU (All):      {miou:.2f}%")
    print(f"mIoU (Static):   {static_iou:.2f}%")
    print(f"mIoU (Dynamic):  {dynamic_iou:.2f}%")
    print("="*60)

    # Per-class IoU
    print("\nPer-class IoU:")
    print("-"*40)
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_class_iou)):
        cls_type = "[D]" if i >= 17 else "[S]"
        print(f"  {cls_type} {i:2d}. {name:20s}: {iou*100:5.2f}%")
    print("-"*40)

    # Inference speed
    print(f"\nInference Speed ({gpu_name}):")
    print(f"  Average time per image: {avg_time_ms:.1f}ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total images: {num_images}")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_results = {
        "model": "GeminiFusion",
        "backbone": args.backbone,
        "timestamp": timestamp,
        "checkpoint": args.resume,
        "dataset": "UAVScenes",
        "split": "test",
        "num_images": num_images,
        "sliding_window": {
            "window_size": args.window_size,
            "stride": args.stride,
        },
        "metrics": {
            "mIoU": round(float(miou), 2),
            "static_mIoU": round(float(static_iou), 2),
            "dynamic_mIoU": round(float(dynamic_iou), 2),
            "global_accuracy": round(float(glob_acc), 2),
            "mean_accuracy": round(float(mean_acc), 2),
        },
        "inference": {
            "gpu": gpu_name,
            "avg_time_per_image_ms": round(avg_time_ms, 1),
            "fps": round(fps, 2),
        },
        "per_class_iou": {
            name: round(float(iou * 100), 2)
            for name, iou in zip(CLASS_NAMES, per_class_iou)
        },
    }

    json_path = os.path.join(results_dir, "GeminiFusion_test_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Save results to TXT
    txt_path = os.path.join(results_dir, "GeminiFusion_test_results.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"GeminiFusion Test Evaluation Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args.resume}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"mIoU (All):      {miou:.2f}%\n")
        f.write(f"mIoU (Static):   {static_iou:.2f}%  (17 classes)\n")
        f.write(f"mIoU (Dynamic):  {dynamic_iou:.2f}%  (2 classes)\n")
        f.write(f"Global Accuracy: {glob_acc:.2f}%\n")
        f.write(f"Mean Accuracy:   {mean_acc:.2f}%\n\n")

        f.write("INFERENCE SPEED\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg time/image:  {avg_time_ms:.1f} ms\n")
        f.write(f"FPS:             {fps:.2f}\n")
        f.write(f"Total images:    {num_images}\n")
        f.write(f"Window/Stride:   {args.window_size}/{args.stride}\n\n")

        f.write("PER-CLASS IoU\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_class_iou)):
            cls_type = "[D]" if i >= 17 else "[S]"
            f.write(f"  {cls_type} {i:2d}. {name:20s}: {iou*100:5.2f}%\n")
        f.write("-" * 40 + "\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    return miou


if __name__ == "__main__":
    args = get_arguments()
    evaluate(args)
