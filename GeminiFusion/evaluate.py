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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from models.segformer import WeTr
from datasets.uavscenes import UAVScenesDataset, CLASS_NAMES
from utils.transforms import ToTensor
from utils.augmentations_mm import Normalize, Resize
from utils.meter import confusion_matrix, getScores

try:
    from torch.amp import autocast as _amp_autocast

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast("cuda", enabled=enabled, dtype=dtype)
except Exception:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import autocast as _amp_autocast

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast(enabled=enabled, dtype=dtype)


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
        "--batch-size",
        type=int,
        default=1,
        help="Test batch size",
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
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP autocast dtype for CUDA evaluation",
    )

    return parser.parse_args()


def resolve_amp_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported --amp-dtype '{dtype_name}'. Expected 'bf16' or 'fp16'.")


def sliding_window_inference(
    model, rgb, hag, window_size=768, stride=512, num_classes=19,
    amp_enabled=True, amp_dtype=torch.bfloat16
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
        Segmentation prediction (B, H, W)
    """
    B, _, H, W = rgb.shape
    device = rgb.device

    # Initialize prediction and count maps
    pred_map = torch.zeros((B, num_classes, H, W), device=device)
    count_map = torch.zeros((B, 1, H, W), device=device)

    h_start = 0
    while h_start < H:
        h_end = min(h_start + window_size, H)
        if h_end - h_start < window_size and h_start > 0:
            h_start = H - window_size
            h_end = H

        w_start = 0
        while w_start < W:
            w_end = min(w_start + window_size, W)
            if w_end - w_start < window_size and w_start > 0:
                w_start = W - window_size
                w_end = W

            rgb_crop = rgb[:, :, h_start:h_end, w_start:w_end]
            hag_crop = hag[:, :, h_start:h_end, w_start:w_end]

            with amp_autocast(enabled=amp_enabled, dtype=amp_dtype):
                outputs, _ = model([rgb_crop, hag_crop])
            pred = outputs[-1]

            if pred.shape[2:] != (h_end - h_start, w_end - w_start):
                pred = F.interpolate(
                    pred,
                    size=(h_end - h_start, w_end - w_start),
                    mode='bilinear',
                    align_corners=False
                )

            pred_map[:, :, h_start:h_end, w_start:w_end] += pred
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

            if w_end == W:
                break
            w_start += stride

        if h_end == H:
            break
        h_start += stride

    # Average predictions
    count_map = torch.clamp(count_map, min=1)
    pred_map = pred_map / torch.clamp(count_map, min=1)

    # Get class predictions
    prediction = pred_map.argmax(dim=1).cpu().numpy().astype(np.uint8)

    return prediction


def evaluate(args):
    """Run evaluation."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    local_rank = 0

    if distributed:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype)

    # Determine output directory
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2")
    os.makedirs(results_dir, exist_ok=True)

    # Create model
    print(f"Loading model with backbone: {args.backbone}")
    model = WeTr(
        args.backbone,
        num_classes=args.num_classes,
        n_heads=args.n_heads,
        dpr=args.dpr,
        drop_rate=0.0,
        load_pretrained=False,
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

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    if rank == 0:
        print(f"Evaluating on {len(dataset)} samples...")
        print(f"Sliding window: {args.window_size}x{args.window_size}, stride={args.stride}")
        if amp_enabled:
            print(f"AMP: enabled ({args.amp_dtype})")
        else:
            print("AMP: disabled (CPU)")
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

    iterator = enumerate(dataloader)
    if rank == 0:
        iterator = enumerate(tqdm(dataloader, total=len(dataloader), desc="Evaluating", leave=True))

    with torch.no_grad():
        for i, sample in iterator:
            rgb = sample["rgb"].to(device).float()
            hag = sample["depth"].to(device).float()
            gt = sample["mask"].numpy().astype(np.uint8)

            # Inference with timing
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.time()

            pred = sliding_window_inference(
                model, rgb, hag,
                window_size=args.window_size,
                stride=args.stride,
                num_classes=args.num_classes,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.time() - start_time
            num_images += gt.shape[0]

            # Update confusion matrix (ignore label 255)
            valid_mask = gt < args.num_classes
            conf_mat += confusion_matrix(
                gt[valid_mask], pred[valid_mask], args.num_classes
            )

            # Save prediction
            if args.save_pred:
                batch_base = i * rgb.shape[0]
                for b in range(pred.shape[0]):
                    pred_path = os.path.join(pred_dir, f"pred_{batch_base + b:04d}.png")
                    cv2.imwrite(pred_path, pred[b])

    # Aggregate across ranks for correct multi-GPU metrics/speed
    if distributed:
        cm_tensor = torch.as_tensor(conf_mat, dtype=torch.int64, device=device)
        dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
        conf_mat = cm_tensor.cpu().numpy()

        speed_tensor = torch.tensor(
            [float(total_time), float(num_images)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(speed_tensor, op=dist.ReduceOp.SUM)
        total_time = float(speed_tensor[0].item())
        num_images = int(round(float(speed_tensor[1].item())))

    if conf_mat.sum() == 0:
        raise RuntimeError("Confusion matrix is empty. Check predictions and labels.")

    # Calculate metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(conf_mat)
        fp = conf_mat.sum(axis=0) - tp
        fn = conf_mat.sum(axis=1) - tp
        union = tp + fp + fn
        per_class_iou = np.where(union > 0, tp / union, 0)

        class_total = conf_mat.sum(axis=1)
        support = class_total.astype(np.int64)
        valid_mask = class_total > 0

        class_acc = np.where(class_total > 0, tp / class_total, 0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0
        )

        glob_acc = tp.sum() / conf_mat.sum() * 100.0
        mean_acc = np.nanmean(class_acc[valid_mask]) * 100.0
        miou = np.nanmean(per_class_iou[valid_mask]) * 100.0
        mean_precision = np.nanmean(precision[valid_mask]) * 100.0
        mean_recall = np.nanmean(recall[valid_mask]) * 100.0
        mean_f1 = np.nanmean(f1[valid_mask]) * 100.0

        # Static and dynamic mIoU
        static_mask = np.zeros(args.num_classes, dtype=bool)
        static_mask[:17] = True
        static_valid = valid_mask & static_mask
        static_iou = np.nanmean(per_class_iou[static_valid]) * 100.0 if static_valid.any() else 0.0

        dynamic_mask = np.zeros(args.num_classes, dtype=bool)
        if args.num_classes > 17:
            dynamic_mask[17:] = True
        dynamic_valid = valid_mask & dynamic_mask
        dynamic_iou = np.nanmean(per_class_iou[dynamic_valid]) * 100.0 if dynamic_valid.any() else 0.0

    # Inference speed
    avg_time_ms = (total_time / max(num_images, 1)) * 1000.0
    fps = (num_images / total_time) if total_time > 0 else 0.0

    # GPU info
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # Only rank0 prints/saves
    if rank != 0:
        if distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return miou

    # Print results
    print(f"\nmIoU: {miou:.2f}% | Static: {static_iou:.2f}% | Dynamic: {dynamic_iou:.2f}% | Acc: {glob_acc:.2f}%")
    print("\n" + "=" * 100)
    print(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}")
    print("-" * 100)
    for i, name in enumerate(CLASS_NAMES):
        if support[i] <= 0:
            continue
        cls_type = "[D]" if i >= 17 else "[S]"
        print(
            f"  {cls_type} {name:<15} "
            f"{per_class_iou[i] * 100:>7.2f}% "
            f"{precision[i] * 100:>7.2f}% "
            f"{recall[i] * 100:>7.2f}% "
            f"{f1[i] * 100:>7.2f}% "
            f"{class_acc[i] * 100:>7.2f}% "
            f"{support[i]:>11,}"
        )
    print("-" * 100)
    print(f"  {'mIoU':<18} {miou:>7.2f}%")
    print(f"  {'Static mIoU':<18} {static_iou:>7.2f}%")
    print(f"  {'Dynamic mIoU':<18} {dynamic_iou:>7.2f}%")
    print(f"  {'Pixel Accuracy':<18} {glob_acc:>7.2f}%")
    print(f"  {'Mean Accuracy':<18} {mean_acc:>7.2f}%")
    print(f"  {'Mean Precision':<18} {mean_precision:>7.2f}%")
    print(f"  {'Mean Recall':<18} {mean_recall:>7.2f}%")
    print(f"  {'Mean F1':<18} {mean_f1:>7.2f}%")
    print("=" * 100)

    # Top confused class pairs
    top_confusions = []
    cm_copy = conf_mat.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_indices = np.argsort(cm_copy.ravel())[::-1][:5]
    print("\nMost Confused Class Pairs (Top 5):")
    print("-" * 100)
    for flat_idx in flat_indices:
        true_cls = flat_idx // args.num_classes
        pred_cls = flat_idx % args.num_classes
        count = int(cm_copy[true_cls, pred_cls])
        if count <= 0:
            continue
        print(f"  {CLASS_NAMES[true_cls]:<18} -> {CLASS_NAMES[pred_cls]:<18}: {count:>12,} pixels")
        top_confusions.append({
            "true_class": CLASS_NAMES[true_cls],
            "pred_class": CLASS_NAMES[pred_cls],
            "count": count,
        })
    print("=" * 100)

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
            "mean_precision": round(float(mean_precision), 2),
            "mean_recall": round(float(mean_recall), 2),
            "mean_f1": round(float(mean_f1), 2),
        },
        "inference": {
            "gpu": gpu_name,
            "avg_time_per_image_ms": round(avg_time_ms, 1),
            "fps": round(fps, 2),
        },
        "per_class": {
            name: {
                "iou": round(float(per_class_iou[i] * 100), 2),
                "precision": round(float(precision[i] * 100), 2),
                "recall": round(float(recall[i] * 100), 2),
                "f1": round(float(f1[i] * 100), 2),
                "accuracy": round(float(class_acc[i] * 100), 2),
                "support": int(support[i]),
                "type": "dynamic" if i >= 17 else "static",
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "per_class_iou": {
            name: round(float(iou * 100), 2)
            for name, iou in zip(CLASS_NAMES, per_class_iou)
        },
        "top_confusions": top_confusions,
    }

    json_path = os.path.join(results_dir, "GeminiFusion_test_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Save results to TXT
    txt_path = os.path.join(results_dir, "GeminiFusion_test_results.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"GeminiFusion Test Evaluation Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args.resume}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write("=" * 100 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"mIoU (All):      {miou:.2f}%\n")
        f.write(f"mIoU (Static):   {static_iou:.2f}%  (17 classes)\n")
        f.write(f"mIoU (Dynamic):  {dynamic_iou:.2f}%  (2 classes)\n")
        f.write(f"Global Accuracy: {glob_acc:.2f}%\n")
        f.write(f"Mean Accuracy:   {mean_acc:.2f}%\n")
        f.write(f"Mean Precision:  {mean_precision:.2f}%\n")
        f.write(f"Mean Recall:     {mean_recall:.2f}%\n")
        f.write(f"Mean F1:         {mean_f1:.2f}%\n\n")

        f.write("INFERENCE SPEED\n")
        f.write("-" * 100 + "\n")
        f.write(f"Avg time/image:  {avg_time_ms:.1f} ms\n")
        f.write(f"FPS:             {fps:.2f}\n")
        f.write(f"Total images:    {num_images}\n")
        f.write(f"Window/Stride:   {args.window_size}/{args.stride}\n\n")

        f.write("PER-CLASS RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}\n")
        f.write("-" * 100 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            if support[i] <= 0:
                continue
            cls_type = "[D]" if i >= 17 else "[S]"
            f.write(
                f"  {cls_type} {name:<15} "
                f"{per_class_iou[i] * 100:>7.2f}% "
                f"{precision[i] * 100:>7.2f}% "
                f"{recall[i] * 100:>7.2f}% "
                f"{f1[i] * 100:>7.2f}% "
                f"{class_acc[i] * 100:>7.2f}% "
                f"{support[i]:>11,}\n"
            )
        f.write("=" * 100 + "\n")
        f.write("\nMOST CONFUSED CLASS PAIRS (TOP 5)\n")
        f.write("-" * 100 + "\n")
        if top_confusions:
            for item in top_confusions:
                f.write(
                    f"  {item['true_class']:<18} -> {item['pred_class']:<18}: "
                    f"{item['count']:>12,} pixels\n"
                )
        else:
            f.write("  No class confusion detected.\n")
        f.write("=" * 100 + "\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return miou


if __name__ == "__main__":
    args = get_arguments()
    evaluate(args)
