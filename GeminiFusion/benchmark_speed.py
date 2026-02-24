"""
Quick inference speed benchmark for GeminiFusion.
Runs sliding window inference on N images and reports timing.
Takes ~2-3 minutes instead of hours.

Usage:
    python benchmark_speed.py --resume ckpt/uavscenes_gf_fair_v1_resume_fp32/model-best.pth.tar \
        --train-dir /home/ganiltd07/UAVScenesData --num-images 20
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models.segformer import WeTr
from datasets.uavscenes import UAVScenesDataset
from utils.transforms import ToTensor
from utils.augmentations_mm import Normalize


def sliding_window_inference(model, inputs, num_classes, window_size=768, stride=512):
    """Sliding window inference matching main.py implementation."""
    B, C, H, W = inputs[0].shape
    device = inputs[0].device

    output_sum = torch.zeros((B, num_classes, H, W), device=device)
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

            crop_inputs = [x[:, :, h_start:h_end, w_start:w_end] for x in inputs]
            outputs, _ = model(crop_inputs)
            crop_output = outputs[-1]

            if crop_output.shape[2:] != (h_end - h_start, w_end - w_start):
                crop_output = F.interpolate(
                    crop_output,
                    size=(h_end - h_start, w_end - w_start),
                    mode='bilinear', align_corners=False
                )

            output_sum[:, :, h_start:h_end, w_start:w_end] += crop_output
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

            if w_end == W:
                break
            w_start += stride

        if h_end == H:
            break
        h_start += stride

    return output_sum / count_map


def main():
    parser = argparse.ArgumentParser(description="GeminiFusion Speed Benchmark")
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to UAVScenes data")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to benchmark")
    parser.add_argument("--backbone", type=str, default="mit_b2")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3, help="Warmup images (excluded from timing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Create model
    model = WeTr(args.backbone, num_classes=args.num_classes, n_heads=8, dpr=0.1, drop_rate=0.0)

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    state_dict = ckpt["segmenter"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # Test transform (full resolution, no resize)
    test_transform = transforms.Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load test dataset
    dataset = UAVScenesDataset(
        data_root=args.train_dir, split='test',
        transform=test_transform, hag_max_height=50.0,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    total_images = min(args.num_images + args.warmup, len(dataset))
    print(f"Benchmarking on {args.num_images} images (+{args.warmup} warmup)")
    print(f"Sliding window: {args.window_size}x{args.window_size}, stride={args.stride}")
    print(f"Image resolution: full (typically 2448x2048)")
    print()

    times = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= total_images:
                break

            rgb = sample["rgb"].float().to(device)
            hag = sample["depth"].float().to(device)
            H, W = rgb.shape[2], rgb.shape[3]

            # Warm up GPU
            torch.cuda.synchronize()
            start = time.time()

            output = sliding_window_inference(
                model, [rgb, hag], args.num_classes,
                window_size=args.window_size, stride=args.stride
            )
            pred = output.argmax(dim=1)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            if i < args.warmup:
                print(f"  [warmup {i+1}/{args.warmup}] {H}x{W} -> {elapsed*1000:.1f}ms")
            else:
                times.append(elapsed)
                print(f"  [{i+1-args.warmup}/{args.num_images}] {H}x{W} -> {elapsed*1000:.1f}ms")

    # Report
    times = np.array(times)
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS - GeminiFusion (Single GPU)")
    print("=" * 60)
    print(f"  GPU:                {torch.cuda.get_device_name(0)}")
    print(f"  Images tested:      {len(times)}")
    print(f"  Avg time/image:     {times.mean()*1000:.1f} ms")
    print(f"  Std time/image:     {times.std()*1000:.1f} ms")
    print(f"  Min time/image:     {times.min()*1000:.1f} ms")
    print(f"  Max time/image:     {times.max()*1000:.1f} ms")
    print(f"  FPS:                {1.0/times.mean():.2f}")
    print(f"  Sliding window:     {args.window_size}x{args.window_size}")
    print(f"  Stride:             {args.stride}")
    print("=" * 60)


if __name__ == "__main__":
    main()
