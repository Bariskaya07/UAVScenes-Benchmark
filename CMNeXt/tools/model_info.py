#!/usr/bin/env python3
"""
Quick script to print model parameters and GFLOPs.
Run: python tools/model_info.py
"""
import sys
sys.path.insert(0, '.')

import torch
from semseg.models.cmnext import CMNeXt

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_gflops(model, input_size=(768, 768)):
    """Estimate GFLOPs using fvcore if available, else manual estimate."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_rgb = torch.randn(1, 3, *input_size).cuda()
        dummy_hag = torch.randn(1, 3, *input_size).cuda()  # 3ch stacked HAG
        model = model.cuda().eval()
        flops = FlopCountAnalysis(model, (dummy_rgb, dummy_hag))
        return flops.total() / 1e9
    except ImportError:
        print("fvcore not installed. Install with: pip install fvcore")
        return None

def main():
    # Create model (same config as training)
    model = CMNeXt(
        backbone='MiT-B2',
        num_classes=19,
        modals=['image', 'hag'],
        extra_in_chans=3,  # 3ch stacked HAG
    )

    total, trainable = count_parameters(model)

    print("=" * 60)
    print("CMNeXt Model Info (MiT-B2 + HAG)")
    print("=" * 60)
    print(f"Total Parameters:     {total / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable / 1e6:.2f}M")

    gflops = estimate_gflops(model)
    if gflops:
        print(f"GFLOPs (768x768):     {gflops:.2f}")

    print("=" * 60)

if __name__ == '__main__':
    main()
