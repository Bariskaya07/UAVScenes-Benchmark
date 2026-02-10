"""
Quick test for training loop - runs just a few iterations to verify everything works.
"""
import sys
import os

import torch
import torch.nn as nn
from importlib import import_module
import numpy as np

# Load config
config = getattr(import_module("local_configs.UAVScenes.DFormerv2_B"), "C")

print("=" * 60)
print("DFormerv2-Base UAVScenes Training Test")
print("=" * 60)
print(f"Dataset: {config.dataset_name}")
print(f"Image size: {config.image_height}x{config.image_width}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.lr}")

# Create dataset and dataloader
print("\n[1] Creating dataset and dataloader...")
from utils.dataloader.UAVScenesDataset import UAVScenesDataset
from utils.dataloader.dataloader import TrainPre
from torch.utils.data import DataLoader

data_setting = {
    "dataset_path": config.dataset_path,
    "transform_gt": config.gt_transform,
    "class_names": config.class_names,
    "dataset_name": config.dataset_name,
    "backbone": config.backbone,
    "hag_max_meters": getattr(config, 'hag_max_meters', 50.0),
}

train_pre = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)
train_dataset = UAVScenesDataset(data_setting, "train", train_pre, file_length=config.batch_size * 10)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Loader batches: {len(train_loader)}")

# Create model
print("\n[2] Building model...")
from models.builder import EncoderDecoder as segmodel

criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
model = segmodel(
    cfg=config,
    criterion=criterion,
    norm_layer=nn.BatchNorm2d,
    syncbn=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Create optimizer
print("\n[3] Creating optimizer...")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    betas=(0.9, 0.999),
    weight_decay=config.weight_decay,
)

# Training loop test
print("\n[4] Running training iterations...")
model.train()
scaler = torch.cuda.amp.GradScaler()

for idx, batch in enumerate(train_loader):
    if idx >= 5:  # Run only 5 iterations
        break

    imgs = batch["data"].to(device)
    gts = batch["label"].to(device)
    modal_xs = batch["modal_x"].to(device)

    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(imgs, modal_xs, gts)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  Iter {idx+1}: loss = {loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ Training test completed successfully!")
print("=" * 60)
print("\nReady to start full training with:")
print("  bash train_uavscenes.sh")
print("or:")
print("  python utils/train.py --config local_configs.UAVScenes.DFormerv2_B --gpus 1")
