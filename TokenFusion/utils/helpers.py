"""
Helper Utilities for TokenFusion Training

Includes:
- Logging setup
- Checkpoint save/load
- AverageMeter for tracking metrics
- Misc utilities
"""

import os
import logging
import torch
import numpy as np
import random
from datetime import datetime


def setup_logger(log_dir=None, name='tokenfusion'):
    """
    Setup logger for training.

    Args:
        log_dir: Directory to save log file
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Log file: {log_file}')

    return logger


def save_checkpoint(state, filename='checkpoint.pth', is_best=False, best_filename='best.pth'):
    """
    Save training checkpoint.

    Args:
        state: State dict (model, optimizer, epoch, metrics)
        filename: Checkpoint filename
        is_best: Whether this is the best model
        best_filename: Best model filename
    """
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """
    Load checkpoint.

    Args:
        filename: Checkpoint file path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to

    Returns:
        Dictionary with checkpoint info (epoch, best_metric, etc.)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_miou': checkpoint.get('best_miou', 0),
        'best_epoch': checkpoint.get('best_epoch', 0)
    }

    return info


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def print_config(cfg, logger=None):
    """Print configuration."""
    msg = "\n" + "=" * 60 + "\n"
    msg += "Configuration:\n"
    msg += "=" * 60 + "\n"

    if hasattr(cfg, '__dict__'):
        for k, v in cfg.__dict__.items():
            if not k.startswith('_'):
                msg += f"  {k}: {v}\n"
    elif isinstance(cfg, dict):
        for k, v in cfg.items():
            msg += f"  {k}: {v}\n"

    msg += "=" * 60 + "\n"

    if logger:
        logger.info(msg)
    else:
        print(msg)


def sliding_window_inference(
    model,
    rgb,
    hag,
    window_size=768,
    stride=512,
    num_classes=19,
    device='cuda'
):
    """
    Sliding window inference for large images.

    Args:
        model: Segmentation model
        rgb: RGB tensor [1, 3, H, W]
        hag: HAG tensor [1, 3, H, W]
        window_size: Window size
        stride: Stride between windows
        num_classes: Number of classes
        device: Device

    Returns:
        Prediction tensor [1, num_classes, H, W]
    """
    model.eval()

    B, C, H, W = rgb.shape
    assert B == 1, "Batch size must be 1 for sliding window inference"

    # Initialize output and count tensors
    output = torch.zeros(1, num_classes, H, W, device=device)
    count = torch.zeros(1, 1, H, W, device=device)

    # Sliding window
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Compute window bounds
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)
            y_start = max(0, y_end - window_size)
            x_start = max(0, x_end - window_size)

            # Extract window
            rgb_window = rgb[:, :, y_start:y_end, x_start:x_end]
            hag_window = hag[:, :, y_start:y_end, x_start:x_end]

            # Pad if necessary
            pad_h = window_size - (y_end - y_start)
            pad_w = window_size - (x_end - x_start)
            if pad_h > 0 or pad_w > 0:
                rgb_window = torch.nn.functional.pad(rgb_window, (0, pad_w, 0, pad_h))
                hag_window = torch.nn.functional.pad(hag_window, (0, pad_w, 0, pad_h))

            # Forward pass
            with torch.no_grad():
                outputs, _ = model([rgb_window.to(device), hag_window.to(device)])
                # Use ensemble output
                pred = outputs[2]  # [B, num_classes, H, W]

                # Upsample to window size if needed
                if pred.shape[2] != window_size or pred.shape[3] != window_size:
                    pred = torch.nn.functional.interpolate(
                        pred, size=(window_size, window_size),
                        mode='bilinear', align_corners=False
                    )

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :window_size - pad_h, :window_size - pad_w]

            # Accumulate
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            output[:, :, y_start:y_end, x_start:x_end] += pred[:, :, :actual_h, :actual_w]
            count[:, :, y_start:y_end, x_start:x_end] += 1

    # Average overlapping regions
    output = output / count

    return output
