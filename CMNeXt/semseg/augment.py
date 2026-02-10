"""
Multi-Modal Augmentation Pipeline for UAVScenes

Synchronizes augmentations across RGB, HAG, and Label modalities.
Photometric augmentations are applied only to RGB (not HAG).
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, hag, label):
        for t in self.transforms:
            rgb, hag, label = t(rgb, hag, label)
        return rgb, hag, label


class RandomResize:
    """Random resize with scale range.

    Applies the same resize to all modalities.
    """
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, rgb, hag, label):
        scale = random.uniform(*self.scale_range)
        h, w = rgb.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize RGB (bilinear interpolation)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Resize HAG (bilinear interpolation for smooth values)
        if hag is not None:
            hag = cv2.resize(hag, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Resize Label (nearest neighbor to preserve class IDs)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return rgb, hag, label


class RandomCrop:
    """Random crop to fixed size.

    If image is smaller than crop size, pads with zeros (RGB/HAG) and ignore_label (Label).
    """
    def __init__(self, crop_size=(1024, 1024), ignore_label=255):
        self.crop_size = crop_size  # (height, width)
        self.ignore_label = ignore_label

    def __call__(self, rgb, hag, label):
        h, w = rgb.shape[:2]
        crop_h, crop_w = self.crop_size

        # Pad if necessary
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            # Pad RGB with zeros
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Pad HAG with zeros
            if hag is not None:
                hag = cv2.copyMakeBorder(hag, 0, pad_h, 0, pad_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Pad Label with ignore_label
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_CONSTANT, value=self.ignore_label)
            h, w = rgb.shape[:2]

        # Random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Apply crop
        rgb = rgb[top:top+crop_h, left:left+crop_w]
        if hag is not None:
            hag = hag[top:top+crop_h, left:left+crop_w]
        label = label[top:top+crop_h, left:left+crop_w]

        return rgb, hag, label


class RandomHorizontalFlip:
    """Random horizontal flip.

    Same flip applied to all modalities.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb, hag, label):
        if random.random() < self.prob:
            rgb = np.fliplr(rgb).copy()
            if hag is not None:
                hag = np.fliplr(hag).copy()
            label = np.fliplr(label).copy()
        return rgb, hag, label


class PhotoMetricDistortion:
    """Photometric distortion for RGB only.

    Does NOT apply to HAG since it's geometric data, not color.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, rgb, hag, label):
        # Only apply to RGB, leave HAG unchanged
        rgb = rgb.astype(np.float32)

        # Random brightness
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness, self.brightness) * 255
            rgb = rgb + delta
            rgb = np.clip(rgb, 0, 255)

        # Random contrast
        if random.random() < 0.5:
            alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
            rgb = rgb * alpha
            rgb = np.clip(rgb, 0, 255)

        # Convert to HSV for saturation and hue adjustments
        rgb_uint8 = rgb.astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Random saturation
        if random.random() < 0.5:
            alpha = random.uniform(1 - self.saturation, 1 + self.saturation)
            hsv[:, :, 1] = hsv[:, :, 1] * alpha
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Random hue
        if random.random() < 0.5:
            delta = random.uniform(-self.hue, self.hue) * 180
            hsv[:, :, 0] = hsv[:, :, 0] + delta
            hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)

        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        return rgb.astype(np.uint8), hag, label


class RandomGaussianBlur:
    """Random Gaussian blur for RGB only.

    Does NOT apply to HAG since it would corrupt geometric data.
    """
    def __init__(self, prob=0.5, kernel_size=5):
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, rgb, hag, label):
        if random.random() < self.prob:
            rgb = cv2.GaussianBlur(rgb, (self.kernel_size, self.kernel_size), 0)
        return rgb, hag, label


class Normalize:
    """Normalize RGB with ImageNet mean/std.

    HAG is already normalized in dataset loading (0-1 range).
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, rgb, hag, label):
        # Normalize RGB: (x - mean) / std
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std

        # HAG is already normalized to 0-1 in dataset, apply same normalization
        # This makes it consistent with RGB processing in the model
        if hag is not None:
            # HAG values are already 0-1, center around 0.5
            hag = (hag - 0.5) / 0.5  # Now in range [-1, 1]

        return rgb, hag, label


class TestTransform:
    """Test-time transform (normalize only, no augmentation).

    For sliding window inference at full resolution.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = Normalize(mean, std)

    def __call__(self, rgb, hag, label):
        return self.normalize(rgb, hag, label)


def get_train_transform(cfg):
    """Build training transforms from config."""
    aug_cfg = cfg.get('AUGMENTATION', {}).get('TRAIN', {})

    transforms = []

    # Random resize
    if aug_cfg.get('RANDOM_RESIZE', {}).get('ENABLE', True):
        scale = aug_cfg.get('RANDOM_RESIZE', {}).get('SCALE', [0.5, 2.0])
        transforms.append(RandomResize(scale_range=tuple(scale)))

    # Random crop
    if aug_cfg.get('RANDOM_CROP', {}).get('ENABLE', True):
        size = aug_cfg.get('RANDOM_CROP', {}).get('SIZE', [1024, 1024])
        ignore_label = cfg.get('DATASET', {}).get('IGNORE_LABEL', 255)
        transforms.append(RandomCrop(crop_size=tuple(size), ignore_label=ignore_label))

    # Random horizontal flip
    if aug_cfg.get('RANDOM_HFLIP', {}).get('ENABLE', True):
        prob = aug_cfg.get('RANDOM_HFLIP', {}).get('PROB', 0.5)
        transforms.append(RandomHorizontalFlip(prob=prob))

    # Photometric distortion (RGB only)
    if aug_cfg.get('PHOTOMETRIC', {}).get('ENABLE', True):
        photo_cfg = aug_cfg.get('PHOTOMETRIC', {})
        transforms.append(PhotoMetricDistortion(
            brightness=photo_cfg.get('BRIGHTNESS', 0.2),
            contrast=photo_cfg.get('CONTRAST', 0.2),
            saturation=photo_cfg.get('SATURATION', 0.2),
            hue=photo_cfg.get('HUE', 0.1)
        ))

    # Gaussian blur (RGB only, as per paper)
    if aug_cfg.get('GAUSSIAN_BLUR', {}).get('ENABLE', False):
        blur_cfg = aug_cfg.get('GAUSSIAN_BLUR', {})
        transforms.append(RandomGaussianBlur(
            prob=blur_cfg.get('PROB', 0.5),
            kernel_size=blur_cfg.get('KERNEL_SIZE', 5)
        ))

    # Normalize
    mean = cfg.get('DATASET', {}).get('MEAN', [0.485, 0.456, 0.406])
    std = cfg.get('DATASET', {}).get('STD', [0.229, 0.224, 0.225])
    transforms.append(Normalize(mean=tuple(mean), std=tuple(std)))

    return Compose(transforms)


def get_test_transform(cfg):
    """Build test transforms from config (normalize only)."""
    mean = cfg.get('DATASET', {}).get('MEAN', [0.485, 0.456, 0.406])
    std = cfg.get('DATASET', {}).get('STD', [0.229, 0.224, 0.225])
    return TestTransform(mean=tuple(mean), std=tuple(std))


# Test code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dummy data
    rgb = np.random.randint(0, 255, (2048, 2448, 3), dtype=np.uint8)
    hag = np.random.rand(2048, 2448, 3).astype(np.float32)
    label = np.random.randint(0, 19, (2048, 2448), dtype=np.uint8)

    print(f"Original shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  HAG: {hag.shape}")
    print(f"  Label: {label.shape}")

    # Test train transform
    train_transform = Compose([
        RandomResize(scale_range=(0.5, 2.0)),
        RandomCrop(crop_size=(1024, 1024)),
        RandomHorizontalFlip(prob=0.5),
        PhotoMetricDistortion(),
        Normalize()
    ])

    rgb_aug, hag_aug, label_aug = train_transform(rgb.copy(), hag.copy(), label.copy())

    print(f"\nAugmented shapes:")
    print(f"  RGB: {rgb_aug.shape}, dtype: {rgb_aug.dtype}, range: [{rgb_aug.min():.2f}, {rgb_aug.max():.2f}]")
    print(f"  HAG: {hag_aug.shape}, dtype: {hag_aug.dtype}, range: [{hag_aug.min():.2f}, {hag_aug.max():.2f}]")
    print(f"  Label: {label_aug.shape}, dtype: {label_aug.dtype}, unique: {len(np.unique(label_aug))}")

    print("\nAugmentation test passed!")
