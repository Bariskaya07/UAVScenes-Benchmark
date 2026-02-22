"""
Data Augmentation Transforms for TokenFusion UAVScenes

Implements training and validation transforms that match
the fair comparison settings from CMNeXt and DFormerv2.

Training augmentations:
- Random horizontal flip
- Random scale (0.5 - 2.0)
- Random crop to 768x768
- Color jitter (optional)

Validation:
- No augmentation (sliding window at eval)
"""

import numpy as np
import cv2
import torch
from PIL import Image


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert numpy arrays to tensors."""

    def __init__(self, rgb_mean=None, rgb_std=None, hag_mean=None, hag_std=None):
        # ImageNet normalization for RGB
        self.rgb_mean = rgb_mean if rgb_mean is not None else [0.485, 0.456, 0.406]
        self.rgb_std = rgb_std if rgb_std is not None else [0.229, 0.224, 0.225]

        # Centered normalization for HAG (0-1 range -> centered)
        self.hag_mean = hag_mean if hag_mean is not None else [0.5, 0.5, 0.5]
        self.hag_std = hag_std if hag_std is not None else [0.5, 0.5, 0.5]

    def __call__(self, sample):
        rgb = sample['rgb']
        hag = sample['hag']
        label = sample['label']

        # Convert to float and normalize
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        # HAG is already normalized to [0, 1]
        hag = hag.astype(np.float32)
        hag = (hag - self.hag_mean) / self.hag_std

        # Convert to tensors (C, H, W)
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        hag = torch.from_numpy(hag.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        sample['rgb'] = rgb
        sample['hag'] = hag
        sample['label'] = label

        return sample


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['rgb'] = np.fliplr(sample['rgb']).copy()
            sample['hag'] = np.fliplr(sample['hag']).copy()
            sample['label'] = np.fliplr(sample['label']).copy()
        return sample


class RandomScale:
    """Random scale augmentation."""

    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, sample):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        h, w = sample['rgb'].shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize RGB
        sample['rgb'] = cv2.resize(
            sample['rgb'],
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize HAG
        sample['hag'] = cv2.resize(
            sample['hag'],
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize label (nearest neighbor)
        sample['label'] = cv2.resize(
            sample['label'],
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        return sample


class RandomCrop:
    """Random crop to fixed size."""

    def __init__(self, crop_size=768, ignore_label=255):
        self.crop_size = crop_size
        self.ignore_label = ignore_label

    def __call__(self, sample):
        h, w = sample['rgb'].shape[:2]
        crop_h, crop_w = self.crop_size, self.crop_size

        # If image smaller than crop size, pad it
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)

            sample['rgb'] = np.pad(
                sample['rgb'],
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            sample['hag'] = np.pad(
                sample['hag'],
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            sample['label'] = np.pad(
                sample['label'],
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=self.ignore_label
            )

            h, w = sample['rgb'].shape[:2]

        # Random crop position
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)

        # Crop
        sample['rgb'] = sample['rgb'][start_h:start_h + crop_h, start_w:start_w + crop_w]
        sample['hag'] = sample['hag'][start_h:start_h + crop_h, start_w:start_w + crop_w]
        sample['label'] = sample['label'][start_h:start_h + crop_h, start_w:start_w + crop_w]

        return sample


class CenterCrop:
    """Center crop to fixed size (for validation)."""

    def __init__(self, crop_size=768, ignore_label=255):
        self.crop_size = crop_size
        self.ignore_label = ignore_label

    def __call__(self, sample):
        h, w = sample['rgb'].shape[:2]
        crop_h, crop_w = self.crop_size, self.crop_size

        # If image smaller than crop size, pad it
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)

            sample['rgb'] = np.pad(
                sample['rgb'],
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            sample['hag'] = np.pad(
                sample['hag'],
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            sample['label'] = np.pad(
                sample['label'],
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=self.ignore_label
            )

            h, w = sample['rgb'].shape[:2]

        # Center crop position
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        # Crop
        sample['rgb'] = sample['rgb'][start_h:start_h + crop_h, start_w:start_w + crop_w]
        sample['hag'] = sample['hag'][start_h:start_h + crop_h, start_w:start_w + crop_w]
        sample['label'] = sample['label'][start_h:start_h + crop_h, start_w:start_w + crop_w]

        return sample


class Resize:
    """Resize to fixed size."""

    def __init__(self, size=768):
        self.size = size

    def __call__(self, sample):
        # Resize RGB
        sample['rgb'] = cv2.resize(
            sample['rgb'],
            (self.size, self.size),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize HAG
        sample['hag'] = cv2.resize(
            sample['hag'],
            (self.size, self.size),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize label (nearest neighbor)
        sample['label'] = cv2.resize(
            sample['label'],
            (self.size, self.size),
            interpolation=cv2.INTER_NEAREST
        )

        return sample


class ColorJitter:
    """Random color jitter for RGB only."""

    def __init__(self, p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        if np.random.random() >= self.p:
            return sample

        rgb = sample['rgb'].astype(np.float32)

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            rgb = rgb * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = rgb.mean()
            rgb = (rgb - mean) * factor + mean

        # Saturation / Hue in HSV space
        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

            if self.saturation > 0:
                sat_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
                hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)

            if self.hue > 0:
                hue_shift = np.random.uniform(-self.hue, self.hue) * 180.0
                hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0

            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Clip values
        rgb = np.clip(rgb, 0, 255)
        sample['rgb'] = rgb.astype(np.uint8)

        return sample


class RandomGaussianBlur:
    """Random gaussian blur for RGB only."""

    def __init__(self, p=0.2, kernel_size=3):
        self.p = p
        self.kernel_size = int(kernel_size)

    def __call__(self, sample):
        if np.random.random() >= self.p:
            return sample

        k = self.kernel_size
        if k % 2 == 0:
            k += 1
        k = max(1, k)

        sample['rgb'] = cv2.GaussianBlur(sample['rgb'], (k, k), 0)
        return sample


class TrainTransform:
    """
    Training transform pipeline.

    Applies:
    1. Random horizontal flip (p=0.5)
    2. Random scale (0.5 - 2.0)
    3. Random crop to 768x768
    4. Color jitter (RGB only)
    5. Gaussian blur (RGB only)
    6. ToTensor with normalization
    """

    def __init__(
        self,
        crop_size=768,
        scale_range=(0.5, 2.0),
        flip_prob=0.5,
        color_jitter_p=0.2,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        gaussian_blur_p=0.2,
        gaussian_blur_kernel=3,
        rgb_mean=None,
        rgb_std=None,
        hag_mean=None,
        hag_std=None,
        ignore_label=255
    ):
        self.transform = Compose([
            RandomHorizontalFlip(p=flip_prob),
            RandomScale(scale_range=scale_range),
            RandomCrop(crop_size=crop_size, ignore_label=ignore_label),
            ColorJitter(
                p=color_jitter_p,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            RandomGaussianBlur(p=gaussian_blur_p, kernel_size=gaussian_blur_kernel),
            ToTensor(rgb_mean=rgb_mean, rgb_std=rgb_std, hag_mean=hag_mean, hag_std=hag_std)
        ])

    def __call__(self, sample):
        return self.transform(sample)


class ValTransform:
    """
    Validation transform pipeline.

    Resize to training size for fast validation (CPU resize before GPU transfer).
    """

    def __init__(
        self,
        rgb_mean=None,
        rgb_std=None,
        hag_mean=None,
        hag_std=None,
        size=768
    ):
        self.transform = Compose([
            Resize(size=size),  # Resize on CPU for fast validation
            ToTensor(rgb_mean=rgb_mean, rgb_std=rgb_std, hag_mean=hag_mean, hag_std=hag_std)
        ])

    def __call__(self, sample):
        return self.transform(sample)


class TestTransform:
    """Test-time transform pipeline.

    IMPORTANT: No resize.
    This is intended for full-resolution sliding-window evaluation.
    """

    def __init__(
        self,
        rgb_mean=None,
        rgb_std=None,
        hag_mean=None,
        hag_std=None,
    ):
        self.transform = Compose([
            ToTensor(rgb_mean=rgb_mean, rgb_std=rgb_std, hag_mean=hag_mean, hag_std=hag_std)
        ])

    def __call__(self, sample):
        return self.transform(sample)


def get_train_transform(cfg=None):
    """Get training transform with config."""
    if cfg is None:
        return TrainTransform()

    return TrainTransform(
        crop_size=getattr(cfg, 'crop_size', 768),
        scale_range=getattr(cfg, 'scale_range', (0.5, 2.0)),
        flip_prob=getattr(cfg, 'flip_prob', 0.5),
        rgb_mean=getattr(cfg, 'rgb_mean', None),
        rgb_std=getattr(cfg, 'rgb_std', None),
        hag_mean=getattr(cfg, 'hag_mean', None),
        hag_std=getattr(cfg, 'hag_std', None),
        ignore_label=getattr(cfg, 'ignore_label', 255)
    )


def get_val_transform(cfg=None):
    """Get validation transform with config."""
    if cfg is None:
        return ValTransform()

    return ValTransform(
        rgb_mean=getattr(cfg, 'rgb_mean', None),
        rgb_std=getattr(cfg, 'rgb_std', None),
        hag_mean=getattr(cfg, 'hag_mean', None),
        hag_std=getattr(cfg, 'hag_std', None)
    )
