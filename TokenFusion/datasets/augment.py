"""
Multi-Modal Augmentation Pipeline for TokenFusion UAVScenes
Standardized augmentations matching CMNeXt for fair comparison.

Photometric augmentations are applied only to RGB (not HAG).
"""

import cv2
import numpy as np
import random


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomResize:
    """Random resize with scale range.
    Applies the same resize to all modalities.
    """
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, sample):
        rgb = sample['rgb']
        hag = sample['hag']
        label = sample['label']

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

        sample['rgb'] = rgb
        sample['hag'] = hag
        sample['label'] = label
        return sample


class RandomCrop:
    """Random crop to fixed size.
    If image is smaller than crop size, pads with zeros (RGB/HAG) and ignore_label (Label).
    """
    def __init__(self, crop_size=(768, 768), ignore_label=255):
        self.crop_size = crop_size  # (height, width)
        self.ignore_label = ignore_label

    def __call__(self, sample):
        rgb = sample['rgb']
        hag = sample['hag']
        label = sample['label']

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

        sample['rgb'] = rgb
        sample['hag'] = hag
        sample['label'] = label
        return sample


class RandomHorizontalFlip:
    """Random horizontal flip.
    Same flip applied to all modalities.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['rgb'] = np.fliplr(sample['rgb']).copy()
            if sample['hag'] is not None:
                sample['hag'] = np.fliplr(sample['hag']).copy()
            sample['label'] = np.fliplr(sample['label']).copy()
        return sample


class PhotoMetricDistortion:
    """Photometric distortion for RGB only.
    Does NOT apply to HAG since it's geometric data, not color.

    Standardized settings (CMNeXt):
    - brightness: 0.2
    - contrast: 0.2
    - saturation: 0.2
    - hue: 0.1
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        rgb = sample['rgb'].astype(np.float32)

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
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        sample['rgb'] = rgb
        return sample


class RandomGaussianBlur:
    """Random Gaussian blur for RGB only.
    Does NOT apply to HAG since it would corrupt geometric data.

    Standardized settings (CMNeXt):
    - prob: 0.5
    - kernel_size: 5
    """
    def __init__(self, prob=0.5, kernel_size=5):
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['rgb'] = cv2.GaussianBlur(
                sample['rgb'],
                (self.kernel_size, self.kernel_size),
                0
            )
        return sample


class Normalize:
    """Normalize RGB with ImageNet mean/std.
    HAG is already normalized in dataset loading (0-1 range).
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        # Normalize RGB: (x - mean) / std
        rgb = sample['rgb'].astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        sample['rgb'] = rgb

        # HAG: center around 0.5 -> range [-1, 1]
        if sample['hag'] is not None:
            hag = sample['hag'].astype(np.float32)
            hag = (hag - 0.5) / 0.5
            sample['hag'] = hag

        return sample


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    def __call__(self, sample):
        import torch

        # RGB: HWC -> CHW
        rgb = sample['rgb']
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        sample['rgb'] = rgb

        # HAG: HWC -> CHW
        hag = sample['hag']
        if hag is not None and isinstance(hag, np.ndarray):
            hag = torch.from_numpy(hag.transpose(2, 0, 1).copy()).float()
        sample['hag'] = hag

        # Label: HW
        label = sample['label']
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label.copy()).long()
        sample['label'] = label

        return sample


def get_train_transform(cfg=None):
    """
    Build training transforms matching CMNeXt augmentation settings.

    Standardized augmentations:
    - Random resize: scale [0.5, 2.0]
    - Random crop: 768x768
    - Random horizontal flip: p=0.5
    - Photometric distortion: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    - Gaussian blur: p=0.5, kernel=5
    - Normalize: ImageNet mean/std
    """
    transforms = [
        RandomResize(scale_range=(0.5, 2.0)),
        RandomCrop(crop_size=(768, 768), ignore_label=255),
        RandomHorizontalFlip(prob=0.5),
        PhotoMetricDistortion(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomGaussianBlur(prob=0.5, kernel_size=5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ]
    return Compose(transforms)


def get_val_transform(cfg=None):
    """
    Build validation/test transforms (normalize only, no augmentation).
    """
    transforms = [
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ]
    return Compose(transforms)
