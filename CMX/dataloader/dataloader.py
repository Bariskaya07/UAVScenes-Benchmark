"""
UAVScenes dataloader for CMX.
Matches CMNeXt training pipeline:
- Photometric distortion (RGB only): brightness, contrast, saturation, hue
- Gaussian blur (RGB only)
- Random mirror, random scale, random crop
- RGB: ImageNet normalization
- HAG: normalized to [-1, 1] (from [0, 1])
"""

import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize


def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scale_range):
    """Random scale with continuous uniform range (matching CMNeXt RandomResize)."""
    scale = random.uniform(scale_range[0], scale_range[-1])
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return rgb, gt, modal_x, scale


def photometric_distortion(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Photometric distortion for RGB only (matching CMNeXt augment.py).
    Input: uint8 RGB [H, W, 3]
    Output: uint8 RGB [H, W, 3]
    """
    rgb = rgb.astype(np.float32)

    # Random brightness
    if random.random() < 0.5:
        delta = random.uniform(-brightness, brightness) * 255
        rgb = np.clip(rgb + delta, 0, 255)

    # Random contrast
    if random.random() < 0.5:
        alpha = random.uniform(1 - contrast, 1 + contrast)
        rgb = np.clip(rgb * alpha, 0, 255)

    # Convert to HSV for saturation and hue
    rgb_uint8 = rgb.astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Random saturation
    if random.random() < 0.5:
        alpha = random.uniform(1 - saturation, 1 + saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)

    # Random hue
    if random.random() < 0.5:
        delta = random.uniform(-hue, hue) * 180
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + delta, 180)

    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def random_gaussian_blur(rgb, prob=0.5, kernel_size=5):
    """Random Gaussian blur for RGB only (matching CMNeXt augment.py)."""
    if random.random() < prob:
        rgb = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
    return rgb


class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.use_photometric = getattr(config, 'use_photometric', True)
        self.use_gaussian_blur = getattr(config, 'use_gaussian_blur', True)
        self.gaussian_blur_prob = getattr(config, 'gaussian_blur_prob', 0.2)
        self.gaussian_blur_kernel = getattr(config, 'gaussian_blur_kernel', 3)

    def __call__(self, rgb, gt, modal_x):
        # Exact CMNeXt order: resize → crop → hflip → photometric → blur → normalize
        # 1. Random resize (continuous uniform scale)
        if config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, config.train_scale_array)

        # 2. Random crop
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        # 3. Random horizontal flip
        p_rgb, p_gt, p_modal_x = random_mirror(p_rgb, p_gt, p_modal_x)

        # 4. Photometric distortion (RGB only)
        if self.use_photometric:
            p_rgb = photometric_distortion(p_rgb)

        # 5. Gaussian blur (RGB only)
        if self.use_gaussian_blur:
            p_rgb = random_gaussian_blur(
                p_rgb,
                prob=self.gaussian_blur_prob,
                kernel_size=self.gaussian_blur_kernel,
            )

        # 6. Normalize (last step, matching CMNeXt)
        p_rgb = normalize(p_rgb, self.norm_mean, self.norm_std)
        p_modal_x = (p_modal_x - 0.5) / 0.5

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_gt, p_modal_x


class ValPre(object):
    """Eval-time preprocessing: pass-through (no normalization, no augmentation).
    Normalization is handled by SegEvaluator.process_image_rgbX in eval.py.
    """
    def __init__(self, resize_to=None):
        self.resize_to = resize_to

    def __call__(self, rgb, gt, modal_x):
        if self.resize_to is not None:
            h, w = self.resize_to
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            modal_x = cv2.resize(modal_x, (w, h), interpolation=cv2.INTER_LINEAR)
        return rgb, gt, modal_x



def get_train_loader(engine, dataset):
    data_setting = {
        'data_root': config.dataset_path,
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
