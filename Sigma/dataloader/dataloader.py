import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize


def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale


def photometric_distortion(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Photometric distortion for RGB only (standardized with CMNeXt).
    Does NOT apply to modal_x (HAG) since it's geometric data.
    """
    rgb = rgb.astype(np.float32)

    # Random brightness
    if random.random() < 0.5:
        delta = random.uniform(-brightness, brightness) * 255
        rgb = rgb + delta
        rgb = np.clip(rgb, 0, 255)

    # Random contrast
    if random.random() < 0.5:
        alpha = random.uniform(1 - contrast, 1 + contrast)
        rgb = rgb * alpha
        rgb = np.clip(rgb, 0, 255)

    # Convert to HSV for saturation and hue adjustments
    rgb_uint8 = rgb.astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Random saturation
    if random.random() < 0.5:
        alpha = random.uniform(1 - saturation, 1 + saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * alpha
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # Random hue
    if random.random() < 0.5:
        delta = random.uniform(-hue, hue) * 180
        hsv[:, :, 0] = hsv[:, :, 0] + delta
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)

    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def gaussian_blur(rgb, prob=0.2, kernel_size=3):
    """
    Random Gaussian blur for RGB only (standardized with CMNeXt).
    Does NOT apply to modal_x (HAG) since it would corrupt geometric data.
    """
    if random.random() < prob:
        rgb = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
    return rgb


class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config
        # Augmentation settings (standardized with CMNeXt)
        self.use_photometric = getattr(config, 'use_photometric', True)
        self.use_gaussian_blur = getattr(config, 'use_gaussian_blur', True)
        self.gaussian_blur_prob = getattr(config, 'gaussian_blur_prob', 0.2)
        self.gaussian_blur_kernel = getattr(config, 'gaussian_blur_kernel', 3)

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        # Photometric distortion (RGB only, standardized with CMNeXt)
        if self.use_photometric:
            rgb = photometric_distortion(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        # Gaussian blur (RGB only, standardized with CMNeXt)
        if self.use_gaussian_blur:
            rgb = gaussian_blur(rgb, prob=self.gaussian_blur_prob, kernel_size=self.gaussian_blur_kernel)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x

class ValPre(object):
    def __init__(self, config=None, resize=True):
        self.config = config
        self.resize = resize
        # Default to 768x768 if no config
        self.image_height = 768
        self.image_width = 768
        if config is not None:
            self.image_height = getattr(config, 'image_height', 768)
            self.image_width = getattr(config, 'image_width', 768)

    def __call__(self, rgb, gt, modal_x):
        if self.resize:
            # Resize for fast validation (CPU resize before GPU transfer)
            rgb = cv2.resize(rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            modal_x = cv2.resize(modal_x, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        return rgb, gt, modal_x

def get_train_loader(engine, dataset, config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                    'dataset_path': config.dataset_path if hasattr(config, 'dataset_path') else '',
                    'hag_max_meters': config.hag_max_meters if hasattr(config, 'hag_max_meters') else 50.0}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

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
