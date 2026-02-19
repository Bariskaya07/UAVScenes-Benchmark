import cv2
import torch
import numpy as np
from torch.utils import data
import random

# from config import config
# from train import config
from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)


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


def gaussian_blur(rgb, prob=0.5, kernel_size=5):
    """
    Random Gaussian blur for RGB only (standardized with CMNeXt).
    Does NOT apply to modal_x (HAG) since it would corrupt geometric data.
    """
    if random.random() < prob:
        rgb = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
    return rgb


class TrainPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign
        # Augmentation settings (standardized with CMNeXt)
        self.use_photometric = getattr(config, 'use_photometric', True)
        self.use_gaussian_blur = getattr(config, 'use_gaussian_blur', True)

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        # Photometric distortion (RGB only, standardized with CMNeXt)
        if self.use_photometric:
            rgb = photometric_distortion(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        # Gaussian blur (RGB only, standardized with CMNeXt)
        if self.use_gaussian_blur:
            rgb = gaussian_blur(rgb, prob=0.5, kernel_size=5)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])  # [0.5,0.5,0.5]
        else:
            modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        # return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        # p_rgb = p_rgb
        # p_modal_x = p_modal_x

        return p_rgb, p_gt, p_modal_x


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class ValPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None, resize: bool = True):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign
        self.resize = resize

    def __call__(self, rgb, gt, modal_x):
        # pad to 730*531
        if self.config.pad:
            rgb = cv2.copyMakeBorder(
                rgb,
                0,
                531 - rgb.shape[0],
                0,
                730 - rgb.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0),
            )
            gt = cv2.copyMakeBorder(
                gt,
                0,
                531 - gt.shape[0],
                0,
                730 - gt.shape[1],
                cv2.BORDER_CONSTANT,
                value=(255,),
            )
            modal_x = cv2.copyMakeBorder(
                modal_x,
                0,
                531 - modal_x.shape[0],
                0,
                730 - modal_x.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0),
            )

        # Resize for fast validation (CPU resize before GPU transfer)
        # IMPORTANT: for final test sliding-window evaluation, keep full resolution.
        if self.resize:
            rgb = cv2.resize(
                rgb,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            gt = cv2.resize(
                gt,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            modal_x = cv2.resize(
                modal_x,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
        return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)
        # return rgb, gt, modal_x


def get_train_loader(engine, dataset, config):
    # Handle UAVScenes dataset differently (uses scene-based loading)
    if config.dataset_name == "UAVScenes":
        data_setting = {
            "dataset_path": config.dataset_path,
            "transform_gt": config.gt_transform,
            "class_names": config.class_names,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
            "hag_max_meters": getattr(config, 'hag_max_meters', 150.0),
        }
    else:
        data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "class_names": config.class_names,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
        }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return train_loader, train_sampler


def get_val_loader(engine, dataset, config, val_batch_size=1):
    # Handle UAVScenes dataset differently (uses scene-based loading)
    if config.dataset_name == "UAVScenes":
        data_setting = {
            "dataset_path": config.dataset_path,
            "transform_gt": config.gt_transform,
            "class_names": config.class_names,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
            "hag_max_meters": getattr(config, 'hag_max_meters', 150.0),
        }
    else:
        data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "class_names": config.class_names,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
        }
    val_preprocess = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config, resize=True)

    val_dataset = dataset(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = val_batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = val_batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=val_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return val_loader, val_sampler


def get_test_loader(engine, dataset, config, test_batch_size=1):
    """Get test dataloader for final evaluation."""
    # Handle UAVScenes dataset differently (uses scene-based loading)
    if config.dataset_name == "UAVScenes":
        data_setting = {
            "dataset_path": config.dataset_path,
            "transform_gt": config.gt_transform,
            "class_names": config.class_names,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
            "hag_max_meters": getattr(config, 'hag_max_meters', 150.0),
        }
    else:
        data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "dataset_name": config.dataset_name,
            "backbone": config.backbone,
        }
    test_preprocess = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config, resize=False)

    test_dataset = dataset(data_setting, "test", test_preprocess)

    test_sampler = None
    is_shuffle = False
    batch_size = test_batch_size

    if engine.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        batch_size = test_batch_size // engine.world_size
        is_shuffle = False

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=test_sampler,
    )

    return test_loader, test_sampler
