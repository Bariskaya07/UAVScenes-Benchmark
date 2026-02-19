"""
UAVScenes Dataset for Mul-VMamba Training

Dataset: UAVScenes RGB + HAG (Height Above Ground) Multi-Modal Semantic Segmentation
Target: Mul-VMamba with VMamba-T backbone for fair comparison benchmark

HAG NORMALIZATION:
- Default: 50m (same as CMNeXt, DFormerV2 for fair comparison)
- Formula: normalized_hag = np.clip(hag_meters / max_height, 0, 1)

Data Structure:
- RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
- Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
- HAG: interval5_HAG_CSF/{scene}/{timestamp}.png (16-bit PNG, encoded as (meters * 1000) + 20000)
"""

import os
import cv2
import torch
import random
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, List


# UAVScenes label remapping: 26 original classes -> 19 used classes
LABEL_REMAP = {
    0: 0,    # background
    1: 1,    # roof
    2: 2,    # dirt_road
    3: 3,    # paved_road
    4: 4,    # river
    5: 5,    # pool
    6: 6,    # bridge
    9: 7,    # container
    10: 8,   # airstrip
    11: 9,   # traffic_barrier
    13: 10,  # green_field
    14: 11,  # wild_field
    15: 12,  # solar_panel
    16: 13,  # umbrella
    17: 14,  # transparent_roof
    18: 15,  # car_park
    19: 16,  # paved_walk
    20: 17,  # sedan (dynamic)
    24: 18,  # truck (dynamic)
}

# NewSplit.md - Train/Val/Test split for fair comparison
TRAIN_SCENES = [
    'interval5_AMtown01',
    'interval5_AMvalley01',
    'interval5_HKairport01',
    'interval5_HKairport02',
    'interval5_HKairport03',
    'interval5_HKairport_GNSS02',
    'interval5_HKairport_GNSS03',
    'interval5_HKairport_GNSS_Evening',  # Night teacher
    'interval5_HKisland01',
    'interval5_HKisland02',
    'interval5_HKisland03',
    'interval5_HKisland_GNSS02',
    'interval5_HKisland_GNSS03',
]

VAL_SCENES = [
    'interval5_AMtown02',
    'interval5_AMvalley02',
    'interval5_HKisland_GNSS01',
]

TEST_SCENES = [
    'interval5_AMtown03',               # City / Day
    'interval5_AMvalley03',             # Valley / Day
    'interval5_HKairport_GNSS01',       # Airport / Day
    'interval5_HKisland_GNSS_Evening',  # Island / Evening
]

CLASS_NAMES = [
    'background', 'roof', 'dirt_road', 'paved_road', 'river', 'pool',
    'bridge', 'container', 'airstrip', 'traffic_barrier', 'green_field',
    'wild_field', 'solar_panel', 'umbrella', 'transparent_roof', 'car_park',
    'paved_walk', 'sedan', 'truck'
]


def remap_label(label):
    """
    Remap UAVScenes labels from 26 classes to 19 classes.
    Unmapped labels become 255 (ignore).
    """
    remapped = np.full_like(label, 255, dtype=np.uint8)
    for orig_id, new_id in LABEL_REMAP.items():
        remapped[label == orig_id] = new_id
    return remapped


def load_hag(hag_path, max_height=50.0, aux_channels=3):
    """
    Load and normalize HAG (Height Above Ground) data.

    HAG encoding: pixel = (HAG_meters * 1000) + 20000
    Invalid pixels have raw value = 0 (decode to -20m)

    Args:
        hag_path: Path to 16-bit PNG HAG file
        max_height: Maximum height for normalization (default 50m for fair comparison)
        aux_channels: Number of output channels (1 for single channel, 3 for stacked)

    Returns:
        Normalized HAG as float32 array in [0, 1] range
        Shape: (H, W, 1) if aux_channels=1, (H, W, 3) if aux_channels=3
    """
    # Load 16-bit HAG
    hag_raw = cv2.imread(hag_path, cv2.IMREAD_UNCHANGED)

    if hag_raw is None:
        raise FileNotFoundError(f"HAG file not found: {hag_path}")

    # Convert from encoded format to meters
    # Format: encoded = (meters * 1000) + 20000
    # So: meters = (encoded - 20000) / 1000.0
    hag_meters = (hag_raw.astype(np.float32) - 20000.0) / 1000.0

    # Handle invalid pixels (raw = 0 decodes to -20m)
    # Set invalid/negative values to 0
    hag_meters = np.maximum(hag_meters, 0)

    # Normalize to [0, 1] range
    normalized_hag = np.clip(hag_meters / max_height, 0, 1)

    # Return with requested number of channels
    if aux_channels == 1:
        return normalized_hag[..., np.newaxis]  # (H, W, 1)
    else:
        # Stack to 3 channels for backward compatibility
        return np.stack([normalized_hag, normalized_hag, normalized_hag], axis=2)


class UAVScenes(Dataset):
    """
    UAVScenes dataset for Mul-VMamba training.

    Supports RGB + HAG (Height Above Ground) multi-modal segmentation.
    Returns format compatible with this repo's evaluation/training scripts:
    (sample_list, label) where sample_list = [image_tensor, hag_tensor]

    19 classes after remapping from 26 original classes.
    """

    CLASSES = CLASS_NAMES

    PALETTE = torch.tensor([
        [0, 0, 0],        # 0: background - black
        [180, 120, 120],  # 1: roof - brownish
        [139, 69, 19],    # 2: dirt_road - saddle brown
        [128, 128, 128],  # 3: paved_road - gray
        [0, 0, 255],      # 4: river - blue
        [0, 191, 255],    # 5: pool - deep sky blue
        [105, 105, 105],  # 6: bridge - dim gray
        [255, 165, 0],    # 7: container - orange
        [220, 220, 220],  # 8: airstrip - light gray
        [255, 0, 0],      # 9: traffic_barrier - red
        [0, 255, 0],      # 10: green_field - green
        [154, 205, 50],   # 11: wild_field - yellow green
        [75, 0, 130],     # 12: solar_panel - indigo
        [255, 20, 147],   # 13: umbrella - deep pink
        [176, 224, 230],  # 14: transparent_roof - powder blue
        [169, 169, 169],  # 15: car_park - dark gray
        [210, 180, 140],  # 16: paved_walk - tan
        [255, 255, 0],    # 17: sedan - yellow
        [255, 140, 0],    # 18: truck - dark orange
    ])

    def __init__(
        self,
        root: str = 'data/UAVScenes',
        split: str = 'train',
        transform=None,
        modals: List[str] = ['image', 'hag'],
        case=None,
        hag_max_meters: float = 50.0,
        aux_channels: int = 3,
    ) -> None:
        """
        Args:
            root: Root directory of UAVScenes dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform (not used, we use internal transforms)
            modals: List of modalities to use ['image', 'hag']
            case: Not used, for compatibility
            hag_max_meters: Max height for HAG normalization (default 50m)
            aux_channels: Number of channels for HAG (1 for single, 3 for stacked)
        """
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.hag_max_meters = hag_max_meters
        self.aux_channels = aux_channels

        # Crop sizes for training (768x768 for fair comparison with CMNeXt/DFormerV2)
        self.base_size = 768
        self.crop_size = 768

        # Get file list based on split
        self.files = self._get_file_names(split)

        if not self.files:
            raise Exception(f"No images found for split '{split}' in {root}")
        print(f"[UAVScenes] Found {len(self.files)} {split} images.")
        print(f"[UAVScenes] HAG max height: {self.hag_max_meters}m")
        print(f"[UAVScenes] Aux channels: {self.aux_channels}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        scene, timestamp = self.files[index]

        # Build file paths
        # RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
        rgb_path = os.path.join(
            self.root,
            "interval5_CAM_LIDAR",
            "interval5_CAM_LIDAR",
            scene,
            "interval5_CAM",
            f"{timestamp}.jpg"
        )

        # Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
        lbl_path = os.path.join(
            self.root,
            "interval5_CAM_label",
            "interval5_CAM_label",
            scene,
            "interval5_CAM_label_id",
            f"{timestamp}.png"
        )

        # HAG: interval5_HAG_CSF/{scene}/{timestamp}.png
        hag_path = os.path.join(
            self.root,
            "interval5_HAG_CSF",
            scene,
            f"{timestamp}.png"
        )

        # Load RGB (BGR -> RGB, normalize to [0, 1])
        _img = cv2.imread(rgb_path, -1)
        if _img is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        _img = _img[:, :, ::-1]  # BGR to RGB
        _img = _img.astype(np.float32) / 255.0

        # Load and remap label
        _target = cv2.imread(lbl_path, -1)
        if _target is None:
            raise FileNotFoundError(f"Label file not found: {lbl_path}")
        _target = remap_label(_target)

        # Load HAG (normalized [0, 1], configurable channels)
        _hag = load_hag(hag_path, max_height=self.hag_max_meters, aux_channels=self.aux_channels)

        # Build sample dict (similar to MCubeS structure)
        sample = {
            'image': _img,
            'label': _target,
            'hag': _hag,
        }

        # Apply transforms based on split
        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == "val":
            sample = self.transform_val(sample)
        else:  # test
            # IMPORTANT: for fair benchmarking with CMNeXt/DFormerV2,
            # do NOT resize/crop at dataset level. Sliding-window inference
            # should run on the original full resolution.
            sample = self.transform_test(sample)

        label = sample['label'].long()

        # Return format: [modality tensors], label
        sample_list = [sample[k] for k in self.modals]
        return sample_list, label

    def transform_tr(self, sample):
        """Training transforms: RandomHorizontalFlip, RandomScaleCrop, Normalize, ToTensor"""
        composed_transforms = Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        """Validation/Test transforms: FixScaleCrop, Normalize, ToTensor"""
        composed_transforms = Compose([
            FixScaleCrop(crop_size=self.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_test(self, sample):
        """Test transforms: Normalize, ToTensor (NO resize/crop)."""
        composed_transforms = Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor(),
        ])
        return composed_transforms(sample)

    def _get_file_names(self, split_name):
        """Get list of (scene, timestamp) tuples for the given split."""
        if split_name == 'train':
            scenes = TRAIN_SCENES
        elif split_name == 'val':
            scenes = VAL_SCENES
        else:  # test
            scenes = TEST_SCENES

        file_names = []

        for scene in scenes:
            # RGB directory path
            rgb_dir = os.path.join(
                self.root,
                "interval5_CAM_LIDAR",
                "interval5_CAM_LIDAR",
                scene,
                "interval5_CAM"
            )

            if not os.path.exists(rgb_dir):
                print(f"[UAVScenes] Warning: RGB directory not found: {rgb_dir}")
                continue

            # List all RGB images in the scene
            for img_file in sorted(os.listdir(rgb_dir)):
                if img_file.endswith('.jpg') and not img_file.endswith(':Zone.Identifier'):
                    timestamp = img_file.replace('.jpg', '')
                    # Check HAG file exists and is not empty
                    hag_path = os.path.join(
                        self.root,
                        "interval5_HAG_CSF",
                        scene,
                        f"{timestamp}.png"
                    )
                    if os.path.exists(hag_path) and os.path.getsize(hag_path) > 0:
                        file_names.append((scene, timestamp))

        return file_names


# ============================================================================
# Transform classes (adapted from MCubeS for 2-modality case)
# ============================================================================

class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Normalize:
    """Normalize RGB image with mean and standard deviation."""
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img -= self.mean
        img /= self.std

        return {
            'image': img,
            'label': sample['label'],
            'hag': sample['hag'],
        }


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        hag = sample['hag']

        # numpy: H x W x C -> torch: C x H x W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        hag = np.array(hag).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        hag = torch.from_numpy(hag).float()

        return {
            'image': img,
            'label': mask,
            'hag': hag,
        }


class RandomHorizontalFlip:
    """Random horizontal flip for data augmentation."""
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = sample['image'][:, ::-1].copy()
            sample['label'] = sample['label'][:, ::-1].copy()
            sample['hag'] = sample['hag'][:, ::-1].copy()
        return sample


class RandomGaussianBlur:
    """Random Gaussian blur for data augmentation."""
    def __call__(self, sample):
        if random.random() < 0.5:
            radius = random.random()
            sample['image'] = cv2.GaussianBlur(sample['image'], (0, 0), radius)
            # Don't blur HAG - it's geometric data
        return sample


class RandomScaleCrop:
    """Random scale and crop for data augmentation."""
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        hag = sample['hag']

        # Random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        h, w = img.shape[:2]

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # Resize
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        hag = cv2.resize(hag, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # Pad if needed
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0

            img_padded = np.zeros((oh + padh, ow + padw, 3), dtype=img.dtype)
            img_padded[:oh, :ow] = img
            img = img_padded

            mask_padded = np.full((oh + padh, ow + padw), self.fill, dtype=mask.dtype)
            mask_padded[:oh, :ow] = mask
            mask = mask_padded

            hag_padded = np.zeros((oh + padh, ow + padw, 3), dtype=hag.dtype)
            hag_padded[:oh, :ow] = hag
            hag = hag_padded

            oh, ow = oh + padh, ow + padw

        # Random crop
        x1 = random.randint(0, max(0, ow - self.crop_size))
        y1 = random.randint(0, max(0, oh - self.crop_size))

        img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
        mask = mask[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
        hag = hag[y1:y1 + self.crop_size, x1:x1 + self.crop_size]

        return {
            'image': img,
            'label': mask,
            'hag': hag,
        }


class FixScaleCrop:
    """Fixed scale and center crop for validation/testing."""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        hag = sample['hag']

        h, w = img.shape[:2]

        # Scale to make short edge equal to crop_size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        # Resize
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        hag = cv2.resize(hag, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # Center crop
        x1 = int(round((ow - self.crop_size) / 2.))
        y1 = int(round((oh - self.crop_size) / 2.))

        img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
        mask = mask[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
        hag = hag[y1:y1 + self.crop_size, x1:x1 + self.crop_size]

        return {
            'image': img,
            'label': mask,
            'hag': hag,
        }


if __name__ == '__main__':
    # Test dataset loading
    from torch.utils.data import DataLoader

    print("Testing UAVScenes dataset...")
    dataset = UAVScenes(root='data/UAVScenes', split='train')
    loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

    for i, (sample, lbl, path) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  - Image shape: {sample[0].shape}")
        print(f"  - HAG shape: {sample[1].shape}")
        print(f"  - Label shape: {lbl.shape}")
        print(f"  - Unique labels: {torch.unique(lbl)}")
        if i >= 2:
            break
    print("Dataset test passed!")
