"""
UAVScenes Dataset for Sigma (Siamese Mamba Network) Training

Dataset: UAVScenes RGB + HAG (Height Above Ground) Multi-Modal Semantic Segmentation
Target: Fair comparison with CMNeXt and DFormerV2 benchmarks

Data Structure:
- RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
- Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
- HAG: interval5_HAG_CSF/{scene}/{timestamp}.png (16-bit PNG, encoded as (meters * 1000) + 20000)

HAG NORMALIZATION:
- Default: 50m max (same as CMNeXt for fair comparison)
- Formula: normalized_hag = np.clip(hag_meters / max_height, 0, 1)
"""

import os
import cv2
import torch
import numpy as np
import torch.utils.data as data


# ==============================================================================
# Train/Val/Test Scene Splits (NewSplit.md)
# Train: 13 scenes, Val: 3 scenes, Test: 4 scenes
# ==============================================================================
TRAIN_SCENES = [
    'interval5_AMtown01',
    'interval5_AMvalley01',
    'interval5_HKairport01', 'interval5_HKairport02', 'interval5_HKairport03',
    'interval5_HKairport_GNSS02', 'interval5_HKairport_GNSS03', 'interval5_HKairport_GNSS_Evening',
    'interval5_HKisland01', 'interval5_HKisland02', 'interval5_HKisland03',
    'interval5_HKisland_GNSS02', 'interval5_HKisland_GNSS03',
]

VAL_SCENES = [
    'interval5_AMtown02',
    'interval5_AMvalley02',
    'interval5_HKisland_GNSS01',
]

TEST_SCENES = [
    'interval5_AMtown03',
    'interval5_AMvalley03',
    'interval5_HKairport_GNSS01',
    'interval5_HKisland_GNSS_Evening',
]

# ==============================================================================
# Class Names (19 classes after remapping)
# ==============================================================================
CLASS_NAMES = [
    'background',       # 0
    'roof',             # 1
    'dirt_road',        # 2
    'paved_road',       # 3
    'river',            # 4
    'pool',             # 5
    'bridge',           # 6
    'container',        # 7
    'airstrip',         # 8
    'traffic_barrier',  # 9
    'green_field',      # 10
    'wild_field',       # 11
    'solar_panel',      # 12
    'umbrella',         # 13
    'transparent_roof', # 14
    'car_park',         # 15
    'paved_walk',       # 16
    'sedan',            # 17 (dynamic class)
    'truck',            # 18 (dynamic class)
]

# ==============================================================================
# Label Remapping (26 → 19 classes, same as CMNeXt/DFormerV2)
# Original labels 0, 18, 19, 20, 22, 23, 25 → 255 (ignore)
# ==============================================================================
LABEL_REMAP = {
    0: 255,   # unlabeled → ignore
    1: 0,     # roof → background
    2: 1,     # dirt_road
    3: 2,     # paved_road
    4: 3,     # river
    5: 4,     # pool
    6: 5,     # bridge
    7: 6,     # container
    8: 7,     # airstrip
    9: 8,     # traffic_barrier
    10: 9,    # green_field
    11: 10,   # wild_field
    12: 11,   # solar_panel
    13: 12,   # umbrella
    14: 13,   # transparent_roof
    15: 14,   # car_park
    16: 15,   # paved_walk
    17: 16,   # (merged to paved_walk or other)
    18: 255,  # ignore
    19: 255,  # ignore
    20: 255,  # ignore
    21: 17,   # sedan
    22: 255,  # ignore
    23: 255,  # ignore
    24: 18,   # truck
    25: 255,  # ignore
}


def remap_label(label):
    """Remap UAVScenes labels from 26 classes to 19 classes."""
    remapped = np.full_like(label, 255, dtype=np.uint8)
    for orig_id, new_id in LABEL_REMAP.items():
        remapped[label == orig_id] = new_id
    return remapped


def load_hag(hag_path, max_height=50.0):
    """
    Load and normalize HAG (Height Above Ground) data.

    HAG encoding: pixel = (HAG_meters * 1000) + 20000
    Invalid pixels have raw value = 0 (decode to -20m)

    Args:
        hag_path: Path to 16-bit PNG HAG file
        max_height: Maximum height for normalization (default 50m)

    Returns:
        Normalized HAG in [0, 255] range as uint8 (for compatibility with Sigma preprocessing)
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

    # Convert to [0, 255] uint8 for Sigma preprocessing compatibility
    hag_uint8 = (normalized_hag * 255).astype(np.uint8)

    return hag_uint8


class UAVScenesDataset(data.Dataset):
    """
    UAVScenes dataset for Sigma training.

    Supports RGB + HAG (Height Above Ground) multi-modal segmentation.
    Compatible with Sigma's Siamese Mamba architecture.
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        """
        Args:
            setting: Dict containing dataset paths and configs
            split_name: 'train' or 'val'
            preprocess: Preprocessing function
            file_length: Optional fixed dataset length
        """
        super(UAVScenesDataset, self).__init__()

        self._split_name = split_name
        self._dataset_path = setting.get('dataset_path', setting.get('rgb_root', ''))

        # For compatibility with Sigma's data loading
        self._rgb_path = setting.get('rgb_root', '')
        self._rgb_format = setting.get('rgb_format', '.jpg')
        self._gt_path = setting.get('gt_root', '')
        self._gt_format = setting.get('gt_format', '.png')
        self._transform_gt = setting.get('transform_gt', False)
        self._x_path = setting.get('x_root', '')
        self._x_format = setting.get('x_format', '.png')
        self._x_single_channel = setting.get('x_single_channel', True)

        # HAG normalization max height
        self.hag_max_meters = setting.get('hag_max_meters', 50.0)

        # Auxiliary channels (1 for UAVScenes HAG, 2 for DELIVER LiDAR)
        self.aux_channels = setting.get('aux_channels', 3)

        # Class names
        self.class_names = setting.get('class_names', CLASS_NAMES)

        # Get file list
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

        print(f"[UAVScenes] Loaded {len(self._file_names)} samples for {split_name}")
        print(f"[UAVScenes] HAG max height: {self.hag_max_meters}m")
        print(f"[UAVScenes] Aux channels: {self.aux_channels}")

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        # Unpack scene and timestamp
        scene, timestamp = item_name

        # Construct paths
        # RGB: RGB/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
        rgb_path = os.path.join(
            self._rgb_path,
            'interval5_CAM_LIDAR',
            scene,
            'interval5_CAM',
            f"{timestamp}{self._rgb_format}"
        )

        # Label: Label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
        gt_path = os.path.join(
            self._gt_path,
            'interval5_CAM_label',
            scene,
            'interval5_CAM_label_id',
            f"{timestamp}{self._gt_format}"
        )

        # HAG: HAG/{scene}/{timestamp}.png
        hag_path = os.path.join(
            self._x_path,
            scene,
            f"{timestamp}{self._x_format}"
        )

        # Load RGB
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
        if rgb is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")

        # Load and remap label
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if gt is None:
            raise FileNotFoundError(f"Label file not found: {gt_path}")
        gt = remap_label(gt)

        # Load and normalize HAG
        hag = load_hag(hag_path, max_height=self.hag_max_meters)

        # Convert HAG to required channels
        # aux_channels=1: Single channel HAG [H, W, 1]
        # aux_channels=3: Stacked 3-channel for backward compatibility [H, W, 3]
        if self.aux_channels == 1:
            x = hag[..., np.newaxis]  # [H, W, 1]
        else:
            x = cv2.merge([hag, hag, hag])  # [H, W, 3]

        # Apply preprocessing
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        # Convert to tensors for training
        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(
            data=rgb,
            label=gt,
            modal_x=x,
            fn=f"{scene}/{timestamp}",
            n=len(self._file_names)
        )

        return output_dict

    def _get_file_names(self, split_name):
        """
        Get list of sample paths based on train/val split.

        Returns list of tuples: (scene, timestamp)
        """
        assert split_name in ['train', 'val', 'test']
        if split_name == 'train':
            scenes = TRAIN_SCENES
        elif split_name == 'val':
            scenes = VAL_SCENES
        else:
            scenes = TEST_SCENES

        file_names = []

        for scene in scenes:
            # RGB directory path
            rgb_dir = os.path.join(
                self._rgb_path,
                'interval5_CAM_LIDAR',
                scene,
                'interval5_CAM'
            )

            if not os.path.exists(rgb_dir):
                print(f"[UAVScenes] Warning: RGB directory not found: {rgb_dir}")
                continue

            # Get all jpg files in the directory
            for filename in sorted(os.listdir(rgb_dir)):
                if filename.endswith(self._rgb_format):
                    # Extract timestamp (remove extension)
                    timestamp = filename.replace(self._rgb_format, '')
                    file_names.append((scene, timestamp))

        return file_names

    def _construct_new_file_names(self, length):
        """Construct file names for fixed length dataset."""
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        """Open image file."""
        img = cv2.imread(filepath, mode)
        if img is not None and dtype is not None:
            img = np.array(img, dtype=dtype)
        return img

    @classmethod
    def get_class_colors(*args):
        """Generate class colors for visualization."""
        def uint82bin(n, count=8):
            """Returns the binary of integer n, count refers to amount of bits."""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
