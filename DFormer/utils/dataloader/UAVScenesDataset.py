"""
UAVScenes Dataset for DFormerv2 Training

Dataset: UAVScenes RGB + HAG (Height Above Ground) Multi-Modal Semantic Segmentation
Paper: UAVScenes dataset paper
Target: DFormerv2 with Geometry Self-Attention

HAG NORMALIZATION:
- Default: 50m (same as CMNeXt for fair comparison)
- Configurable via hag_max_meters parameter
- Formula: normalized_hag = np.clip(hag_meters / max_height, 0, 1)

Data Structure:
- RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
- Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
- HAG: interval5_HAG/{scene}/{timestamp}.png (16-bit PNG, encoded as (meters * 1000) + 20000)
"""

import os
import cv2
import torch
import numpy as np
from torch.utils import data
from pathlib import Path


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

# Train/Val/Test split (NewSplit.md)
# Train: 13 scenes, Val: 3 scenes, Test: 4 scenes
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


def load_hag(hag_path, max_height=150.0):
    """
    Load and normalize HAG (Height Above Ground) data.

    HAG encoding: pixel = (HAG_meters * 1000) + 20000
    Invalid pixels have raw value = 0 (decode to -20m)

    Args:
        hag_path: Path to 16-bit PNG HAG file
        max_height: Maximum height for normalization

    Returns:
        Normalized HAG in [0, 255] range as uint8 (for compatibility with DFormer preprocessing)
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

    # Convert to [0, 255] uint8 for DFormer preprocessing compatibility
    # The preprocessing pipeline expects uint8 input and will divide by 255
    hag_uint8 = (normalized_hag * 255).astype(np.uint8)

    return hag_uint8


class UAVScenesDataset(data.Dataset):
    """
    UAVScenes dataset for DFormerv2 training.

    Supports RGB + HAG (Height Above Ground) multi-modal segmentation.
    HAG is used as geometry prior in DFormerv2's Geometry Self-Attention.
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        """
        Args:
            setting: Dictionary containing dataset configuration
            split_name: 'train' or 'val'
            preprocess: Preprocessing function (TrainPre or ValPre)
            file_length: Optional length override for training

        Setting keys:
            - dataset_path: Path to UAVScenes data root
            - aux_channels: Number of channels for HAG (1 or 3, default 1)
                           - 1: Native single-channel (DFormer model supports this natively)
                           - 3: Stack to 3 channels (backward compatibility)
        """
        super(UAVScenesDataset, self).__init__()

        self._split_name = split_name
        self._dataset_path = setting["dataset_path"]
        self._transform_gt = setting.get("transform_gt", False)
        self._file_length = file_length
        self.preprocess = preprocess
        self.class_names = setting.get("class_names", CLASS_NAMES)
        self.backbone = setting.get("backbone", "DFormerv2_B")
        self.dataset_name = "UAVScenes"

        # HAG normalization max height (default 50m to match CMNeXt)
        self.hag_max_meters = setting.get("hag_max_meters", 50.0)

        # Auxiliary modality channels (default 1 for DFormer - it natively supports 1-channel)
        self.aux_channels = setting.get("aux_channels", 1)

        # Get file list based on split
        self._file_names = self._get_file_names(split_name)

        print(f"[UAVScenes] Loaded {len(self._file_names)} samples for {split_name}")
        print(f"[UAVScenes] HAG max height: {self.hag_max_meters}m, aux_channels: {self.aux_channels}")

    def _get_file_names(self, split_name):
        """
        Get list of sample paths based on train/val split.

        Returns list of tuples: (scene, timestamp)

        UAVScenes data structure:
        - RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
        - Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
        - HAG: interval5_HAG/{scene}/{timestamp}.png
        """
        if split_name == "train":
            scenes = TRAIN_SCENES
        elif split_name == "val":
            scenes = VAL_SCENES
        else:
            scenes = TEST_SCENES

        file_names = []

        for scene in scenes:
            # RGB directory path (note the nested structure)
            rgb_dir = os.path.join(
                self._dataset_path,
                "interval5_CAM_LIDAR",
                "interval5_CAM_LIDAR",
                scene,
                "interval5_CAM"
            )

            if not os.path.exists(rgb_dir):
                print(f"[UAVScenes] Warning: RGB directory not found: {rgb_dir}")
                continue

            # List all RGB images in the scene (filter out Zone.Identifier files)
            for img_file in sorted(os.listdir(rgb_dir)):
                if img_file.endswith('.jpg') and not img_file.endswith(':Zone.Identifier'):
                    timestamp = img_file.replace('.jpg', '')
                    # Check HAG file exists and is not empty
                    hag_path = os.path.join(
                        self._dataset_path,
                        "interval5_HAG_CSF",
                        scene,
                        f"{timestamp}.png"
                    )
                    if os.path.exists(hag_path) and os.path.getsize(hag_path) > 0:
                        file_names.append((scene, timestamp))

        return file_names

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item = self._construct_new_file_names(self._file_length)[index]
        else:
            item = self._file_names[index]

        scene, timestamp = item

        # Build file paths (note the nested structure of UAVScenes)
        # RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
        rgb_path = os.path.join(
            self._dataset_path,
            "interval5_CAM_LIDAR",
            "interval5_CAM_LIDAR",
            scene,
            "interval5_CAM",
            f"{timestamp}.jpg"
        )

        # Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
        label_path = os.path.join(
            self._dataset_path,
            "interval5_CAM_label",
            "interval5_CAM_label",
            scene,
            "interval5_CAM_label_id",
            f"{timestamp}.png"
        )

        # HAG: interval5_HAG_CSF/{scene}/{timestamp}.png
        hag_path = os.path.join(
            self._dataset_path,
            "interval5_HAG_CSF",
            scene,
            f"{timestamp}.png"
        )

        # Load RGB (as BGR for OpenCV, will be converted by preprocess)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")

        # Load and remap label
        gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Label file not found: {label_path}")
        gt = remap_label(gt)

        # Load and normalize HAG
        # Returns uint8 in [0, 255] for preprocessing compatibility
        hag = load_hag(hag_path, max_height=self.hag_max_meters)

        # Convert HAG based on aux_channels configuration
        if self.aux_channels == 1:
            # Native single-channel (DFormer model supports this natively)
            modal_x = hag[..., np.newaxis]  # [H, W, 1]
        else:
            # Stack to 3 channels (backward compatibility)
            modal_x = cv2.merge([hag, hag, hag])  # [H, W, 3]

        # Apply preprocessing (augmentation for training)
        if self.preprocess is not None:
            rgb, gt, modal_x = self.preprocess(rgb, gt, modal_x)

        # Convert to tensors
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()

        output_dict = dict(
            data=rgb,
            label=gt,
            modal_x=modal_x,
            fn=rgb_path,
            n=len(self._file_names)
        )

        return output_dict

    def _construct_new_file_names(self, length):
        """Construct file list for training with specified length."""
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @classmethod
    def get_class_colors(cls):
        """Get color map for UAVScenes classes (for visualization)."""
        # Using a distinct color palette for 19 classes
        colors = [
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
        ]
        return colors


# For compatibility with DFormer's data loading functions
def get_uavscenes_path(
    dataset_name,
    _rgb_path,
    _rgb_format,
    _x_path,
    _x_format,
    _gt_path,
    _gt_format,
    x_modal,
    item_name,
):
    """
    Path getter for UAVScenes dataset.
    Compatible with DFormer's RGBXDataset.get_path() interface.

    item_name format: "scene/timestamp"
    """
    scene, timestamp = item_name.split("/")

    rgb_path = os.path.join(
        _rgb_path,
        scene,
        "interval5_CAM",
        timestamp + _rgb_format
    )

    gt_path = os.path.join(
        _gt_path,
        scene,
        "interval5_CAM_label_id",
        timestamp + _gt_format
    )

    d_path = os.path.join(
        _x_path,
        scene,
        timestamp + _x_format
    )

    path_result = {"rgb_path": rgb_path, "gt_path": gt_path}
    for modal in x_modal:
        path_result[modal + "_path"] = d_path

    return path_result
