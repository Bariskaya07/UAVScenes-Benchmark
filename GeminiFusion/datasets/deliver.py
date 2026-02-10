"""
DELIVER Dataset for GeminiFusion Training

Dataset: DELIVER RGB + LiDAR Multi-Modal Semantic Segmentation
Target: Cross-dataset evaluation with configurable input projection

LiDAR Channels:
- aux_channels=2: Range + Intensity (native DELIVER format)
- aux_channels=3: Stacked 3-channel for backward compatibility

Data Structure:
- RGB: img/{weather}/{split}/{scene}/{frame}_rgb.png
- LiDAR: lidar/{weather}/{split}/{scene}/{frame}_lidar.png
- Semantic: semantic/{weather}/{split}/{scene}/{frame}_semantic.png
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import glob


# DELIVER 25 class names
CLASS_NAMES = [
    "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road",
    "SideWalk", "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground",
    "Bridge", "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "TwoWheeler", "Bus", "Truck"
]

# Color palette for visualization
PALETTE = [
    [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
    [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
    [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
    [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160], [170, 120, 50],
    [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100], [0, 0, 70],
]


class DELIVERDataset(Dataset):
    """
    DELIVER dataset for GeminiFusion training.

    25 semantic classes for urban driving scenarios.
    Supports RGB + LiDAR multi-modal segmentation.
    Compatible with GeminiFusion's data loading interface.
    """

    CLASSES = CLASS_NAMES
    PALETTE = PALETTE

    def __init__(
        self,
        root: str = 'data/DELIVER',
        split: str = 'train',
        transform=None,
        modals: List[str] = ['rgb', 'lidar'],
        case=None,
        aux_channels: int = 3,
    ) -> None:
        """
        Args:
            root: Root directory of DELIVER dataset
            split: 'train', 'val', or 'test'
            transform: Transform function
            modals: List of modalities to use
            case: Weather/condition filter
            aux_channels: Number of channels for LiDAR
        """
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.root = root
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.aux_channels = aux_channels

        # Find all RGB files
        self.files = sorted(glob.glob(os.path.join(root, 'img', '*', split, '*', '*.png')))

        # Filter by case if specified
        if case is not None:
            valid_cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur',
                          'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
            assert case in valid_cases, f"Case '{case}' not in {valid_cases}"
            self.files = [f for f in self.files if case in f]

        if not self.files:
            raise Exception(f"No images found in {root}")

        print(f"[DELIVER] Found {len(self.files)} {split} images.")
        print(f"[DELIVER] Aux channels: {self.aux_channels}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict:
        rgb_path = str(self.files[index])

        # Build paths
        lidar_path = rgb_path.replace('/img', '/lidar').replace('_rgb', '_lidar')
        lbl_path = rgb_path.replace('/img', '/semantic').replace('_rgb', '_semantic')

        # Load RGB (BGR -> RGB)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load Label
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label file not found: {lbl_path}")
        label[label == 255] = 0
        label = label - 1  # DELIVER labels are 1-indexed
        label = np.clip(label, 0, 24).astype(np.uint8)

        # Load LiDAR
        lidar = self._load_lidar(lidar_path)

        # Build sample dict (GeminiFusion format - same as TokenFusion)
        sample = {
            'rgb': rgb,
            'lidar': lidar,  # Using 'lidar' for DELIVER (not 'hag')
            'label': label,
        }

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_lidar(self, filepath):
        """Load LiDAR data with configurable channels."""
        lidar = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if lidar is None:
            raise FileNotFoundError(f"LiDAR file not found: {filepath}")

        # Normalize to [0, 1]
        if lidar.dtype == np.uint16:
            lidar = lidar.astype(np.float32) / 65535.0
        else:
            lidar = lidar.astype(np.float32) / 255.0

        # Handle channels
        if len(lidar.shape) == 2:
            if self.aux_channels == 1:
                lidar = lidar[..., np.newaxis]
            elif self.aux_channels == 2:
                lidar = np.stack([lidar, lidar], axis=-1)
            else:
                lidar = np.stack([lidar, lidar, lidar], axis=-1)

        return lidar


def get_deliver_dataset(cfg, split='train'):
    """Factory function to create DELIVER dataset."""
    root = getattr(cfg, 'data_root', 'data/DELIVER')
    aux_channels = getattr(cfg, 'aux_channels', 3)
    modals = getattr(cfg, 'modals', ['rgb', 'lidar'])

    return DELIVERDataset(
        root=root,
        split=split,
        transform=None,  # Use external transform
        modals=modals,
        aux_channels=aux_channels,
    )
