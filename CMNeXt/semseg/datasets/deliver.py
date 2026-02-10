"""
DELIVER Dataset for CMNeXt Training

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
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, List
import glob


class DELIVER(Dataset):
    """
    DELIVER dataset for CMNeXt training.

    25 semantic classes for urban driving scenarios.
    Supports RGB + LiDAR (Range + Intensity) multi-modal segmentation.
    """

    CLASSES = [
        "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road",
        "SideWalk", "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground",
        "Bridge", "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
        "Water", "Terrain", "TwoWheeler", "Bus", "Truck"
    ]

    PALETTE = torch.tensor([
        [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
        [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
        [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
        [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160], [170, 120, 50],
        [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100], [0, 0, 70],
    ])

    def __init__(
        self,
        root: str = 'data/DELIVER',
        split: str = 'train',
        transform=None,
        modals: List[str] = ['img', 'lidar'],
        case=None,
        aux_channels: int = 3,
    ) -> None:
        """
        Args:
            root: Root directory of DELIVER dataset
            split: 'train', 'val', or 'test'
            transform: Transform function
            modals: List of modalities to use ['img', 'lidar', 'depth', 'event']
            case: Weather/condition filter (cloud, fog, night, rain, sun, etc.)
            aux_channels: Number of channels for LiDAR (2 for native, 3 for backward compat)
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

        # Filter by case/condition if specified
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

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        rgb_path = str(self.files[index])

        # Build paths for other modalities
        lidar_path = rgb_path.replace('/img', '/lidar').replace('_rgb', '_lidar')
        depth_path = rgb_path.replace('/img', '/hha').replace('_rgb', '_depth')
        event_path = rgb_path.replace('/img', '/event').replace('_rgb', '_event')
        lbl_path = rgb_path.replace('/img', '/semantic').replace('_rgb', '_semantic')

        # Load RGB (BGR -> RGB, normalize to [0, 1])
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        # Load Label
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label file not found: {lbl_path}")
        label[label == 255] = 0
        label = label - 1  # DELIVER labels are 1-indexed
        label = np.clip(label, 0, 24).astype(np.uint8)

        # Build sample dict
        sample = {'image': rgb, 'label': label}

        # Load LiDAR if needed
        if 'lidar' in self.modals:
            lidar = self._load_lidar(lidar_path)
            sample['lidar'] = lidar

        # Load Depth if needed
        if 'depth' in self.modals:
            depth = self._load_aux_image(depth_path)
            sample['depth'] = depth

        # Load Event if needed
        if 'event' in self.modals:
            event = self._load_aux_image(event_path)
            # Resize event to match RGB size
            event = cv2.resize(event, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            sample['event'] = event

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        # Convert to tensors
        label = torch.from_numpy(sample['label']).long()

        # Build output list based on modals
        sample_list = []
        for modal in self.modals:
            if modal == 'img':
                img = sample['image']
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                sample_list.append(img)
            elif modal in sample:
                aux = sample[modal]
                if isinstance(aux, np.ndarray):
                    aux = torch.from_numpy(aux.transpose(2, 0, 1)).float()
                sample_list.append(aux)

        return sample_list, label

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

        # Handle channel dimension
        if len(lidar.shape) == 2:
            # Single channel - stack to required channels
            if self.aux_channels == 1:
                lidar = lidar[..., np.newaxis]
            elif self.aux_channels == 2:
                lidar = np.stack([lidar, lidar], axis=-1)  # Placeholder for Range+Intensity
            else:
                lidar = np.stack([lidar, lidar, lidar], axis=-1)

        return lidar

    def _load_aux_image(self, filepath):
        """Load auxiliary modality image."""
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Auxiliary file not found: {filepath}")

        # Normalize to [0, 1]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        return img


if __name__ == '__main__':
    # Test dataset loading
    print("Testing DELIVER dataset...")
    try:
        dataset = DELIVER(root='data/DELIVER', split='train', modals=['img', 'lidar'])
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            sample, label = dataset[0]
            print(f"Sample shapes: {[s.shape for s in sample]}")
            print(f"Label shape: {label.shape}")
    except Exception as e:
        print(f"Dataset test skipped: {e}")
