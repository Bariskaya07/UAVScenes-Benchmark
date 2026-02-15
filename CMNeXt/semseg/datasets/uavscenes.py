"""
UAVScenes Dataset for Multi-Modal Semantic Segmentation (RGB + HAG)

Dataset structure:
- RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
- Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
- HAG: interval5_HAG/{scene}/{timestamp}.png

20 scenes total, ~24,126 frames (interval5 subset)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Label remapping: original cmap.py ID -> contiguous ID (0-18)
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
    20: 17,  # sedan
    24: 18,  # truck
}


def create_label_map():
    """Create NumPy LUT for O(1) label remapping.

    This is ~100x faster than Python loop-based remapping!
    """
    label_map = np.full(256, 255, dtype=np.uint8)  # default: ignore (255)
    for orig_id, new_id in LABEL_REMAP.items():
        label_map[orig_id] = new_id
    return label_map


class UAVScenes(Dataset):
    """UAVScenes Dataset for RGB + HAG semantic segmentation.

    Args:
        root: Path to UAVScenes data root
        split: 'train' or 'test'
        transform: Augmentation transforms
        modals: List of modalities ['img', 'hag']
    """

    CLASSES = [
        'background',       # 0  - static
        'roof',             # 1  - static
        'dirt_road',        # 2  - static
        'paved_road',       # 3  - static
        'river',            # 4  - static
        'pool',             # 5  - static
        'bridge',           # 6  - static
        'container',        # 7  - static
        'airstrip',         # 8  - static
        'traffic_barrier',  # 9  - static
        'green_field',      # 10 - static
        'wild_field',       # 11 - static
        'solar_panel',      # 12 - static
        'umbrella',         # 13 - static
        'transparent_roof', # 14 - static
        'car_park',         # 15 - static
        'paved_walk',       # 16 - static
        'sedan',            # 17 - DYNAMIC
        'truck'             # 18 - DYNAMIC
    ]

    PALETTE = [
        [0, 0, 0],         # background
        [119, 11, 32],     # roof
        [180, 165, 180],   # dirt_road
        [128, 64, 128],    # paved_road
        [173, 216, 230],   # river
        [0, 80, 100],      # pool
        [150, 100, 100],   # bridge
        [250, 170, 30],    # container
        [81, 0, 81],       # airstrip
        [102, 102, 156],   # traffic_barrier
        [107, 142, 35],    # green_field
        [210, 180, 140],   # wild_field
        [220, 220, 0],     # solar_panel
        [153, 153, 153],   # umbrella
        [0, 0, 90],        # transparent_roof
        [250, 170, 160],   # car_park
        [244, 35, 232],    # paved_walk
        [0, 0, 142],       # sedan
        [0, 0, 70]         # truck
    ]

    # Static classes: 0-16 (including background)
    # Dynamic classes: 17 (sedan), 18 (truck)
    STATIC_CLASSES = list(range(17))   # 0-16
    DYNAMIC_CLASSES = [17, 18]         # sedan, truck

    # Scene-based train/val/test split (NewSplit.md)
    # Train: 13 scenes, Val: 3 scenes, Test: 4 scenes
    TRAIN_SCENES = [
        # City (1 scene)
        'interval5_AMtown01',
        # Valley/Nature (1 scene)
        'interval5_AMvalley01',
        # Airport (3 scenes)
        'interval5_HKairport01', 'interval5_HKairport02', 'interval5_HKairport03',
        # Airport GNSS (3 scenes - including Evening)
        'interval5_HKairport_GNSS02', 'interval5_HKairport_GNSS03', 'interval5_HKairport_GNSS_Evening',
        # Island (3 scenes)
        'interval5_HKisland01', 'interval5_HKisland02', 'interval5_HKisland03',
        # Island GNSS (2 scenes)
        'interval5_HKisland_GNSS02', 'interval5_HKisland_GNSS03',
    ]

    # VALIDATION SET (3 scenes)
    VAL_SCENES = [
        'interval5_AMtown02',
        'interval5_AMvalley02',
        'interval5_HKisland_GNSS01',
    ]

    # TEST SET (4 scenes)
    TEST_SCENES = [
        'interval5_AMtown03',
        'interval5_AMvalley03',
        'interval5_HKairport_GNSS01',
        'interval5_HKisland_GNSS_Evening',
    ]

    def __init__(self, root, split='train', transform=None, modals=['img', 'hag'], aux_channels=3):
        """
        Args:
            root: Path to UAVScenes data root
            split: 'train', 'val', or 'test'
            transform: Augmentation transforms
            modals: List of modalities ['img', 'hag']
            aux_channels: Number of channels for auxiliary modality (1 or 3)
                         - 1: Native single-channel HAG (recommended for new models)
                         - 3: Stack HAG to 3 channels (backward compatibility)
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals
        self.aux_channels = aux_channels

        # Create label remapping LUT (fast O(1) lookup)
        self.label_map = create_label_map()

        # Select scenes based on split
        if split == 'train':
            self.scenes = self.TRAIN_SCENES
        elif split == 'val':
            self.scenes = self.VAL_SCENES
        else:
            self.scenes = self.TEST_SCENES

        # Build file list
        self.samples = self._load_samples()

        print(f"UAVScenes {split}: {len(self.samples)} samples from {len(self.scenes)} scenes (aux_channels={aux_channels})")

    def _load_samples(self):
        """Build list of (rgb_path, hag_path, label_path) tuples."""
        samples = []

        for scene in self.scenes:
            # RGB folder
            rgb_dir = self.root / 'interval5_CAM_LIDAR' / 'interval5_CAM_LIDAR' / scene / 'interval5_CAM'

            if not rgb_dir.exists():
                print(f"Warning: RGB directory not found: {rgb_dir}")
                continue

            # Get all RGB files
            rgb_files = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])

            for rgb_file in rgb_files:
                timestamp = rgb_file.stem  # e.g., '1658137057.641204937'

                # Construct paths for other modalities
                label_path = self.root / 'interval5_CAM_label' / 'interval5_CAM_label' / scene / 'interval5_CAM_label_id' / f'{timestamp}.png'
                hag_path = self.root / 'interval5_HAG_CSF' / scene / f'{timestamp}.png'

                # Only add if all files exist and HAG is not empty (0 bytes = corrupted)
                if label_path.exists() and hag_path.exists() and hag_path.stat().st_size > 0:
                    samples.append({
                        'rgb': str(rgb_file),
                        'label': str(label_path),
                        'hag': str(hag_path),
                        'scene': scene,
                        'timestamp': timestamp
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load RGB image (BGR -> RGB)
        rgb = cv2.imread(sample['rgb'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # [H, W, 3]

        # Load Label (grayscale, 8-bit)
        label = cv2.imread(sample['label'], cv2.IMREAD_UNCHANGED)  # [H, W]

        # Apply label remapping using NumPy LUT (O(1) - FAST!)
        label = self.label_map[label]  # This is ~100x faster than loop!

        # Load HAG if needed
        hag = None
        if 'hag' in self.modals:
            hag = self._load_hag(sample['hag'])  # [H, W, 3]

        # Apply transforms
        if self.transform is not None:
            if hag is not None:
                rgb, hag, label = self.transform(rgb, hag, label)
            else:
                rgb, label = self.transform(rgb, label)

        # Convert to tensors
        # RGB: [H, W, 3] -> [3, H, W], normalized
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))

        # HAG: [H, W, C] -> [C, H, W], already normalized (C=1 or 3 based on aux_channels)
        if hag is not None:
            hag = torch.from_numpy(hag.transpose(2, 0, 1).astype(np.float32))

        # Label: [H, W] -> long tensor
        label = torch.from_numpy(label.astype(np.int64))

        # Return format depends on modals
        if hag is not None:
            return [rgb, hag], label
        else:
            return rgb, label

    def _load_hag(self, filepath):
        """Load HAG from 16-bit PNG and convert to normalized format.

        HAG encoding: pixel = (HAG_meters * 1000) + 20000
        Normalization: divide by 50m (max observed ~45m)

        Returns:
            If aux_channels=1: [H, W, 1] single-channel HAG
            If aux_channels=3: [H, W, 3] stacked HAG (backward compatibility)
        """
        # Load 16-bit PNG
        hag_raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # uint16

        # Error handling for missing/corrupted files
        if hag_raw is None:
            raise FileNotFoundError(f"HAG file could not be loaded: {filepath}")

        # Convert to meters
        hag_meters = (hag_raw.astype(np.float32) - 20000) / 1000.0

        # Normalize to 0-1 range (50m max with safety margin)
        hag_normalized = np.clip(hag_meters / 50.0, 0, 1)

        # Return based on aux_channels configuration
        if self.aux_channels == 1:
            # Native single-channel HAG
            return hag_normalized[..., np.newaxis]  # [H, W, 1]
        else:
            # Stack to 3 channels (backward compatibility)
            return np.stack([hag_normalized] * 3, axis=-1)  # [H, W, 3]

    @staticmethod
    def get_class_colors():
        """Return class colors for visualization."""
        return np.array(UAVScenes.PALETTE, dtype=np.uint8)

    @staticmethod
    def decode_segmap(label, num_classes=19):
        """Convert label IDs to RGB visualization.

        Args:
            label: [H, W] label map with class IDs
            num_classes: Number of classes

        Returns:
            RGB image [H, W, 3] for visualization
        """
        colors = UAVScenes.get_class_colors()
        rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

        for cls_id in range(num_classes):
            mask = label == cls_id
            rgb[mask] = colors[cls_id]

        return rgb


# Test code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test dataset loading
    dataset = UAVScenes(
        root='/home/bariskaya/Projelerim/UAV/UAVScenesData',
        split='train',
        transform=None,
        modals=['img', 'hag']
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Classes: {len(UAVScenes.CLASSES)}")
    print(f"Static classes: {len(UAVScenes.STATIC_CLASSES)}")
    print(f"Dynamic classes: {len(UAVScenes.DYNAMIC_CLASSES)}")

    # Load a sample
    (rgb, hag), label = dataset[0]

    print(f"\nSample shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  HAG: {hag.shape}")
    print(f"  Label: {label.shape}")
    print(f"  Unique labels: {torch.unique(label).tolist()}")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # RGB
    axes[0].imshow(rgb.permute(1, 2, 0).numpy().astype(np.uint8))
    axes[0].set_title('RGB')
    axes[0].axis('off')

    # HAG
    axes[1].imshow(hag[0].numpy(), cmap='viridis')
    axes[1].set_title('HAG (Height Above Ground)')
    axes[1].axis('off')

    # Label
    label_rgb = UAVScenes.decode_segmap(label.numpy())
    axes[2].imshow(label_rgb)
    axes[2].set_title('Label (Segmentation)')
    axes[2].axis('off')

    # Label histogram
    unique, counts = torch.unique(label, return_counts=True)
    axes[3].bar(unique.numpy(), counts.numpy())
    axes[3].set_title('Label Distribution')
    axes[3].set_xlabel('Class ID')
    axes[3].set_ylabel('Pixel Count')

    plt.tight_layout()
    plt.savefig('uavscenes_sample.png', dpi=150)
    print("\nSample visualization saved to 'uavscenes_sample.png'")
