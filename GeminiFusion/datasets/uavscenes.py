"""
UAVScenes Dataset for GeminiFusion

RGB + HAG (Height Above Ground) multi-modal semantic segmentation dataset.
Compatible with fair comparison settings from CMNeXt, DFormerv2, and TokenFusion.

Dataset Structure:
    UAVScenesData/
    ├── interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg  (RGB)
    ├── interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png  (Label)
    └── interval5_HAG/{scene}/{timestamp}.png  (HAG - 16-bit PNG)
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


def _rgb_dir_candidates(data_root: str, scene: str):
    return (
        os.path.join(data_root, "interval5_CAM_LIDAR", "interval5_CAM_LIDAR", scene, "interval5_CAM"),
        os.path.join(data_root, "interval5_CAM_LIDAR", scene, "interval5_CAM"),
    )


def _looks_like_uavscenes_root(data_root: str, probe_scene: str) -> bool:
    return any(os.path.isdir(p) for p in _rgb_dir_candidates(data_root, probe_scene))


def _resolve_uavscenes_root(data_root: str, probe_scene: str) -> str:
    """Resolve common dataset-root nesting mistakes.

    Users often pass a parent folder that contains another UAVScenesData folder, or
    the data is nested one level deeper after extraction.
    """
    if not data_root:
        return data_root

    data_root = os.path.abspath(os.path.expanduser(data_root))

    # If the provided path doesn't exist (very common typo), try a few standard locations
    # under the current user's home.
    if not os.path.exists(data_root):
        home = os.path.abspath(os.path.expanduser("~"))
        for candidate in (
            os.path.join(home, "data", "UAVScenesData"),
            os.path.join(home, "datasets", "UAVScenesData"),
            os.path.join(home, "UAVScenesData"),
        ):
            if _looks_like_uavscenes_root(candidate, probe_scene):
                print(f"[UAVScenes] Resolved missing data_root '{data_root}' -> '{candidate}'")
                return candidate

    # Fast path: already correct
    if _looks_like_uavscenes_root(data_root, probe_scene):
        return data_root

    # Common extra nesting patterns
    for candidate in (
        os.path.join(data_root, "UAVScenesData"),
        os.path.join(data_root, "UAVScenesData", "UAVScenesData"),
    ):
        if _looks_like_uavscenes_root(candidate, probe_scene):
            print(f"[UAVScenes] Resolved data_root '{data_root}' -> '{candidate}'")
            return candidate

    # Heuristic search: find an 'interval5_CAM_LIDAR' folder within a small depth
    # and treat its parent as the dataset root.
    max_depth = 4
    base_depth = data_root.rstrip(os.sep).count(os.sep)
    try:
        for current_root, dirnames, _filenames in os.walk(data_root, topdown=True):
            depth = current_root.rstrip(os.sep).count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
                continue

            if "interval5_CAM_LIDAR" in dirnames:
                candidate = current_root
                if _looks_like_uavscenes_root(candidate, probe_scene):
                    print(f"[UAVScenes] Resolved data_root '{data_root}' -> '{candidate}'")
                    return candidate

    except OSError:
        # Permissions / broken mount: keep the original and let caller error out.
        return data_root

    return data_root


# Scene splits (NewSplit.md)
# Train: 13 scenes, Val: 3 scenes, Test: 4 scenes
TRAIN_SCENES = [
    "interval5_AMtown01",
    "interval5_AMvalley01",
    "interval5_HKairport01", "interval5_HKairport02", "interval5_HKairport03",
    "interval5_HKairport_GNSS02", "interval5_HKairport_GNSS03", "interval5_HKairport_GNSS_Evening",
    "interval5_HKisland01", "interval5_HKisland02", "interval5_HKisland03",
    "interval5_HKisland_GNSS02", "interval5_HKisland_GNSS03",
]  # 13 scenes

VAL_SCENES = [
    "interval5_AMtown02",
    "interval5_AMvalley02",
    "interval5_HKisland_GNSS01",
]  # 3 scenes

TEST_SCENES = [
    "interval5_AMtown03",
    "interval5_AMvalley03",
    "interval5_HKairport_GNSS01",
    "interval5_HKisland_GNSS_Evening",
]  # 4 scenes

# Label mapping: Original UAVScenes labels -> 19 contiguous classes
# Based on cmap.py - IDs 7, 8, 12, 21, 22, 23, 25 are EMPTY (unused)
# This mapping must match CMNeXt exactly for fair comparison!
LABEL_MAPPING = {
    0: 0,      # background
    1: 1,      # roof
    2: 2,      # dirt_road (dirt_motor_road)
    3: 3,      # paved_road (paved_motor_road)
    4: 4,      # river
    5: 5,      # pool
    6: 6,      # bridge
    7: 255,    # EMPTY -> ignore
    8: 255,    # EMPTY -> ignore
    9: 7,      # container
    10: 8,     # airstrip
    11: 9,     # traffic_barrier
    12: 255,   # EMPTY -> ignore
    13: 10,    # green_field
    14: 11,    # wild_field
    15: 12,    # solar_panel (solar_board)
    16: 13,    # umbrella
    17: 14,    # transparent_roof
    18: 15,    # car_park
    19: 16,    # paved_walk
    20: 17,    # sedan (dynamic)
    21: 255,   # EMPTY -> ignore
    22: 255,   # EMPTY -> ignore
    23: 255,   # EMPTY -> ignore
    24: 18,    # truck (dynamic)
    25: 255,   # EMPTY -> ignore
}

# Class names for 19 classes
CLASS_NAMES = [
    "background",       # 0
    "roof",             # 1
    "dirt_road",        # 2
    "paved_road",       # 3
    "river",            # 4
    "pool",             # 5
    "bridge",           # 6
    "container",        # 7
    "airstrip",         # 8
    "traffic_barrier",  # 9
    "green_field",      # 10
    "wild_field",       # 11
    "solar_panel",      # 12
    "umbrella",         # 13
    "transparent_roof", # 14
    "car_park",         # 15
    "paved_walk",       # 16
    "sedan",            # 17 (dynamic)
    "truck",            # 18 (dynamic)
]


def load_hag(hag_path, max_height=50.0, aux_channels=3):
    """
    Load HAG (Height Above Ground) data from 16-bit PNG.

    Encoding: pixel = (HAG_meters * 1000) + 20000
    Decoding: HAG_meters = (pixel - 20000) / 1000.0

    Args:
        hag_path: Path to HAG PNG file
        max_height: Maximum height for normalization (50m for fair comparison)
        aux_channels: Number of output channels (1 or 3)
                     - 1: Native single-channel HAG
                     - 3: Stack to 3 channels (backward compatibility)

    Returns:
        Normalized HAG array in range [0, 1], shape (H, W, aux_channels)
    """
    # Read 16-bit PNG
    hag_raw = cv2.imread(hag_path, cv2.IMREAD_UNCHANGED)

    if hag_raw is None:
        raise FileNotFoundError(f"HAG file not found: {hag_path}")

    # Decode HAG values (meters)
    hag_meters = (hag_raw.astype(np.float32) - 20000.0) / 1000.0

    # Clip negative values (ground level)
    hag_meters = np.maximum(hag_meters, 0)

    # Normalize to [0, 1] with max_height cap
    normalized_hag = np.clip(hag_meters / max_height, 0, 1)

    # Return based on aux_channels configuration
    if aux_channels == 1:
        return normalized_hag[..., np.newaxis]  # [H, W, 1]
    else:
        return np.stack([normalized_hag] * 3, axis=-1)  # [H, W, 3]


def remap_labels(label):
    """
    Remap original UAVScenes labels to 19 classes.

    Args:
        label: Original label array

    Returns:
        Remapped label array
    """
    remapped = np.full_like(label, 255, dtype=np.uint8)
    for orig_id, new_id in LABEL_MAPPING.items():
        remapped[label == orig_id] = new_id
    return remapped


class UAVScenesDataset(Dataset):
    """
    UAVScenes RGB + HAG Dataset for GeminiFusion.

    Returns samples in format compatible with GeminiFusion's training loop:
    - 'rgb': RGB image as numpy array (H, W, 3)
    - 'depth': HAG data as numpy array (H, W, 3) - named 'depth' for compatibility
    - 'mask': Segmentation mask as numpy array (H, W)

    Args:
        data_root: Root directory of UAVScenes dataset
        split: 'train' or 'test'
        transform: Optional transform to apply
        hag_max_height: Maximum HAG height for normalization (default: 50.0m)
        aux_channels: Number of channels for HAG (1 or 3)
                     - 1: Native single-channel HAG
                     - 3: Stack to 3 channels (backward compatibility)
    """

    def __init__(
        self,
        data_root,
        split='train',
        transform=None,
        hag_max_height=50.0,
        aux_channels=3
    ):
        super().__init__()

        probe_scene = TRAIN_SCENES[0]
        resolved_root = _resolve_uavscenes_root(data_root, probe_scene)
        self.data_root = resolved_root
        self.split = split
        self.transform = transform
        self.hag_max_height = hag_max_height
        self.aux_channels = aux_channels
        self.stage = split  # For compatibility with GeminiFusion's set_stage()

        # Select scenes based on split
        if split == 'train':
            scenes = TRAIN_SCENES
        elif split == 'val':
            scenes = VAL_SCENES
        else:
            scenes = TEST_SCENES

        # Build file list
        self.samples = self._build_file_list(scenes)

        print(f"UAVScenes {split}: {len(self.samples)} samples from {len(scenes)} scenes (aux_channels={aux_channels})")

    def _build_file_list(self, scenes):
        """Build list of (rgb_path, hag_path, label_path) tuples."""
        samples = []

        def first_existing_dir(*candidates):
            for candidate in candidates:
                if os.path.isdir(candidate):
                    return candidate
            return None

        # Fail fast (and avoid per-scene spam) if the dataset root is clearly wrong.
        has_cam_lidar = os.path.isdir(os.path.join(self.data_root, "interval5_CAM_LIDAR"))
        has_cam_label = os.path.isdir(os.path.join(self.data_root, "interval5_CAM_label"))
        has_hag = (
            os.path.isdir(os.path.join(self.data_root, "interval5_HAG_CSF"))
            or os.path.isdir(os.path.join(self.data_root, "interval5_HAG"))
        )
        if not has_cam_lidar:
            print(
                "[UAVScenes] Error: 'interval5_CAM_LIDAR' folder not found under "
                f"data_root='{self.data_root}'. "
                "Pass --train-dir to the folder that directly contains interval5_CAM_LIDAR/ interval5_CAM_label/ interval5_HAG*/."
            )
            return samples
        if not has_cam_label:
            print(
                "[UAVScenes] Error: 'interval5_CAM_label' folder not found under "
                f"data_root='{self.data_root}'."
            )
            return samples
        if not has_hag:
            print(
                "[UAVScenes] Error: neither 'interval5_HAG_CSF' nor 'interval5_HAG' folder found under "
                f"data_root='{self.data_root}'."
            )
            return samples

        for scene in scenes:
            # RGB path pattern
            rgb_dir = first_existing_dir(*_rgb_dir_candidates(self.data_root, scene))

            if rgb_dir is None:
                print(f"Warning: RGB directory not found for scene '{scene}' under data_root='{self.data_root}'")
                continue

            # HAG path pattern (CSF version)
            hag_dir = first_existing_dir(
                os.path.join(self.data_root, "interval5_HAG_CSF", scene),
                os.path.join(self.data_root, "interval5_HAG", scene),
            )

            if hag_dir is None:
                print(f"Warning: HAG directory not found for scene '{scene}' (tried interval5_HAG_CSF and interval5_HAG)")
                continue

            # Label path pattern
            label_dir = first_existing_dir(
                os.path.join(
                    self.data_root,
                    "interval5_CAM_label", "interval5_CAM_label",
                    scene, "interval5_CAM_label_id"
                ),
                os.path.join(
                    self.data_root,
                    "interval5_CAM_label",
                    scene, "interval5_CAM_label_id"
                ),
            )

            if label_dir is None:
                print(f"Warning: Label directory not found for scene '{scene}'")
                continue

            # Find all RGB images
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))

            for rgb_path in rgb_files:
                # Extract timestamp
                timestamp = os.path.splitext(os.path.basename(rgb_path))[0]

                # Build corresponding HAG and label paths
                hag_path = os.path.join(hag_dir, f"{timestamp}.png")
                label_path = os.path.join(label_dir, f"{timestamp}.png")

                # Only add if all files exist and HAG file is not empty
                if os.path.exists(hag_path) and os.path.exists(label_path) and os.path.getsize(hag_path) > 0:
                    samples.append({
                        'rgb': rgb_path,
                        'hag': hag_path,
                        'label': label_path,
                        'scene': scene,
                        'timestamp': timestamp
                    })

        return samples

    def set_stage(self, stage):
        """Set stage (train/val) for compatibility with GeminiFusion's training loop."""
        self.stage = stage

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Load RGB image
        rgb = np.array(Image.open(sample_info['rgb']).convert('RGB'))

        # Load HAG (normalized, aux_channels determines 1 or 3 channels)
        hag = load_hag(sample_info['hag'], max_height=self.hag_max_height, aux_channels=self.aux_channels)

        # Convert HAG to uint8 range [0, 255] for consistency with RGB
        hag = (hag * 255).astype(np.uint8)

        # Load and remap label
        label = np.array(Image.open(sample_info['label']))
        label = remap_labels(label)

        # Create sample dict - use 'depth' key for HAG (GeminiFusion compatibility)
        sample = {
            'rgb': rgb,
            'depth': hag,  # Named 'depth' for GeminiFusion compatibility
            'mask': label,
        }

        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def num_classes(self):
        return 19

    @property
    def class_names(self):
        return CLASS_NAMES

    @property
    def ignore_label(self):
        return 255


def get_dataloader(
    data_root,
    split='train',
    batch_size=8,
    num_workers=8,
    transform=None,
    hag_max_height=50.0,
    distributed=False
):
    """
    Create DataLoader for UAVScenes dataset.

    Args:
        data_root: Root directory of UAVScenes dataset
        split: 'train' or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        transform: Optional transform
        hag_max_height: Maximum HAG height for normalization
        distributed: Whether to use distributed sampler

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dataset = UAVScenesDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        hag_max_height=hag_max_height
    )

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


if __name__ == '__main__':
    # Test dataset loading
    data_root = "/home/bariskaya/Projelerim/UAV/UAVScenesData"

    # Create dataset
    train_dataset = UAVScenesDataset(data_root, split='train')
    test_dataset = UAVScenesDataset(data_root, split='test')

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")

    # Test loading a sample
    sample = train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Depth (HAG) shape: {sample['depth'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask unique values: {np.unique(sample['mask'])}")
