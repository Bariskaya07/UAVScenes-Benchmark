"""
DELIVER Dataset for DFormer Training

Dataset: DELIVER RGB + LiDAR Multi-Modal Semantic Segmentation
Target: Cross-dataset evaluation with configurable input projection

LiDAR Channels:
- aux_channels=1: Single channel (compressed)
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
import torch.utils.data as data
import glob


# DELIVER 25 class names
CLASS_NAMES = [
    "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road",
    "SideWalk", "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground",
    "Bridge", "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "TwoWheeler", "Bus", "Truck"
]


class DELIVERDataset(data.Dataset):
    """
    DELIVER dataset for DFormer training.

    25 semantic classes for urban driving scenarios.
    Supports RGB + LiDAR multi-modal segmentation.
    Compatible with DFormer's data loading interface.
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        """
        Args:
            setting: Dict containing dataset paths and configs
            split_name: 'train' or 'val'
            preprocess: Preprocessing function
            file_length: Optional fixed dataset length
        """
        super(DELIVERDataset, self).__init__()

        self._split_name = split_name
        self._dataset_path = setting.get('dataset_path', 'data/DELIVER')
        self._transform_gt = setting.get('transform_gt', False)

        # Auxiliary channels (1, 2, or 3)
        self.aux_channels = setting.get('aux_channels', 3)

        # Class names
        self.class_names = setting.get('class_names', CLASS_NAMES)

        # Weather/condition filter
        self.case = setting.get('case', None)

        # Get file list
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

        print(f"[DELIVER] Loaded {len(self._file_names)} samples for {split_name}")
        print(f"[DELIVER] Aux channels: {self.aux_channels}")

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_path = self._construct_new_file_names(self._file_length)[index]
        else:
            item_path = self._file_names[index]

        rgb_path = item_path
        lidar_path = rgb_path.replace('/img', '/lidar').replace('_rgb', '_lidar')
        lbl_path = rgb_path.replace('/img', '/semantic').replace('_rgb', '_semantic')

        # Load RGB (BGR format from cv2)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")

        # Load Label
        gt = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Label file not found: {lbl_path}")
        gt[gt == 255] = 0
        gt = gt - 1  # DELIVER labels are 1-indexed
        gt = np.clip(gt, 0, 24).astype(np.uint8)

        # Load and process LiDAR
        lidar = self._load_lidar(lidar_path)

        # Convert LiDAR to required channels for DFormer
        if self.aux_channels == 1:
            modal_x = lidar[..., np.newaxis] if len(lidar.shape) == 2 else lidar[:, :, :1]
        elif self.aux_channels == 2:
            if len(lidar.shape) == 2:
                modal_x = np.stack([lidar, lidar], axis=-1)
            else:
                modal_x = lidar[:, :, :2] if lidar.shape[2] >= 2 else np.stack([lidar[:,:,0]]*2, axis=-1)
        else:
            if len(lidar.shape) == 2:
                modal_x = cv2.merge([lidar, lidar, lidar])
            else:
                modal_x = lidar[:, :, :3] if lidar.shape[2] >= 3 else np.stack([lidar[:,:,0]]*3, axis=-1)

        # Apply preprocessing
        if self.preprocess is not None:
            rgb, gt, modal_x = self.preprocess(rgb, gt, modal_x)

        # Convert to tensors
        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()

        output_dict = dict(
            data=rgb,
            label=gt,
            modal_x=modal_x,
            fn=os.path.basename(rgb_path),
            n=len(self._file_names)
        )

        return output_dict

    def _load_lidar(self, filepath):
        """Load LiDAR data."""
        lidar = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if lidar is None:
            raise FileNotFoundError(f"LiDAR file not found: {filepath}")

        # Convert to uint8 range for compatibility with DFormer preprocessing
        if lidar.dtype == np.uint16:
            lidar = (lidar / 256).astype(np.uint8)

        return lidar

    def _get_file_names(self, split_name):
        """Get list of RGB file paths for the given split."""
        assert split_name in ['train', 'val', 'test']

        # Find all RGB files
        pattern = os.path.join(self._dataset_path, 'img', '*', split_name, '*', '*.png')
        files = sorted(glob.glob(pattern))

        # Filter by case if specified
        if self.case is not None:
            files = [f for f in files if self.case in f]

        return files

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

    @classmethod
    def get_class_colors(*args):
        """Generate class colors for visualization."""
        def uint82bin(n, count=8):
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
        return cmap.tolist()
