"""
UAVScenes Dataset for CMX (RGBX_Semantic_Segmentation).
Handles scene-based file structure, 16-bit HAG loading, and label remapping.
Returns data in the same dict format as RGBXDataset for compatibility.
"""

import os
import cv2
import torch
import numpy as np
import torch.utils.data as data


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

# Create NumPy LUT for O(1) label remapping
_LABEL_LUT = np.full(256, 255, dtype=np.uint8)
for _orig_id, _new_id in LABEL_REMAP.items():
    _LABEL_LUT[_orig_id] = _new_id


# Scene-based train/val/test split
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
    'background', 'roof', 'dirt_road', 'paved_road', 'river',
    'pool', 'bridge', 'container', 'airstrip', 'traffic_barrier',
    'green_field', 'wild_field', 'solar_panel', 'umbrella',
    'transparent_roof', 'car_park', 'paved_walk', 'sedan', 'truck'
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


class UAVScenesDataset(data.Dataset):
    """UAVScenes Dataset compatible with CMX (RGBX) training pipeline.

    Handles:
    - Scene-based directory structure (not flat RGB/Label/X folders)
    - 16-bit HAG PNG loading with proper normalization
    - Label remapping from 26 original classes to 19 contiguous classes
    - Returns dict format compatible with CMX train.py: {data, label, modal_x, fn, n}
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(UAVScenesDataset, self).__init__()
        self._split_name = split_name
        self._data_root = setting['data_root']
        self.class_names = CLASS_NAMES
        self.preprocess = preprocess

        # Select scenes based on split
        if split_name == 'train':
            self.scenes = TRAIN_SCENES
        elif split_name == 'val':
            self.scenes = VAL_SCENES
        elif split_name == 'test':
            self.scenes = TEST_SCENES
        else:
            self.scenes = TEST_SCENES

        # Build file list
        self._file_names = self._load_file_list()
        self._file_length = file_length

        print(f"UAVScenesDataset [{split_name}]: {len(self._file_names)} samples from {len(self.scenes)} scenes")

    def _load_file_list(self):
        """Build list of sample dicts with paths."""
        samples = []
        for scene in self.scenes:
            rgb_dir = os.path.join(
                self._data_root,
                'interval5_CAM_LIDAR', 'interval5_CAM_LIDAR',
                scene, 'interval5_CAM'
            )
            if not os.path.exists(rgb_dir):
                print(f"Warning: RGB directory not found: {rgb_dir}")
                continue

            rgb_files = sorted([
                f for f in os.listdir(rgb_dir)
                if f.endswith('.jpg')
            ])

            for rgb_file in rgb_files:
                timestamp = os.path.splitext(rgb_file)[0]

                label_path = os.path.join(
                    self._data_root,
                    'interval5_CAM_label', 'interval5_CAM_label',
                    scene, 'interval5_CAM_label_id',
                    f'{timestamp}.png'
                )
                hag_path = os.path.join(
                    self._data_root,
                    'interval5_HAG_CSF',
                    scene,
                    f'{timestamp}.png'
                )

                if os.path.exists(label_path) and os.path.exists(hag_path):
                    samples.append({
                        'rgb': os.path.join(rgb_dir, rgb_file),
                        'label': label_path,
                        'hag': hag_path,
                        'name': f'{scene}/{timestamp}',
                    })

        return samples

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item = self._construct_new_file_names(self._file_length)[index]
        else:
            item = self._file_names[index]

        # Load RGB (BGR -> RGB)
        rgb = cv2.imread(item['rgb'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load Label and remap
        gt = cv2.imread(item['label'], cv2.IMREAD_UNCHANGED).astype(np.uint8)
        gt = _LABEL_LUT[gt]  # O(1) label remapping

        # Load HAG (16-bit PNG -> normalized 0-1 -> stacked to 3ch)
        hag = self._load_hag(item['hag'])

        # Apply preprocessing (augmentation)
        if self.preprocess is not None:
            rgb, gt, hag = self.preprocess(rgb, gt, hag)

        if self._split_name == 'train':
            # Train: data is [C,H,W] from TrainPre transpose, convert to tensor
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            hag = torch.from_numpy(np.ascontiguousarray(hag)).float()
        # val/test: stay as numpy for evaluator's sliding window

        output_dict = dict(
            data=rgb,
            label=gt,
            modal_x=hag,
            fn=str(item['name']),
            n=len(self._file_names)
        )
        return output_dict

    def _load_hag(self, filepath):
        """Load HAG from 16-bit PNG and convert to normalized 3-channel format.

        HAG encoding: pixel = (HAG_meters * 1000) + 20000
        Normalization: divide by 50m (max observed ~45m, clipped to 0-1)
        Output: [H, W, 3] float32 array with values in [0, 1]
        """
        hag_raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # uint16
        hag_meters = (hag_raw.astype(np.float32) - 20000) / 1000.0
        hag_normalized = np.clip(hag_meters / 50.0, 0, 1)
        # Stack to 3 channels for backbone compatibility
        return np.stack([hag_normalized] * 3, axis=-1)  # [H, W, 3]

    def _construct_new_file_names(self, length):
        """Construct file list of given length (with repetition)."""
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
        return PALETTE

    @staticmethod
    def get_class_names():
        return CLASS_NAMES
