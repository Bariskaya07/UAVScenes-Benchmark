"""
UAVScenes Dataset Base Configuration for DFormerv2

Dataset: UAVScenes RGB + HAG Multi-Modal Semantic Segmentation
Target: Fair comparison with CMNeXt benchmark at 768x768 resolution

NOTES:
1. HAG normalization: 50m max (same as CMNeXt for fair comparison)
2. Training resolution: 768x768 (matching CMNeXt for fair comparison)
3. Augmentation: DFormerv2 paper settings (random flip + scale 0.5-1.75)
"""

from .. import *

# Dataset configuration
C.dataset_name = "UAVScenes"
C.dataset_path = "data/UAVScenes"  # VM: ln -s ~/data/UAVScenesData data/UAVScenes

# Path structure for UAVScenes (note the nested directory structure!)
# RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
# Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
# HAG: interval5_HAG/{scene}/{timestamp}.png

# NOTE: These path configs are kept for template compatibility but NOT used by UAVScenesDataset
# UAVScenesDataset builds paths internally using TRAIN_SCENES/TEST_SCENES lists
C.rgb_root_folder = osp.join(C.dataset_path, "interval5_CAM_LIDAR", "interval5_CAM_LIDAR")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "interval5_CAM_label", "interval5_CAM_label")
C.gt_format = ".png"
C.gt_transform = False  # Label remapping handled in UAVScenesDataset

C.x_root_folder = osp.join(C.dataset_path, "interval5_HAG")
C.x_format = ".png"
# x_is_single_channel: Controls normalization values in preprocessing (NOT channel count!)
# True = depth-style normalization [0.48, 0.28], False = ImageNet normalization
C.x_is_single_channel = True

# NOTE: train.txt/test.txt NOT used - UAVScenesDataset uses hardcoded scene lists
C.train_source = osp.join(C.dataset_path, "train.txt")  # Unused
C.eval_source = osp.join(C.dataset_path, "test.txt")    # Unused
C.is_test = True

# Dataset statistics (approximate, will be updated after data exploration)
C.num_train_imgs = 19000  # ~19k training images
C.num_eval_imgs = 4000    # ~4k test images

# 19 classes after remapping (from original 26)
C.num_classes = 19
C.class_names = [
    "background",      # 0
    "roof",            # 1
    "dirt_road",       # 2
    "paved_road",      # 3
    "river",           # 4
    "pool",            # 5
    "bridge",          # 6
    "container",       # 7
    "airstrip",        # 8
    "traffic_barrier", # 9
    "green_field",     # 10
    "wild_field",      # 11
    "solar_panel",     # 12
    "umbrella",        # 13
    "transparent_roof",# 14
    "car_park",        # 15
    "paved_walk",      # 16
    "sedan",           # 17 (dynamic class)
    "truck",           # 18 (dynamic class)
]

# Image configuration - MUST BE 768x768 for fair comparison with CMNeXt!
C.background = 255  # Ignore label value
C.image_height = 768
C.image_width = 768

# ImageNet normalization (same as CMNeXt)
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# HAG normalization max height
# Actual data shows max ~20m, CMNeXt uses 50m for safety margin
# Using 50m for fair comparison with CMNeXt benchmark
C.hag_max_meters = 50.0

# Padding setting for validation (no padding needed for sliding window)
C.pad = False
