"""
DELIVER Dataset Base Configuration for DFormerv2

Dataset: DELIVER RGB + LiDAR Multi-Modal Semantic Segmentation
Target: Cross-dataset evaluation with UAVScenes

NOTES:
1. LiDAR: 1-ch PNG stacked to 3-ch (same as CMNeXt original)
2. Training resolution: 1024x1024 (DELIVER native)
3. 25 semantic classes for urban driving
"""

from .. import *

# Dataset configuration
C.dataset_name = "DELIVER"
C.dataset_path = "data/DELIVER"

# Path structure for DELIVER
# RGB: img/{weather}/{split}/{scene}/{frame}_rgb.png
# LiDAR: lidar/{weather}/{split}/{scene}/{frame}_lidar.png
# Semantic: semantic/{weather}/{split}/{scene}/{frame}_semantic.png

C.rgb_root_folder = osp.join(C.dataset_path, "img")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "semantic")
C.gt_format = ".png"
C.gt_transform = False

C.x_root_folder = osp.join(C.dataset_path, "lidar")
C.x_format = ".png"
C.x_is_single_channel = True  # LiDAR normalization

# Dataset statistics (approximate)
C.num_train_imgs = 30000
C.num_eval_imgs = 10000

# 25 classes for DELIVER
C.num_classes = 25
C.class_names = [
    "Building", "Fence", "Other", "Pedestrian", "Pole",
    "RoadLine", "Road", "SideWalk", "Vegetation", "Cars",
    "Wall", "TrafficSign", "Sky", "Ground", "Bridge",
    "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "TwoWheeler", "Bus", "Truck"
]

# Image configuration
C.background = 255
C.image_height = 1024
C.image_width = 1024

# Normalization
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# Padding
C.pad = False
