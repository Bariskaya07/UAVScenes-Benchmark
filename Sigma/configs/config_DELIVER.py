"""
DELIVER Dataset Configuration for Sigma (Siamese Mamba Network)

Dataset: DELIVER RGB + LiDAR Multi-Modal Semantic Segmentation
Target: Cross-dataset evaluation with UAVScenes

Key Settings:
- Resolution: 1024x1024 (DELIVER native)
- Classes: 25
- LiDAR: Range + Intensity = 2 channels
"""

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# ==============================================================================
# Dataset Config
# ==============================================================================
C.dataset_name = 'DELIVER'
C.dataset_path = osp.join(C.root_dir, 'data', 'DELIVER')

# Path structure for DELIVER
C.rgb_root_folder = osp.join(C.dataset_path, 'img')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'semantic')
C.gt_format = '.png'
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, 'lidar')
C.x_format = '.png'
C.x_is_single_channel = True

C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

# Dataset statistics
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

# ==============================================================================
# Image Config
# ==============================================================================
C.background = 255
C.image_height = 1024
C.image_width = 1024

# Normalization
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# ==============================================================================
# Network Config
# ==============================================================================
C.backbone = 'sigma_small'
C.pretrained_model = None
C.decoder = 'MambaDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.extra_in_chans = 3  # LiDAR: 1-ch stacked to 3-ch (same as CMNeXt original)
C.aux_channels = 3

# ==============================================================================
# Training Config
# ==============================================================================
C.lr = 6e-5
C.lr_power = 1.0
C.momentum = 0.9
C.weight_decay = 0.01
C.warmup_ratio = 1e-6
C.batch_size = 4  # Reduced for 1024x1024
C.nepochs = 60
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8

# Augmentation
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
C.use_photometric = True
C.use_gaussian_blur = True
C.warm_up_epoch = 3

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# ==============================================================================
# Evaluation Config
# ==============================================================================
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [1024, 1024]

# ==============================================================================
# Checkpoint Config
# ==============================================================================
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

# ==============================================================================
# Path Config
# ==============================================================================
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_deliver/' + 'log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()
