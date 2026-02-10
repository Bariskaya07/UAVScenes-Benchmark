"""
UAVScenes Dataset Configuration for Sigma (Siamese Mamba Network)

Dataset: UAVScenes RGB + HAG (Height Above Ground) Multi-Modal Semantic Segmentation
Target: Fair comparison with CMNeXt and DFormerV2 benchmarks

Key Settings:
- Resolution: 768x768 (matching CMNeXt/DFormerV2 for fair comparison)
- Classes: 19 (remapped from 26)
- HAG normalization: 50m max (same as CMNeXt)
- Backbone: sigma_small (comparable params to MiT-B2)
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
C.dataset_name = 'UAVScenes'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'UAVScenes')

# Path structure for UAVScenes
# RGB: interval5_CAM_LIDAR/interval5_CAM_LIDAR/{scene}/interval5_CAM/{timestamp}.jpg
# Label: interval5_CAM_label/interval5_CAM_label/{scene}/interval5_CAM_label_id/{timestamp}.png
# HAG: interval5_HAG_CSF/{scene}/{timestamp}.png
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')  # Symlink to actual data
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')  # Symlink to actual data
C.gt_format = '.png'
C.gt_transform = False  # Label remapping handled in UAVScenesDataset
C.x_root_folder = osp.join(C.dataset_path, 'HAG')  # Symlink to actual data
C.x_format = '.png'
C.x_is_single_channel = True  # HAG is single channel (16-bit PNG)

# NOTE: These txt files are not used - UAVScenesDataset uses hardcoded scene lists
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

# Dataset statistics
C.num_train_imgs = 15000  # ~15k training images (13 scenes - NewSplit)
C.num_eval_imgs = 4500    # ~4.5k test images (4 scenes)

# 19 classes after remapping (from original 26)
C.num_classes = 19
C.class_names = [
    'background',       # 0
    'roof',             # 1
    'dirt_road',        # 2
    'paved_road',       # 3
    'river',            # 4
    'pool',             # 5
    'bridge',           # 6
    'container',        # 7
    'airstrip',         # 8
    'traffic_barrier',  # 9
    'green_field',      # 10
    'wild_field',       # 11
    'solar_panel',      # 12
    'umbrella',         # 13
    'transparent_roof', # 14
    'car_park',         # 15
    'paved_walk',       # 16
    'sedan',            # 17 (dynamic class)
    'truck',            # 18 (dynamic class)
]

# ==============================================================================
# Image Config (768x768 for fair comparison with CMNeXt/DFormerV2)
# ==============================================================================
C.background = 255  # Ignore label value
C.image_height = 768
C.image_width = 768

# ImageNet normalization (same as CMNeXt)
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# HAG normalization
C.hag_max_meters = 50.0  # Maximum height for normalization (same as CMNeXt)

# ==============================================================================
# Network Config
# ==============================================================================
C.backbone = 'sigma_small'  # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None   # Handled internally by backbone
C.decoder = 'MambaDecoder'  # Sigma's Mamba-based decoder
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.extra_in_chans = 3  # Input channels for aux modality (3=stacked HAG for backward compat)
C.aux_channels = 3    # Dataset aux channels (3=stacked HAG for backward compat)

# ==============================================================================
# Training Config (Standardized for fair comparison)
# ==============================================================================
C.lr = 6e-5           # Standard LR for fair comparison
C.lr_power = 0.9      # PolyLR power (CMNeXt paper setting)
C.momentum = 0.9
C.weight_decay = 0.01
C.warmup_ratio = 0.1  # Initial LR = LR * warmup_ratio (CMNeXt paper setting)
C.batch_size = 8      # Same as CMNeXt/DFormerV2
C.nepochs = 60        # Training epochs
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8

# Augmentation (Standardized with CMNeXt for fair comparison)
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]  # Scale range 0.5-2.0
C.use_photometric = True   # Photometric distortion (brightness, contrast, saturation, hue)
C.use_gaussian_blur = True  # Gaussian blur (p=0.5, kernel=5)
C.warm_up_epoch = 3   # Warmup epochs (5% of total)

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# ==============================================================================
# Evaluation Config
# ==============================================================================
C.eval_stride_rate = 2 / 3  # ~33% overlap for sliding window
C.eval_scale_array = [1]    # Single scale evaluation
C.eval_flip = False
C.eval_crop_size = [768, 768]  # [height, width]

# ==============================================================================
# Checkpoint & Evaluation Config (Fair comparison: eval every 5 epochs)
# ==============================================================================
C.checkpoint_start_epoch = 5   # Start evaluation from epoch 5
C.checkpoint_step = 5          # Evaluate every 5 epochs (same as other models)

# ==============================================================================
# Path Config
# ==============================================================================
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_uavscenes/' + 'log_' + C.dataset_name + '_' + C.backbone)
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

    if args.tensorboard:
        open_tensorboard()
