import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared_paths import resolve_pretrained_path


C = edict()
config = C
cfg = C

C.seed = 42

C.root_dir = osp.dirname(osp.abspath(__file__))
C.abs_dir = C.root_dir

# Dataset
C.dataset_name = 'UAVScenes'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'UAVScenes')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, 'HAG')
C.x_format = '.png'
C.x_is_single_channel = False
C.train_source = osp.join(C.dataset_path, 'train.txt')
C.eval_source = osp.join(C.dataset_path, 'test.txt')
C.is_test = False
C.num_train_imgs = None
C.num_eval_imgs = None
C.num_classes = 19
C.class_names = [
    'background', 'roof', 'dirt_road', 'paved_road', 'river',
    'pool', 'bridge', 'container', 'airstrip', 'traffic_barrier',
    'green_field', 'wild_field', 'solar_panel', 'umbrella',
    'transparent_roof', 'car_park', 'paved_walk', 'sedan', 'truck',
]

# Image
C.background = 255
C.image_height = 768
C.image_width = 768
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# Network
C.backbone = 'mit_b2'
C.pretrained_model = resolve_pretrained_path('pretrained/mit_b2.pth', C.root_dir)
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 768
C.optimizer = 'AdamW'

# StitchFusion-specific
C.moa_type = 'obMoA'
C.moa_r = 8
C.freeze_backbone = False

# Training
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 30
C.niters_per_epoch = None
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
C.warm_up_epoch = 2
C.warmup_ratio = 0.1
C.use_photometric = True
C.use_gaussian_blur = True
C.gaussian_blur_prob = 0.2
C.gaussian_blur_kernel = 3
C.freeze_bn = True
C.amp_dtype = 'bf16'
C.activation_checkpoint = True

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Eval
C.eval_iter = 5
C.eval_batch_size = 8
C.test_batch_size = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [768, 768]

# Checkpoint / validation cadence
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

# Paths
C.log_dir = osp.join(C.root_dir, 'log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.join(C.log_dir, 'tb')
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.root_dir, 'stitchfusion_checkpoints')

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'
