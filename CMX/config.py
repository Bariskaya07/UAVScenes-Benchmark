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

C.seed = 42

# root_dir = directory where THIS config.py lives (not cwd)
C.root_dir = osp.dirname(osp.abspath(__file__))
C.abs_dir = C.root_dir

# Dataset config
"""Dataset Path"""
C.dataset_name = 'UAVScenes'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'UAVScenes')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False  # UAVScenes uses custom label remapping in dataset class
C.x_root_folder = osp.join(C.dataset_path, 'HAG')
C.x_format = '.png'
C.x_is_single_channel = False  # HAG loaded as 3-channel (stacked) by custom dataset
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = None   # Determined at runtime from dataset
C.num_eval_imgs = None    # Determined at runtime from dataset
C.num_classes = 19
C.class_names = [
    'background', 'roof', 'dirt_road', 'paved_road', 'river',
    'pool', 'bridge', 'container', 'airstrip', 'traffic_barrier',
    'green_field', 'wild_field', 'solar_panel', 'umbrella',
    'transparent_roof', 'car_park', 'paved_walk', 'sedan', 'truck'
]

"""Image Config"""
C.background = 255
C.image_height = 768
C.image_width = 768
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2'
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 768  # Match CMNeXt benchmark (768, not default 512)
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 30
C.niters_per_epoch = None  # Determined at runtime: len(train_loader)
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
C.warm_up_epoch = 2  # 2 warmup epochs + 28 decay epochs (30 total)
C.warmup_ratio = 0.1  # Linear warmup starts at lr * 0.1
C.use_photometric = True
C.use_gaussian_blur = True
C.gaussian_blur_prob = 0.2
C.gaussian_blur_kernel = 3
C.freeze_bn = True
C.amp_dtype = 'bf16'  # 'bf16' for stability on A100, set to 'fp16' to revert precision behavior

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 5
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [768, 768]

"""Store Config"""
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.join(C.root_dir, 'log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.join(C.log_dir, "tb")
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.log_dir, "checkpoint")

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
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
