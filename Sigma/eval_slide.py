"""
Sliding window evaluation for Sigma on TEST set.
Run on a free GPU while training continues on others.

Usage:
  CUDA_VISIBLE_DEVICES=3 python eval_slide.py \
      --checkpoint log_final/log_uavscenes/log_UAVScenes_sigma_tiny/checkpoint/epoch-45-best.pth
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from configs.config_UAVScenes import config
from models.builder import EncoderDecoder as segmodel
from dataloader.UAVScenesDataset import UAVScenesDataset
from dataloader.dataloader import ValPre
from eval import SegEvaluator
from utils.visualize import print_iou
from utils.metric import hist_info, compute_score

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')

# Build model
model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
model.to(device)

# Load checkpoint
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
if isinstance(ckpt, dict) and 'model' in ckpt:
    state = ckpt['model']
elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
    state = {k.replace('module.', ''): v for k, v in ckpt.items()}
else:
    state = ckpt
model.load_state_dict(state, strict=False)
model.eval()
print(f'Loaded: {args.checkpoint}')

# Build dataset (NO resize - full resolution for sliding window)
data_setting = {
    'rgb_root':       config.rgb_root_folder,
    'rgb_format':     config.rgb_format,
    'gt_root':        config.gt_root_folder,
    'gt_format':      config.gt_format,
    'transform_gt':   config.gt_transform,
    'x_root':         config.x_root_folder,
    'x_format':       config.x_format,
    'x_single_channel': config.x_is_single_channel,
    'class_names':    config.class_names,
    'train_source':   config.train_source,
    'eval_source':    config.eval_source,
    'dataset_path':   config.dataset_path,
    'hag_max_meters': config.hag_max_meters if hasattr(config, 'hag_max_meters') else 50.0,
}
test_pre = ValPre(config, resize=False)  # NO resize for sliding window
dataset = UAVScenesDataset(data_setting, args.split, test_pre)
print(f'{args.split} samples: {len(dataset)}')

# Create evaluator for sliding window
segmentor = SegEvaluator(
    dataset=dataset,
    class_num=config.num_classes,
    norm_mean=config.norm_mean,
    norm_std=config.norm_std,
    network=model,
    multi_scales=config.eval_scale_array,
    is_flip=config.eval_flip,
    devices=[args.gpu],
    verbose=False,
    config=config,
)

# Run sliding window evaluation
crop_size = config.eval_crop_size       # [768, 768]
stride_rate = config.eval_stride_rate   # 2/3

hist = np.zeros((config.num_classes, config.num_classes))
correct = 0
labeled = 0

import time
total_time = 0

with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc=f'Slide eval ({args.split})'):
        sample = dataset[idx]
        img = sample['data']
        label = sample['label']
        modal_x = sample['modal_x']

        torch.cuda.synchronize()
        t0 = time.time()

        pred = segmentor.sliding_eval_rgbX(img, modal_x, crop_size, stride_rate, device)

        torch.cuda.synchronize()
        total_time += time.time() - t0

        h, l, c = hist_info(config.num_classes, pred, label)
        hist += h
        correct += c
        labeled += l

# Results
iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                        dataset.class_names, show_no_back=False)
print(result_line)
print(f'\nmean_IoU: {mean_IoU:.4f}')

avg_ms = (total_time / len(dataset)) * 1000
fps = len(dataset) / total_time
print(f'Avg time: {avg_ms:.1f}ms, FPS: {fps:.2f}')
