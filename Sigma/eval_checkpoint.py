"""
Standalone checkpoint evaluation script for Sigma.
Usage:
  CUDA_VISIBLE_DEVICES=3 python eval_checkpoint.py \
      --checkpoint log_final/log_uavscenes/log_UAVScenes_sigma_tiny/checkpoint/epoch-10.pth \
      --batch_size 8
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
from utils.visualize import print_iou
from utils.metric import hist_info, compute_score

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')

# Build model (no criterion needed for eval)
model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
model.to(device)

# Load checkpoint
ckpt = torch.load(args.checkpoint, map_location=device)
# checkpoint may be state_dict directly or wrapped
if isinstance(ckpt, dict) and 'model' in ckpt:
    state = ckpt['model']
elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
    state = {k.replace('module.', ''): v for k, v in ckpt.items()}
else:
    state = ckpt
model.load_state_dict(state, strict=False)
model.eval()
print(f'Loaded: {args.checkpoint}')

# Build val dataset
val_setting = {
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
val_pre = ValPre(config)
val_dataset = UAVScenesDataset(val_setting, 'val', val_pre)
n = val_dataset.get_length()
print(f'Val samples: {n}')

# Run batched evaluation
mean = config.norm_mean.reshape(1, 1, 3).astype(np.float32)
std  = config.norm_std.reshape(1, 1, 3).astype(np.float32)

hist    = np.zeros((config.num_classes, config.num_classes))
correct = 0
labeled = 0

bs = args.batch_size
with torch.no_grad():
    for batch_start in tqdm(range(0, n, bs), desc='Eval'):
        batch_end = min(batch_start + bs, n)
        batch_rgb, batch_modal, batch_labels = [], [], []

        for idx in range(batch_start, batch_end):
            dd = val_dataset[idx]
            img   = dd['data'].astype(np.float32)   / 255.0
            modal = dd['modal_x'].astype(np.float32) / 255.0
            batch_rgb.append(  ((img   - mean) / std).transpose(2, 0, 1))
            batch_modal.append(((modal - mean) / std).transpose(2, 0, 1))
            batch_labels.append(dd['label'])

        rgb_t   = torch.from_numpy(np.stack(batch_rgb)).float().to(device)
        modal_t = torch.from_numpy(np.stack(batch_modal)).float().to(device)
        preds   = model(rgb_t, modal_t).argmax(dim=1).cpu().numpy()

        for pred, label in zip(preds, batch_labels):
            h, l, c = hist_info(config.num_classes, pred, label)
            hist += h; correct += c; labeled += l

iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                        val_dataset.class_names, show_no_back=False)
print(result_line)
print(f'\nmean_IoU: {mean_IoU:.4f}')
