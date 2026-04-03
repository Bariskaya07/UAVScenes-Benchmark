import os
import cv2
import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, format_detailed_report
from dataloader.UAVScenesDataset import UAVScenesDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from utils.transforms import pad_image_to_shape, normalize

logger = get_logger()
SAVE_LOG_INTERVAL = 100


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        start = time.perf_counter()
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        elapsed = time.perf_counter() - start
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {
            'hist': hist_tmp,
            'labeled': labeled_tmp,
            'correct': correct_tmp,
            'elapsed': elapsed,
        }

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name.replace('/', '_') + '.png'

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            saved_count = getattr(self, '_saved_image_count', 0) + 1
            self._saved_image_count = saved_count
            if saved_count % SAVE_LOG_INTERVAL == 0 or saved_count == self.ndata:
                logger.info(
                    'Saved %d/%d predictions (latest: %s)',
                    saved_count,
                    self.ndata,
                    fn,
                )

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        total_time = 0.0
        for d in results:
            hist += d['hist']
            total_time += d.get('elapsed', 0.0)

        result_line = format_detailed_report(
            hist,
            dataset.class_names,
            total_time=total_time,
            num_images=len(results),
        )
        logger.info("\n" + result_line)
        return result_line

    def process_image_rgbX(self, img, modal_x, crop_size=None):
        """Override to handle HAG normalization properly.
        RGB: ImageNet normalization
        HAG: [0, 1] -> [-1, 1] normalization (matching CMNeXt)
        """
        p_img = img
        p_modal_x = modal_x

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        # RGB: standard ImageNet normalization
        p_img = normalize(p_img, self.norm_mean, self.norm_std)

        # HAG: normalize from [0, 1] to [-1, 1] (matching CMNeXt and training pipeline)
        p_modal_x = (p_modal_x - 0.5) / 0.5

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            # HAG pad value = -1.0 (0 meters HAG in [-1,1] normalized space)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=-1.0)
            p_img = p_img.transpose(2, 0, 1)
            if len(modal_x.shape) == 2:
                p_modal_x = p_modal_x[np.newaxis, ...]
            else:
                p_modal_x = p_modal_x.transpose(2, 0, 1)

            return p_img, p_modal_x, margin

        p_img = p_img.transpose(2, 0, 1)

        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_img, p_modal_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=os.path.join(config.root_dir, 'results2'))

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {
        'data_root': config.dataset_path,
    }
    val_pre = ValPre(resize_to=None)  # No resize, no normalize — handled by SegEvaluator
    dataset = UAVScenesDataset(data_setting, 'test', val_pre)
    ensure_dir(args.save_path)
    results_log_file = os.path.join(args.save_path, 'cmx_results.txt')
    results_log_link = os.path.join(args.save_path, 'cmx_results_last.txt')

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, results_log_file,
                      results_log_link)
