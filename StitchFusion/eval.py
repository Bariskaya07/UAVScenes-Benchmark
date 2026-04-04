import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import config
from dataloader.UAVScenesDataset import UAVScenesDataset
from dataloader.dataloader import ValPre
from engine.evaluator import Evaluator
from engine.logger import get_logger
from models.builder import EncoderDecoder as segmodel
from utils.metric import format_detailed_report, hist_info
from utils.pyt_utils import ensure_dir, parse_devices
from utils.transforms import normalize, pad_image_to_shape
from utils.visualize import show_img

logger = get_logger()
SAVE_LOG_INTERVAL = 100


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        start = time.perf_counter()
        eval_mode = getattr(self, 'eval_mode', 'slide')
        if eval_mode == 'whole':
            pred = self.whole_eval_rgbX(img, modal_x, label.shape, device)
        else:
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
            ensure_dir(self.save_path + '_color')
            fn = name.replace('/', '_') + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred.astype(np.uint8))
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
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, img, clean, label, pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def func_per_batch(self, batch_data, device):
        if len(batch_data) == 1:
            return [self.func_per_iteration(batch_data[0], device)]

        sample_shapes = [(item['data'].shape, item['modal_x'].shape, item['label'].shape) for item in batch_data]
        if any(shape != sample_shapes[0] for shape in sample_shapes[1:]):
            logger.warning('Mixed image shapes in StitchFusion eval batch; falling back to single-image inference.')
            return [self.func_per_iteration(data, device) for data in batch_data]

        imgs = [item['data'] for item in batch_data]
        labels = [item['label'] for item in batch_data]
        modal_xs = [item['modal_x'] for item in batch_data]
        names = [item['fn'] for item in batch_data]

        start = time.perf_counter()
        eval_mode = getattr(self, 'eval_mode', 'slide')
        if eval_mode == 'whole':
            preds = self.whole_eval_rgbX_batch(imgs, modal_xs, labels[0].shape, device)
        else:
            preds = self.sliding_eval_rgbX_batch(imgs, modal_xs, config.eval_crop_size, config.eval_stride_rate, device)
        elapsed = time.perf_counter() - start
        elapsed_per_image = elapsed / max(1, len(batch_data))

        results = []
        for pred, label, name in zip(preds, labels, names):
            hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
            results_dict = {
                'hist': hist_tmp,
                'labeled': labeled_tmp,
                'correct': correct_tmp,
                'elapsed': elapsed_per_image,
            }

            if self.save_path is not None:
                ensure_dir(self.save_path)
                ensure_dir(self.save_path + '_color')
                fn = name.replace('/', '_') + '.png'
                cv2.imwrite(os.path.join(self.save_path, fn), pred.astype(np.uint8))
                saved_count = getattr(self, '_saved_image_count', 0) + 1
                self._saved_image_count = saved_count
                if saved_count % SAVE_LOG_INTERVAL == 0 or saved_count == self.ndata:
                    logger.info(
                        'Saved %d/%d predictions (latest: %s)',
                        saved_count,
                        self.ndata,
                        fn,
                    )

            results.append(results_dict)

        return results

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        total_time = 0.0
        for result in results:
            hist += result['hist']
            total_time += result.get('elapsed', 0.0)
        result_line = format_detailed_report(
            hist,
            dataset.class_names,
            total_time=total_time,
            num_images=len(results),
        )
        logger.info('\n' + result_line)
        return result_line

    def process_image_rgbX(self, img, modal_x, crop_size=None):
        p_img = img
        p_modal_x = modal_x

        if img.shape[2] < 3:
            p_img = np.concatenate((p_img, p_img, p_img), axis=2)

        p_img = normalize(p_img, self.norm_mean, self.norm_std)
        p_modal_x = (p_modal_x - 0.5) / 0.5

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=-1.0)
            p_img = p_img.transpose(2, 0, 1)
            p_modal_x = p_modal_x[np.newaxis, ...] if len(modal_x.shape) == 2 else p_modal_x.transpose(2, 0, 1)
            return p_img, p_modal_x, margin

        p_img = p_img.transpose(2, 0, 1)
        p_modal_x = p_modal_x[np.newaxis, ...] if len(modal_x.shape) == 2 else p_modal_x.transpose(2, 0, 1)
        return p_img, p_modal_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--eval-mode', default='slide', choices=['slide', 'whole'], help='Inference mode')
    parser.add_argument('--legacy-resize-eval', default=False, action='store_true', help='Resize inputs to 768x768 before eval to mimic fast whole-image validation')
    parser.add_argument('--save_path', '-p', default=os.path.join(config.root_dir, 'results2'))
    parser.add_argument('--save-preds', default=False, action='store_true', help='Save per-image prediction PNGs')
    parser.add_argument('--batch-size', default=config.test_batch_size, type=int, help='Eval batch size for batched sliding-window inference')
    args = parser.parse_args()

    all_dev = parse_devices(args.devices)
    amp_dtype_name = str(getattr(config, 'amp_dtype', 'bf16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16
    logger.info('AMP: enabled (dtype=%s)', amp_dtype_name)
    logger.info('Eval mode: %s', args.eval_mode)
    logger.info('Legacy resize eval: %s', 'enabled' if args.legacy_resize_eval else 'disabled')
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'data_root': config.dataset_path}
    resize_to = (config.image_height, config.image_width) if args.legacy_resize_eval else None
    val_pre = ValPre(resize_to=resize_to)
    dataset = UAVScenesDataset(data_setting, 'test', val_pre)
    ensure_dir(args.save_path)
    results_log_file = os.path.join(args.save_path, 'stitchfusion_results.txt')
    results_log_link = os.path.join(args.save_path, 'stitchfusion_results_last.txt')
    pred_save_path = args.save_path if args.save_preds else None
    if args.save_preds:
        logger.info('Prediction saving: enabled')
    else:
        logger.info('Prediction saving: disabled (metrics/log only)')
    logger.info('Eval batch size: %d', args.batch_size)

    with torch.no_grad():
        segmentor = SegEvaluator(
            dataset,
            config.num_classes,
            config.norm_mean,
            config.norm_std,
            network,
            config.eval_scale_array,
            config.eval_flip,
            all_dev,
            args.verbose,
            pred_save_path,
            args.show_image,
            batch_size=args.batch_size,
            use_amp=torch.cuda.is_available(),
            amp_dtype=amp_dtype,
        )
        segmentor.eval_mode = args.eval_mode
        segmentor.run(config.checkpoint_dir, args.epochs, results_log_file, results_log_link)
