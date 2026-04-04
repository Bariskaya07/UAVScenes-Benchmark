import multiprocessing as mp
import os
import re
import time

import cv2
import numpy as np
from timm.models.layers import to_2tuple
from tqdm import tqdm

import torch

from engine.logger import get_logger
from utils.pyt_utils import ensure_dir, link_file, load_model
from utils.transforms import normalize, pad_image_to_shape

logger = get_logger()

try:
    from torch.amp import autocast as _amp_autocast

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast('cuda', enabled=enabled, dtype=dtype)

except ImportError:
    from torch.cuda.amp import autocast as _amp_autocast

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast(enabled=enabled, dtype=dtype)

MODEL_SLUG = 'stitchfusion'
NEW_EPOCH_CKPT_PATTERN = re.compile(r"^stitchfusion_epoch_(\d+)\.pth$")
LEGACY_EPOCH_CKPT_PATTERN = re.compile(r"^epoch-(\d+)\.pth$")


class Evaluator(object):
    def __init__(
        self,
        dataset,
        class_num,
        norm_mean,
        norm_std,
        network,
        multi_scales,
        is_flip,
        devices,
        verbose=False,
        save_path=None,
        show_image=False,
        batch_size=1,
        use_amp=False,
        amp_dtype=torch.bfloat16,
    ):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image
        self.batch_size = max(1, int(batch_size))
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def run(self, model_path, model_indice, log_file, log_file_link):
        if '.pth' in model_indice:
            models = [model_indice]
        elif '-' in model_indice:
            start_epoch = int(model_indice.split('-')[0])
            end_epoch = model_indice.split('-')[1]
            models = self._resolve_epoch_range_models(model_path, start_epoch, end_epoch)
        else:
            models = [self._resolve_single_model_path(model_path, model_indice)]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            if len(self.devices) == 1:
                result_line = self.single_process_evalutation()
            else:
                result_line = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()

    @staticmethod
    def _resolve_epoch_range_models(model_path, start_epoch, end_epoch):
        candidates = []
        for filename in os.listdir(model_path):
            match = NEW_EPOCH_CKPT_PATTERN.match(filename) or LEGACY_EPOCH_CKPT_PATTERN.match(filename)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch < start_epoch:
                continue
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                if epoch > end_epoch:
                    continue
            candidates.append((epoch, os.path.join(model_path, filename)))
        candidates.sort(key=lambda item: item[0])
        return [path for _, path in candidates]

    @staticmethod
    def _resolve_single_model_path(model_path, model_indice):
        if model_indice == 'last':
            preferred = os.path.join(model_path, 'last.pth')
            legacy = os.path.join(model_path, 'epoch-last.pth')
            return preferred if os.path.exists(preferred) else legacy
        if model_indice == 'best':
            preferred = os.path.join(model_path, 'best.pth')
            legacy = os.path.join(model_path, 'epoch-best.pth')
            return preferred if os.path.exists(preferred) else legacy

        preferred = os.path.join(model_path, f'{MODEL_SLUG}_epoch_{model_indice}.pth')
        legacy = os.path.join(model_path, f'epoch-{model_indice}.pth')
        return preferred if os.path.exists(preferred) else legacy

    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()
        logger.info('GPU %s handle %d data (eval_batch_size=%d).' % (self.devices[0], self.ndata, self.batch_size))
        all_results = []
        progress_bar = tqdm(total=self.ndata, desc='Evaluating', leave=True, unit='img')
        for start_idx in range(0, self.ndata, self.batch_size):
            batch_indices = range(start_idx, min(start_idx + self.batch_size, self.ndata))
            batch_data = [self.dataset[idx] for idx in batch_indices]
            batch_results = self.func_per_batch(batch_data, self.devices[0])
            all_results.extend(batch_results)
            progress_bar.update(len(batch_results))
        progress_bar.close()
        result_line = self.compute_metric(all_results)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line

    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info('GPU %s handle %d data.' % (device, len(shred_list)))
            p = self.context.Process(target=self.worker, args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata), desc='Evaluating', leave=True):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line = self.compute_metric(all_results)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (device, time.time() - start_load_time))
        for start_idx in range(0, len(shred_list), self.batch_size):
            batch_indices = shred_list[start_idx:start_idx + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            batch_results = self.func_per_batch(batch_data, device)
            for results_dict in batch_results:
                self.results_queue.put(results_dict)

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def func_per_batch(self, batch_data, device):
        return [self.func_per_iteration(data, device) for data in batch_data]

    def compute_metric(self, results):
        raise NotImplementedError

    def whole_eval(self, img, output_size, device=None):
        processed_pred = np.zeros((output_size[0], output_size[1], self.class_num))
        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            scaled_img = self.process_image(scaled_img, None)
            pred = self.val_func_process(scaled_img, device)
            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(
                pred.cpu().numpy(),
                (output_size[1], output_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return processed_pred.argmax(2)

    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))
        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            processed_pred += self.scale_process(img_scale, (ori_rows, ori_cols), crop_size, stride_rate, device)
        return processed_pred.argmax(2)

    def scale_process(self, img, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, _ = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x:e_x, :]
                    count_scale[:, s_y:e_y, s_x:e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[
                        :,
                        tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                        tmargin[2]:(temp_score.shape[2] - tmargin[3]),
                    ]
                    data_scale[:, s_y:e_y, s_x:e_x] += temp_score

            score = data_scale / count_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                with amp_autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    score = self.val_func(input_data)
                    if self.is_flip:
                        input_data = input_data.flip(-1)
                        score_flip = self.val_func(input_data)
                        score += score_flip.flip(-1)
                score = score[0]
        return score

    def process_image(self, img, crop_size=None):
        p_img = img
        if img.shape[2] < 3:
            p_img = np.concatenate((p_img, p_img, p_img), axis=2)
        p_img = normalize(p_img, self.norm_mean, self.norm_std)
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            return p_img, margin
        p_img = p_img.transpose(2, 0, 1)
        return p_img

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
            p_modal_x = p_modal_x.transpose(2, 0, 1) if len(modal_x.shape) != 2 else p_modal_x[np.newaxis, ...]
            return p_img, p_modal_x, margin
        p_img = p_img.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1) if len(modal_x.shape) != 2 else p_modal_x[np.newaxis, ...]
        return p_img, p_modal_x

    def _stack_rgbx_batch(self, batch_items, crop_size=None):
        input_data_list = []
        input_modal_x_list = []
        margins = []
        for img, modal_x in batch_items:
            processed = self.process_image_rgbX(img, modal_x, crop_size)
            if crop_size is None:
                input_data, input_modal_x = processed
                margin = (0, 0, 0, 0)
            else:
                input_data, input_modal_x, margin = processed
            input_data_list.append(input_data)
            input_modal_x_list.append(input_modal_x)
            margins.append(margin)

        first_margin = margins[0]
        if any(tuple(m) != tuple(first_margin) for m in margins[1:]):
            raise ValueError('Inconsistent padding margins inside eval batch.')

        input_data = np.ascontiguousarray(np.stack(input_data_list, axis=0), dtype=np.float32)
        input_modal_x = np.ascontiguousarray(np.stack(input_modal_x_list, axis=0), dtype=np.float32)
        return input_data, input_modal_x, first_margin

    def whole_eval_rgbX(self, img, modal_x, output_size, device=None):
        processed_pred = np.zeros((output_size[0], output_size[1], self.class_num), dtype=np.float32)
        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            input_data, input_modal_x = self.process_image_rgbX(img_scale, modal_x_scale)
            pred = self.val_func_process_rgbX(input_data, input_modal_x, device)
            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(
                pred.cpu().numpy(),
                (output_size[1], output_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return processed_pred.argmax(2)

    def whole_eval_rgbX_batch(self, imgs, modal_xs, output_size, device=None):
        batch_size = len(imgs)
        processed_pred = np.zeros(
            (batch_size, output_size[0], output_size[1], self.class_num),
            dtype=np.float32,
        )
        for s in self.multi_scales:
            img_scale = [
                cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                for img in imgs
            ]
            modal_x_scale = [
                cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                for modal_x in modal_xs
            ]
            batch_pairs = list(zip(img_scale, modal_x_scale))
            input_data, input_modal_x, _ = self._stack_rgbx_batch(batch_pairs, crop_size=None)
            preds = self.val_func_process_rgbX_batch(input_data, input_modal_x, device)
            for batch_idx in range(batch_size):
                pred = preds[batch_idx].permute(1, 2, 0)
                processed_pred[batch_idx] += cv2.resize(
                    pred.cpu().numpy(),
                    (output_size[1], output_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        return processed_pred.argmax(3)

    def sliding_eval_rgbX(self, img, modal_x, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            processed_pred += self.scale_process_rgbX(
                img_scale,
                modal_x_scale,
                (ori_rows, ori_cols),
                crop_size,
                stride_rate,
                device,
            )
        return processed_pred.argmax(2)

    def sliding_eval_rgbX_batch(self, imgs, modal_xs, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        batch_size = len(imgs)
        ori_rows, ori_cols, _ = imgs[0].shape
        processed_pred = np.zeros((batch_size, ori_rows, ori_cols, self.class_num), dtype=np.float32)

        for s in self.multi_scales:
            img_scale = [
                cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                for img in imgs
            ]
            modal_x_scale = [
                cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                for modal_x in modal_xs
            ]
            processed_pred += self.scale_process_rgbX_batch(
                img_scale,
                modal_x_scale,
                (ori_rows, ori_cols),
                crop_size,
                stride_rate,
                device,
            )
        return processed_pred.argmax(3)

    def scale_process_rgbX(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, _ = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows
        if long_size <= crop_size[0]:
            input_data, input_modal_x, margin = self.process_image_rgbX(img, modal_x, crop_size)
            score = self.val_func_process_rgbX(input_data, input_modal_x, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size[0] * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            modal_pad, _ = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            pad_rows, pad_cols = img_pad.shape[:2]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]

                    img_sub = img_pad[s_y:e_y, s_x:e_x, :]
                    modal_sub = modal_pad[s_y:e_y, s_x:e_x, :]
                    count_scale[:, s_y:e_y, s_x:e_x] += 1

                    input_data, input_modal_x, tmargin = self.process_image_rgbX(img_sub, modal_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, input_modal_x, device)
                    temp_score = temp_score[
                        :,
                        tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                        tmargin[2]:(temp_score.shape[2] - tmargin[3]),
                    ]
                    data_scale[:, s_y:e_y, s_x:e_x] += temp_score

            score = data_scale / count_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        return cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

    def scale_process_rgbX_batch(self, imgs, modal_xs, ori_shape, crop_size, stride_rate, device=None):
        batch_size = len(imgs)
        new_rows, new_cols, _ = imgs[0].shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size[0]:
            batch_pairs = list(zip(imgs, modal_xs))
            input_data, input_modal_x, margin = self._stack_rgbx_batch(batch_pairs, crop_size)
            score = self.val_func_process_rgbX_batch(input_data, input_modal_x, device)
            score = score[:, :, margin[0]:(score.shape[2] - margin[1]), margin[2]:(score.shape[3] - margin[3])]
        else:
            stride = int(np.ceil(crop_size[0] * stride_rate))
            img_pads = []
            modal_pads = []
            margins = []
            for img, modal_x in zip(imgs, modal_xs):
                img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
                modal_pad, _ = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
                img_pads.append(img_pad)
                modal_pads.append(modal_pad)
                margins.append(margin)

            margin = margins[0]
            if any(tuple(m) != tuple(margin) for m in margins[1:]):
                raise ValueError('Inconsistent image shapes inside eval batch.')

            pad_rows, pad_cols = img_pads[0].shape[:2]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride)) + 1
            data_scale = torch.zeros(batch_size, self.class_num, pad_rows, pad_cols).cuda(device)
            count_scale = torch.zeros(batch_size, 1, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]

                    batch_pairs = [
                        (
                            img_pads[b][s_y:e_y, s_x:e_x, :],
                            modal_pads[b][s_y:e_y, s_x:e_x, :],
                        )
                        for b in range(batch_size)
                    ]
                    input_data, input_modal_x, tmargin = self._stack_rgbx_batch(batch_pairs, crop_size)
                    temp_score = self.val_func_process_rgbX_batch(input_data, input_modal_x, device)
                    temp_score = temp_score[
                        :,
                        :,
                        tmargin[0]:(temp_score.shape[2] - tmargin[1]),
                        tmargin[2]:(temp_score.shape[3] - tmargin[3]),
                    ]
                    data_scale[:, :, s_y:e_y, s_x:e_x] += temp_score
                    count_scale[:, :, s_y:e_y, s_x:e_x] += 1

            score = data_scale / count_scale
            score = score[:, :, margin[0]:(score.shape[2] - margin[1]), margin[2]:(score.shape[3] - margin[3])]

        outputs = []
        for batch_idx in range(batch_size):
            score_b = score[batch_idx].permute(1, 2, 0)
            outputs.append(
                cv2.resize(
                    score_b.cpu().numpy(),
                    (ori_shape[1], ori_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        return np.stack(outputs, axis=0)

    def val_func_process_rgbX(self, input_data, input_modal_x, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                with amp_autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    score = self.val_func(input_data, input_modal_x)
                    if self.is_flip:
                        input_data = input_data.flip(-1)
                        input_modal_x = input_modal_x.flip(-1)
                        score_flip = self.val_func(input_data, input_modal_x)
                        score += score_flip.flip(-1)
                score = score[0]
        return score

    def val_func_process_rgbX_batch(self, input_data, input_modal_x, device=None):
        input_data = torch.FloatTensor(input_data).cuda(device)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                with amp_autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    score = self.val_func(input_data, input_modal_x)
                    if self.is_flip:
                        input_data_flip = input_data.flip(-1)
                        input_modal_x_flip = input_modal_x.flip(-1)
                        score_flip = self.val_func(input_data_flip, input_modal_x_flip)
                        score += score_flip.flip(-1)
        return score
