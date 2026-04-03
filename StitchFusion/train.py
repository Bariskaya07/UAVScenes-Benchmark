import argparse
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

try:
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler as _AmpGradScaler

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast('cuda', enabled=enabled, dtype=dtype)

    def make_grad_scaler(enabled):
        return _AmpGradScaler('cuda', enabled=enabled)

except ImportError:
    from torch.cuda.amp import autocast as _amp_autocast
    from torch.cuda.amp import GradScaler as _AmpGradScaler

    def amp_autocast(*, enabled, dtype):
        return _amp_autocast(enabled=enabled, dtype=dtype)

    def make_grad_scaler(enabled):
        return _AmpGradScaler(enabled=enabled)

try:
    from fvcore.nn import FlopCountAnalysis

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from config import config
from dataloader.dataloader import ValPre, get_train_loader
from dataloader.UAVScenesDataset import UAVScenesDataset
from engine.engine import Engine
from engine.logger import get_logger
from models.builder import EncoderDecoder as segmodel
from utils.lr_policy import WarmUpPolyLR
from utils.metric import compute_score, hist_info
from utils.pyt_utils import all_reduce_tensor, link_file
from utils.visualize import print_iou

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from checkpoint_ops import epoch_checkpoint_name, maybe_sync_checkpoint_dir, promote_best_checkpoint

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - fallback for environments without tensorboardX
    from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'


def get_amp_dtype():
    amp_dtype = str(getattr(config, 'amp_dtype', 'bf16')).lower()
    if amp_dtype == 'bf16':
        return torch.bfloat16
    return torch.float16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_cuda_memory_stats_mb(device):
    if not torch.cuda.is_available():
        return None
    current_device = device if isinstance(device, torch.device) else torch.device(device)
    scale = 1024 ** 2
    return {
        'allocated': torch.cuda.memory_allocated(current_device) / scale,
        'reserved': torch.cuda.memory_reserved(current_device) / scale,
        'peak_allocated': torch.cuda.max_memory_allocated(current_device) / scale,
        'peak_reserved': torch.cuda.max_memory_reserved(current_device) / scale,
    }


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{sec:02d}'
    return f'{minutes:02d}:{sec:02d}'


@torch.no_grad()
def validate_batched(model, val_dataset, device, num_classes, batch_size=8):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    correct = 0
    labeled = 0
    mean = config.norm_mean.reshape(1, 1, 3).astype(np.float32)
    std = config.norm_std.reshape(1, 1, 3).astype(np.float32)

    for batch_start in tqdm(range(0, len(val_dataset), batch_size), desc='Validating', file=sys.stdout):
        batch_end = min(batch_start + batch_size, len(val_dataset))
        batch_rgb = []
        batch_modal = []
        batch_labels = []

        for idx in range(batch_start, batch_end):
            dd = val_dataset[idx]
            img = dd['data'].astype(np.float32) / 255.0
            modal = dd['modal_x'].astype(np.float32)
            batch_rgb.append(((img - mean) / std).transpose(2, 0, 1))
            batch_modal.append((((modal - 0.5) / 0.5)).transpose(2, 0, 1))
            batch_labels.append(dd['label'])

        rgb_t = torch.from_numpy(np.stack(batch_rgb)).float().to(device)
        modal_t = torch.from_numpy(np.stack(batch_modal)).float().to(device)
        preds = model(rgb_t, modal_t).argmax(dim=1).cpu().numpy()

        for pred, label in zip(preds, batch_labels):
            hist_tmp, labeled_tmp, correct_tmp = hist_info(num_classes, pred, label)
            hist += hist_tmp
            correct += correct_tmp
            labeled += labeled_tmp

    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    return mean_IoU, iou, freq_IoU, mean_pixel_acc, pixel_acc


def get_fair_param_groups(model, lr, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = (
            param.ndim == 1
            or name.endswith('.bias')
            or 'norm' in name.lower()
            or 'bn' in name.lower()
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0},
    ]


def apply_freeze_bn_if_needed(net):
    if not getattr(config, 'freeze_bn', False):
        return
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()


def log_model_complexity(model, device):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable')

    if not FVCORE_AVAILABLE:
        logger.info('GFLOPs: skipped (fvcore not installed)')
        return

    try:
        model.eval()
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, config.image_height, config.image_width, device=device)
            dummy_hag = torch.randn(1, 3, config.image_height, config.image_width, device=device)
            analysis = FlopCountAnalysis(model, (dummy_rgb, dummy_hag))
            flops = analysis.total()
        logger.info(f'GFLOPs: {flops / 1e9:.2f}')
    except Exception as exc:
        logger.warning(f'GFLOPs: skipped ({exc})')
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        apply_freeze_bn_if_needed(model)


def main():
    global logger
    logger = get_logger(config.log_dir, config.log_file)
    link_file(config.log_file, config.link_log_file)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        set_seed(config.seed)
        if engine.distributed:
            set_seed(config.seed + engine.local_rank)

        train_loader, train_sampler = get_train_loader(engine, UAVScenesDataset)
        niters_per_epoch = len(train_loader)
        logger.info(f'niters_per_epoch (real): {niters_per_epoch}')

        val_pre = ValPre(resize_to=(config.image_height, config.image_width))
        val_data_setting = {'data_root': config.dataset_path}
        val_dataset = UAVScenesDataset(val_data_setting, 'val', val_pre)

        if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        BatchNorm2d = nn.SyncBatchNorm if engine.distributed else nn.BatchNorm2d
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        base_lr = config.lr
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                get_fair_param_groups(model, lr=base_lr, weight_decay=config.weight_decay),
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(
                get_fair_param_groups(model, lr=base_lr, weight_decay=config.weight_decay),
                lr=base_lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise NotImplementedError

        total_iteration = config.nepochs * niters_per_epoch
        warmup_iters = niters_per_epoch * config.warm_up_epoch
        logger.info(
            "[LR] epochs=%d warmup_epochs=%d warmup_iters=%d power=%.3f warmup_ratio=%.3f warmup=linear",
            config.nepochs,
            config.warm_up_epoch,
            warmup_iters,
            config.lr_power,
            config.warmup_ratio,
        )
        lr_policy = WarmUpPolyLR(
            start_lr=base_lr,
            lr_power=config.lr_power,
            total_iters=total_iteration,
            warmup_steps=warmup_iters,
            warmup_ratio=config.warmup_ratio,
        )

        amp_dtype = get_amp_dtype()
        use_grad_scaler = torch.cuda.is_available() and amp_dtype == torch.float16
        scaler = make_grad_scaler(use_grad_scaler)
        device = torch.device(
            f'cuda:{engine.local_rank}'
            if engine.distributed and torch.cuda.is_available()
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        if engine.distributed:
            logger.info('.............distributed training.............')
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(
                    model,
                    device_ids=[engine.local_rank],
                    output_device=engine.local_rank,
                    find_unused_parameters=False,
                )
        else:
            model.to(device)

        is_main = (engine.distributed and engine.local_rank == 0) or (not engine.distributed)
        if is_main:
            complexity_model = model.module if engine.distributed else model
            log_model_complexity(complexity_model, next(complexity_model.parameters()).device)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad(set_to_none=True)
        model.train()
        apply_freeze_bn_if_needed(model)
        logger.info('begin training:')

        best_miou = 0.0
        for epoch in range(engine.state.epoch, config.nepochs + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format='{desc}')
            dataloader = iter(train_loader)
            sum_loss = 0.0
            data_time = AverageMeter()
            batch_time = AverageMeter()
            epoch_start_time = time.time()
            end = time.time()

            for idx in pbar:
                engine.update_iteration(epoch, idx)
                current_idx = (epoch - 1) * niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)
                for group in optimizer.param_groups:
                    group['lr'] = lr

                minibatch = next(dataloader)
                data_time.update(time.time() - end)
                imgs = minibatch['data'].cuda(non_blocking=True)
                gts = minibatch['label'].cuda(non_blocking=True)
                modal_xs = minibatch['modal_x'].cuda(non_blocking=True)
                valid_pixels = int((gts != config.background).sum().item())

                if valid_pixels == 0:
                    logger.warning(f'All-ignore batch at epoch {epoch} iter {idx}, skipping...')
                    optimizer.zero_grad(set_to_none=True)
                    del imgs, gts, modal_xs, minibatch
                    torch.cuda.empty_cache()
                    end = time.time()
                    continue

                with amp_autocast(enabled=torch.cuda.is_available(), dtype=amp_dtype):
                    loss = model(imgs, modal_xs, gts)

                if torch.isnan(loss) or torch.isinf(loss):
                    rgb_finite = bool(torch.isfinite(imgs).all().item())
                    modal_finite = bool(torch.isfinite(modal_xs).all().item())
                    logger.warning(
                        f'NaN/Inf loss at epoch {epoch} iter {idx}, skipping... '
                        f'(valid_pixels={valid_pixels}, rgb_finite={rgb_finite}, modal_finite={modal_finite})'
                    )
                    optimizer.zero_grad(set_to_none=True)
                    del loss, imgs, gts, modal_xs, minibatch
                    torch.cuda.empty_cache()
                    end = time.time()
                    continue

                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(
                        f'NaN/Inf gradient at epoch {epoch} iter {idx}, skipping step... '
                        f'(valid_pixels={valid_pixels})'
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.update()
                    if engine.distributed:
                        del reduce_loss
                    del grad_norm, loss, imgs, gts, modal_xs, minibatch
                    torch.cuda.empty_cache()
                    end = time.time()
                    continue

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                batch_time.update(time.time() - end)
                elapsed = time.time() - epoch_start_time
                remaining_iters = niters_per_epoch - (idx + 1)
                eta = batch_time.avg * remaining_iters

                if engine.distributed:
                    sum_loss += reduce_loss.item()
                    print_str = (
                        f'Epoch [{epoch}][{idx + 1}/{niters_per_epoch}] '
                        f'Loss: {sum_loss / (idx + 1):.4f} '
                        f'LR: {lr:.6f} '
                        f'Data: {data_time.avg:.3f}s '
                        f'Batch: {batch_time.avg:.3f}s '
                        f'[{format_duration(elapsed)}<{format_duration(eta)}, {batch_time.avg:.2f}s/it]'
                    )
                else:
                    sum_loss += loss.item()
                    print_str = (
                        f'Epoch [{epoch}][{idx + 1}/{niters_per_epoch}] '
                        f'Loss: {sum_loss / (idx + 1):.4f} '
                        f'LR: {lr:.6f} '
                        f'Data: {data_time.avg:.3f}s '
                        f'Batch: {batch_time.avg:.3f}s '
                        f'[{format_duration(elapsed)}<{format_duration(eta)}, {batch_time.avg:.2f}s/it]'
                    )

                if torch.cuda.is_available():
                    mem = get_cuda_memory_stats_mb(device)
                    print_str += (
                        ' alloc=%.0fMiB reserved=%.0fMiB peak_alloc=%.0fMiB peak_reserved=%.0fMiB'
                        % (
                            mem['allocated'],
                            mem['reserved'],
                            mem['peak_allocated'],
                            mem['peak_reserved'],
                        )
                    )
                    if is_main and ((idx + 1) == 1 or (idx + 1) % 100 == 0):
                        logger.info(
                            'Epoch %d Iter %d memory: alloc=%.0fMiB reserved=%.0fMiB peak_alloc=%.0fMiB peak_reserved=%.0fMiB',
                            epoch,
                            idx + 1,
                            mem['allocated'],
                            mem['reserved'],
                            mem['peak_allocated'],
                            mem['peak_reserved'],
                        )

                if engine.distributed:
                    del reduce_loss
                del grad_norm, loss, imgs, gts, modal_xs, minibatch
                pbar.set_description(print_str, refresh=False)
                end = time.time()

            if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
                if torch.cuda.is_available():
                    epoch_mem = get_cuda_memory_stats_mb(device)
                    logger.info(
                        'Epoch %d peak memory: peak_alloc=%.0fMiB peak_reserved=%.0fMiB',
                        epoch,
                        epoch_mem['peak_allocated'],
                        epoch_mem['peak_reserved'],
                    )

            if is_main:
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)
                epoch_ckpt = osp.join(config.checkpoint_dir, epoch_checkpoint_name('stitchfusion', epoch))

            if ((epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0)) or (epoch == config.nepochs):
                if is_main:
                    eval_model = model.module if engine.distributed else model
                    mean_IoU, iou, freq_IoU, mean_pixel_acc, pixel_acc = validate_batched(
                        eval_model,
                        val_dataset,
                        device=next(eval_model.parameters()).device,
                        num_classes=config.num_classes,
                        batch_size=getattr(config, 'eval_batch_size', config.batch_size),
                    )
                    logger.info(f'Epoch {epoch}: mIoU = {mean_IoU:.4f}')
                    tb.add_scalar('val_mIoU', mean_IoU, epoch)
                    logger.info(
                        print_iou(
                            iou,
                            freq_IoU,
                            mean_pixel_acc,
                            pixel_acc,
                            UAVScenesDataset.get_class_names(),
                            show_no_back=False,
                        )
                    )

                    if mean_IoU > best_miou:
                        best_miou = mean_IoU
                        if osp.exists(epoch_ckpt):
                            best_epoch_path = promote_best_checkpoint(epoch_ckpt, 'stitchfusion', epoch)
                            from utils.pyt_utils import link_file

                            link_file(best_epoch_path, osp.join(config.checkpoint_dir, 'best.pth'))
                        logger.info(f'New best mIoU: {best_miou:.4f} at epoch {epoch}')

                    model.train()
                    apply_freeze_bn_if_needed(model)

        if is_main:
            maybe_sync_checkpoint_dir(config.checkpoint_dir, logger=logger.info)
            logger.info(f'Training complete. Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()
