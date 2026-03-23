import os.path as osp
import os
import sys
import time
import random
import shutil
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from config import config
from dataloader.dataloader import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from dataloader.UAVScenesDataset import UAVScenesDataset
from utils.init_func import init_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, ensure_dir
from utils.metric import hist_info, compute_score
from utils.visualize import print_iou

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'


def get_amp_dtype():
    amp_dtype = str(getattr(config, 'amp_dtype', 'bf16')).lower()
    if amp_dtype == 'bf16':
        return torch.bfloat16
    return torch.float16


def set_seed(seed):
    """Set all seeds for reproducibility (matching CMNeXt)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


@torch.no_grad()
def validate_batched(model, val_dataset, device, num_classes, batch_size=8):
    """Fast whole-image validation on resized VAL_SCENES samples."""
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
        if 'analysis' in locals():
            del analysis
        if 'dummy_rgb' in locals():
            del dummy_rgb
        if 'dummy_hag' in locals():
            del dummy_hag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        apply_freeze_bn_if_needed(model)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    # Fix 9: Full reproducibility (matching CMNeXt)
    set_seed(config.seed)
    if engine.distributed:
        set_seed(config.seed + engine.local_rank)

    # data loader - no file_length, use real dataset size
    train_loader, train_sampler = get_train_loader(engine, UAVScenesDataset)
    niters_per_epoch = len(train_loader)  # Fix 1: real dataset length
    logger.info(f'niters_per_epoch (real): {niters_per_epoch}')

    # Validation dataset (raw numpy, no normalization — handled by evaluator)
    val_pre = ValPre(resize_to=(config.image_height, config.image_width))
    val_data_setting = {'data_root': config.dataset_path}
    val_dataset = UAVScenesDataset(val_data_setting, 'val', val_pre)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # CMNeXt-style optimizer semantics: no weight decay on bias / norm / 1D params
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

    # Fix 1+2: LR policy with real dataset length and warmup_ratio=0.1
    total_iteration = config.nepochs * niters_per_epoch
    warmup_iters = niters_per_epoch * config.warm_up_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, warmup_iters, warmup_ratio=0.1)

    # AMP precision is configurable to allow bf16 on A100 while keeping mixed precision.
    amp_dtype = get_amp_dtype()
    use_grad_scaler = torch.cuda.is_available() and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    is_main = (engine.distributed and engine.local_rank == 0) or (not engine.distributed)
    if is_main:
        complexity_model = model.module if engine.distributed else model
        complexity_device = next(complexity_model.parameters()).device
        log_model_complexity(complexity_model, complexity_device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad(set_to_none=True)
    model.train()
    apply_freeze_bn_if_needed(model)
    logger.info('begin training:')

    best_miou = 0.0
    best_epoch = 0

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            # Set LR BEFORE forward/step (no 1-iter lag)
            current_idx = (epoch - 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            valid_pixels = int((gts != config.background).sum().item())

            if valid_pixels == 0:
                logger.warning(f'All-ignore batch at epoch {epoch} iter {idx}, skipping...')
                optimizer.zero_grad(set_to_none=True)
                del imgs, gts, modal_xs, minibatch
                torch.cuda.empty_cache()
                continue

            # AMP forward pass with configurable precision (fp16 or bf16)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=amp_dtype):
                loss = model(imgs, modal_xs, gts)

            # NaN/Inf loss guard (matching CMNeXt train_mm.py)
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
                continue

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Gradient clipping (matching CMNeXt: max_norm=1.0)
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
                continue

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss.item(), (sum_loss / (idx + 1)))

            if engine.distributed:
                del reduce_loss
            del grad_norm, loss, imgs, gts, modal_xs, minibatch
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        # Validation + best checkpoint using fast whole-image validation on val set
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if is_main:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)

                # Use the network without DDP wrapper for evaluation
                eval_model = model.module if engine.distributed else model
                mean_IoU, iou, freq_IoU, mean_pixel_acc, pixel_acc = validate_batched(
                    eval_model, val_dataset, device=next(eval_model.parameters()).device,
                    num_classes=config.num_classes
                )

                logger.info(f'Epoch {epoch}: mIoU = {mean_IoU:.4f}')
                tb.add_scalar('val_mIoU', mean_IoU, epoch)

                result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                        UAVScenesDataset.get_class_names(), show_no_back=False)
                logger.info(result_line)

                # Save best checkpoint
                if mean_IoU > best_miou:
                    previous_best_epoch = best_epoch
                    best_miou = mean_IoU
                    best_epoch = epoch
                    best_path = osp.join(config.checkpoint_dir, 'epoch-best.pth')
                    best_epoch_path = osp.join(config.checkpoint_dir, f'epoch-{epoch}-best.pth')
                    last_path = osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                    if osp.exists(last_path):
                        if previous_best_epoch > 0:
                            previous_best_path = osp.join(
                                config.checkpoint_dir, f'epoch-{previous_best_epoch}-best.pth'
                            )
                            if osp.exists(previous_best_path):
                                os.remove(previous_best_path)
                        shutil.copy(last_path, best_epoch_path)
                        shutil.copy(last_path, best_path)
                    logger.info(f'New best mIoU: {best_miou:.4f} at epoch {epoch}')

                model.train()
                apply_freeze_bn_if_needed(model)

    if is_main:
        logger.info(f'Training complete. Best mIoU: {best_miou:.4f}')
