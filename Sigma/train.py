import os.path as osp
import os
import sys
import time
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from dataloader.UAVScenesDataset import UAVScenesDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from checkpoint_ops import materialize_epoch_checkpoint, promote_best_checkpoint, maybe_sync_checkpoint_dir

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '16005'


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        from configs.config_MFNet import config
    elif dataset_name == 'pst':
        from configs.config_pst900 import config
    elif dataset_name == 'nyu':
        from configs.config_nyu import config
    elif dataset_name == 'sun':
        from configs.config_sunrgbd import config
    elif dataset_name == 'uavscenes':
        from configs.config_UAVScenes import config
    else:
        raise ValueError('Not a valid dataset name')

    print("=======================================")
    print(config.tb_dir)
    print("=======================================")

    seed = config.seed
    if engine.distributed:
        seed = config.seed + engine.local_rank
    set_seed(seed)

    # data loader - use UAVScenesDataset for UAVScenes, RGBXDataset for others
    if dataset_name == 'uavscenes':
        DatasetClass = UAVScenesDataset
    else:
        DatasetClass = RGBXDataset
    train_loader, train_sampler = get_train_loader(engine, DatasetClass, config)
    niters_per_epoch = len(train_loader)
    config.niters_per_epoch = niters_per_epoch
    if hasattr(train_loader, 'dataset'):
        config.num_train_imgs = len(train_loader.dataset)

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
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
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

    # Make the effective LR schedule explicit at the call site so runtime
    # values are visible without relying on scheduler class defaults.
    total_iteration = config.nepochs * niters_per_epoch
    schedule_power = config.lr_power
    schedule_warmup_epochs = config.warm_up_epoch
    schedule_warmup_iters = niters_per_epoch * schedule_warmup_epochs
    schedule_warmup_ratio = config.warmup_ratio
    logger.info(
        "[LR] epochs=%d warmup_epochs=%d warmup_iters=%d power=%.3f warmup_ratio=%.3f warmup=linear",
        config.nepochs,
        schedule_warmup_epochs,
        schedule_warmup_iters,
        schedule_power,
        schedule_warmup_ratio,
    )
    lr_policy = WarmUpPolyLR(
        start_lr=base_lr,
        lr_power=schedule_power,
        total_iters=total_iteration,
        warmup_steps=schedule_warmup_iters,
        warmup_ratio=schedule_warmup_ratio,
    )

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # Calculate FLOPs after model is on CUDA
    if FVCORE_AVAILABLE:
        try:
            _device = next(model.parameters()).device
            dummy_rgb = torch.zeros(1, 3, 768, 768).to(_device)
            dummy_modal = torch.zeros(1, 3, 768, 768).to(_device)
            model.eval()
            flops = FlopCountAnalysis(model, (dummy_rgb, dummy_modal))
            logger.info(f"FLOPs: {flops.total() / 1e9:.2f}G")
            model.train()
        except Exception as e:
            logger.info(f"Could not calculate FLOPs: {e}")
    else:
        logger.info("FLOPs: fvcore not installed (pip install fvcore)")

    def apply_freeze_bn_if_needed(net):
        if not getattr(config, 'freeze_bn', False):
            return
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def validate_batched(model, val_dataset, config, device, val_log_file, batch_size=8):
        """Batched whole-image validation (~8x faster than single-sample evaluation)."""
        n = val_dataset.get_length()
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        mean = config.norm_mean.reshape(1, 1, 3).astype(np.float32)
        std = config.norm_std.reshape(1, 1, 3).astype(np.float32)

        for batch_start in tqdm(range(0, n, batch_size), desc='Val'):
            batch_end = min(batch_start + batch_size, n)
            batch_rgb, batch_modal, batch_labels = [], [], []

            for idx in range(batch_start, batch_end):
                dd = val_dataset[idx]
                img   = dd['data'].astype(np.float32) / 255.0       # (H, W, 3)
                modal = dd['modal_x'].astype(np.float32)            # (H, W, 3), already in [0, 1]
                batch_rgb.append(((img   - mean) / std).transpose(2, 0, 1))
                batch_modal.append((((modal - 0.5) / 0.5)).transpose(2, 0, 1))
                batch_labels.append(dd['label'])

            rgb_t   = torch.from_numpy(np.stack(batch_rgb)).float().cuda(device)
            modal_t = torch.from_numpy(np.stack(batch_modal)).float().cuda(device)
            preds = model(rgb_t, modal_t).argmax(dim=1).cpu().numpy()  # (B, H, W)

            for pred, label in zip(preds, batch_labels):
                h, l, c = hist_info(config.num_classes, pred, label)
                hist += h; correct += c; labeled += l

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                val_dataset.class_names, show_no_back=False)
        with open(val_log_file, 'a') as f:
            f.write(result_line + '\n')
        return result_line, mean_IoU

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    use_amp = bool(getattr(config, 'amp', True))
    scaler = GradScaler(init_scale=256, enabled=use_amp)
    optimizer.zero_grad()
    model.train()
    apply_freeze_bn_if_needed(model)
    logger.info('begin trainning:')
    logger.info(f'AMP: {use_amp}')
    
    # Initialize the evaluation dataset and evaluator
    val_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                    'dataset_path': config.dataset_path if hasattr(config, 'dataset_path') else '',
                    'hag_max_meters': config.hag_max_meters if hasattr(config, 'hag_max_meters') else 50.0,
                    'aux_channels': getattr(config, 'aux_channels', 3)}
    val_pre = ValPre(config)
    val_dataset = DatasetClass(val_setting, 'val', val_pre)

    best_mean_iou = 0.0
    best_epoch = -1
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

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

            with autocast(enabled=use_amp):
                loss = model(imgs, modal_xs, gts)

            if engine.distributed:
                finite_flag = torch.tensor(1 if torch.isfinite(loss) else 0, device=imgs.device)
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                is_finite = finite_flag.item() == 1
            else:
                is_finite = torch.isfinite(loss).item()

            if not is_finite:
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if engine.distributed:
                grad_finite_flag = torch.tensor(1 if torch.isfinite(grad_norm) else 0, device=imgs.device)
                dist.all_reduce(grad_finite_flag, op=dist.ReduceOp.MIN)
                grad_is_finite = grad_finite_flag.item() == 1
            else:
                grad_is_finite = torch.isfinite(grad_norm).item()

            if not grad_is_finite:
                optimizer.zero_grad()
                scaler.update()
                torch.cuda.empty_cache()
                continue

            scaler.step(optimizer)
            scaler.update()

            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss.item(), (sum_loss / (idx + 1)))
                pbar.set_description(print_str, refresh=False)
            del loss
            
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        # Save resume checkpoint every epoch (single overwritten file, safe for spot VMs)
        if (engine.distributed and engine.local_rank == 0) or not engine.distributed:
            ensure_dir(config.checkpoint_dir)
            if not osp.exists(config.log_dir_link):
                link_file(config.log_dir, config.log_dir_link)
            last_ckpt = osp.join(config.checkpoint_dir, 'epoch-last.pth')
            if osp.islink(last_ckpt):
                os.remove(last_ckpt)
            engine.save_checkpoint(last_ckpt)

        # Save named checkpoint for evaluation/best-model tracking (every checkpoint_step epochs)
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_checkpoint(osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth'))
                materialize_epoch_checkpoint(osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth'), 'sigma', epoch)
            elif not engine.distributed:
                engine.save_checkpoint(osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth'))
                materialize_epoch_checkpoint(osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth'), 'sigma', epoch)

        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()
        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval()
                    with torch.no_grad():
                        _, mean_IoU = validate_batched(model, val_dataset, config,
                                                       engine.local_rank, config.val_log_file)
                        print('mean_IoU:', mean_IoU)

                        # Determine if the model performance improved
                        if mean_IoU > best_mean_iou:
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                            best_ckpt_path = os.path.join(config.checkpoint_dir, 'epoch-best.pth')
                            src_ckpt_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(src_ckpt_path):
                                shutil.copy(src_ckpt_path, best_ckpt_path)
                                promote_best_checkpoint(src_ckpt_path, 'sigma', epoch)

                    model.train()
                    apply_freeze_bn_if_needed(model)
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval()
                with torch.no_grad():
                    _, mean_IoU = validate_batched(model, val_dataset, config,
                                                   0, config.val_log_file)
                    print('mean_IoU:', mean_IoU)

                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                        best_ckpt_path = os.path.join(config.checkpoint_dir, 'epoch-best.pth')
                        src_ckpt_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(src_ckpt_path):
                            shutil.copy(src_ckpt_path, best_ckpt_path)
                            promote_best_checkpoint(src_ckpt_path, 'sigma', epoch)
                model.train()
                apply_freeze_bn_if_needed(model)

    # Save last epoch checkpoint
    last_ckpt = os.path.join(config.checkpoint_dir, 'epoch-last.pth')
    if engine.distributed and engine.local_rank == 0:
        torch.save(model.module.state_dict(), last_ckpt)
        logger.info(f"Saved last epoch checkpoint: {last_ckpt}")
        maybe_sync_checkpoint_dir(config.checkpoint_dir, logger.info)
    elif not engine.distributed:
        torch.save(model.state_dict(), last_ckpt)
        logger.info(f"Saved last epoch checkpoint: {last_ckpt}")
        maybe_sync_checkpoint_dir(config.checkpoint_dir, logger.info)

    # Final evaluation on test set with slide mode
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation on TEST set with SLIDE mode")
    logger.info("=" * 60)

    # Create test dataset
    test_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                    'dataset_path': config.dataset_path if hasattr(config, 'dataset_path') else '',
                    'hag_max_meters': config.hag_max_meters if hasattr(config, 'hag_max_meters') else 50.0,
                    'aux_channels': getattr(config, 'aux_channels', 3)}
    test_pre = ValPre(config, resize=False)  # Test uses full resolution for sliding window
    test_dataset = DatasetClass(test_setting, 'test', test_pre)
    logger.info(f"Test set: {len(test_dataset)} samples")

    # Load best model
    best_ckpt = os.path.join(config.checkpoint_dir, 'epoch-best.pth')
    if os.path.exists(best_ckpt):
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

    # Evaluate with slide mode using UAVScenesMetrics for detailed output
    from utils.metric import UAVScenesMetrics
    from tqdm import tqdm

    model.eval()
    test_metrics = UAVScenesMetrics(num_classes=config.num_classes, ignore_label=config.background)

    # Create evaluator for sliding window inference
    segmentor = SegEvaluator(dataset=test_dataset, class_num=config.num_classes,
                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                            network=model, multi_scales=config.eval_scale_array,
                            is_flip=config.eval_flip, devices=[0],
                            verbose=False, config=config)

    logger.info("Running test evaluation with sliding window inference...")
    import time
    total_time = 0
    num_images = 0

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for idx in tqdm(range(len(test_dataset)), desc="Test evaluation"):
            sample = test_dataset[idx]
            img = sample['data']
            label = sample['label']
            modal_x = sample['modal_x']

            # Sliding window inference with timing
            torch.cuda.synchronize()
            start_time = time.time()

            pred = segmentor.sliding_eval_rgbX(img, modal_x,
                                               config.eval_crop_size, config.eval_stride_rate, device)

            torch.cuda.synchronize()
            total_time += time.time() - start_time
            num_images += 1

            test_metrics.update(pred, label)

    # Calculate inference speed
    avg_time_ms = (total_time / num_images) * 1000
    fps = num_images / total_time

    # Print detailed results
    test_metrics.print_results(logger)

    # Print and save inference speed
    logger.info(f"\nInference speed:")
    logger.info(f"  Average time per image: {avg_time_ms:.1f}ms")
    logger.info(f"  FPS: {fps:.2f}")

    # Save results to file
    results_dir = os.path.join(config.log_dir, 'results')
    test_metrics.save_results(results_dir, 'Sigma', avg_time_ms, fps, num_images, logger)
