import os
import torch 
import torch.nn as nn
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm import evaluate
import warnings
warnings.filterwarnings("ignore")


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = 1

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    dataset_extra_kwargs = {}
    if dataset_cfg['NAME'] == 'UAVScenes':
        dataset_extra_kwargs = {
            'hag_max_meters': dataset_cfg.get('HAG_MAX_METERS', 50.0),
            'aux_channels': dataset_cfg.get('AUX_CHANNELS', 3),
        }

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'], **dataset_extra_kwargs)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'], **dataset_extra_kwargs)
    class_names = trainset.CLASSES

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        # print(msg)
        logger.info(msg)
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(
        sched_cfg['NAME'],
        optimizer,
        int(epochs * iters_per_epoch),
        sched_cfg['POWER'],
        iters_per_epoch * sched_cfg['WARMUP'],
        sched_cfg['WARMUP_RATIO']
    )

    def apply_freeze_bn_if_needed(net):
        if not train_cfg.get('FREEZE_BN', False):
            return
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint['loss']        
        best_mIoU = resume_checkpoint['best_miou']
           
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val)

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
        # Calculate FLOPs
        try:
            cal_flops(model, dataset_cfg['MODALS'], logger)
        except Exception as e:
            logger.info(f"Could not calculate FLOPs: {e}")
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

    for epoch in range(start_epoch, epochs):
        model.train()
        apply_freeze_bn_if_needed(model)
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0        
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)

            logits = model(sample)
            loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1) >= train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                # Get eval mode from config (default: whole for fast validation)
                eval_mode = eval_cfg.get('MODE', 'whole')
                sliding = (eval_mode == 'slide')
                eval_size = tuple(eval_cfg.get('IMAGE_SIZE', [768, 768]))
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device, eval_size=eval_size, sliding=sliding)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    # if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))

    # Final evaluation on test set with slide mode
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        logger.info("\n" + "="*60)
        logger.info("Final evaluation on TEST set with SLIDE mode")
        logger.info("="*60)

        # Load best model
        best_ckpt_pattern = save_dir / f"*_epoch{best_epoch}_*_checkpoint.pth"
        import glob
        best_ckpts = glob.glob(str(best_ckpt_pattern))
        if best_ckpts:
            best_ckpt = best_ckpts[0]
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded best checkpoint: {best_ckpt}")

        # Create test dataloader
        testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', valtransform, dataset_cfg['MODALS'], **dataset_extra_kwargs)
        testloader = DataLoader(testset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False)
        logger.info(f"Test set: {len(testset)} samples")

        # Evaluate with slide mode using UAVScenesMetrics for detailed output
        from semseg.metrics import UAVScenesMetrics

        model.eval()
        test_metrics = UAVScenesMetrics(num_classes=testset.n_classes, ignore_label=testset.ignore_label)

        logger.info(f"Evaluating with SLIDE mode...")
        import time
        total_time = 0
        num_images = 0

        for images, labels in tqdm(testloader, desc="Test evaluation"):
            images = [x.to(device) for x in images]
            labels = labels.numpy()

            # Sliding window inference with timing
            torch.cuda.synchronize()
            start_time = time.time()

            from val_mm import sliding_predict
            preds = sliding_predict(model, images, num_classes=testset.n_classes)

            torch.cuda.synchronize()
            total_time += time.time() - start_time
            num_images += labels.shape[0]

            preds = preds.argmax(dim=1).cpu().numpy()
            test_metrics.update(preds, labels)

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
        results_dir = os.path.join(save_dir, 'results')
        test_metrics.save_results(results_dir, 'Mul_VMamba', avg_time_ms, fps, num_images, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mcubes_rgbadnmulmamba.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(42)  # Fair comparison with CMNeXt
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])

    if 'EXP' in cfg['MODEL'].keys():
        exp_name = '_'.join([cfg['DATASET']['NAME'], model,cfg['MODEL']['EXP'], modals])
    else:
        exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])

    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()
