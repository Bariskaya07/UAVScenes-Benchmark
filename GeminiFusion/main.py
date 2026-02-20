"""
GeminiFusion Training Script for UAVScenes Dataset

Adapted from original GeminiFusion for fair comparison with CMNeXt, DFormerv2, and TokenFusion.
Uses RGB + HAG (Height Above Ground) multimodal input.

Fair Comparison Settings:
- Image Size: 768x768
- Batch Size: 8
- HAG Max Height: 50m
- Classes: 19
- Train/Test Split: 16/4 scenes
- Backbone: MiT-B2
"""

import os
import sys
import argparse
import random
import time
import warnings
import datetime
import functools

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from utils import *
import utils.helpers as helpers
from utils.optimizer import PolyWarmupAdamW
from models.segformer import WeTr
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.augmentations_mm import *
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, CPUOffload, BackwardPrefetch
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    FSDP_AVAILABLE = True
except Exception:
    FSDP_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from datasets.uavscenes import UAVScenesDataset


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def setup_ddp():
    """Setup distributed data parallel training."""
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        # Multi-node SLURM
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GeminiFusion UAVScenes Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="uavscenes",
        choices=["uavscenes", "nyudv2", "sunrgbd"],
        help="Dataset name",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/home/bariskaya/Projelerim/UAV/UAVScenesData",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8 for fair comparison)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size (default: 1). Increase for faster eval if VRAM allows.",
    )
    parser.add_argument(
        "--val-progress",
        type=str,
        default="rank0",
        choices=["rank0", "all", "none"],
        help="Validation progress display: rank0 (default), all ranks, or none.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Label to ignore during training",
    )

    # General
    parser.add_argument("--name", default="", type=str, help="Experiment name")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate only (no training)",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        default=True,
        help="Freeze batch normalization",
    )

    # Distributed / FSDP
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="Enable Fully Sharded Data Parallel (FSDP) when launched with torchrun",
    )
    parser.add_argument(
        "--fsdp-auto-wrap",
        type=str,
        default="size",
        choices=["none", "size"],
        help="FSDP auto-wrapping policy",
    )
    parser.add_argument(
        "--fsdp-min-params",
        type=int,
        default=1_000_000,
        help="Min parameter count for size-based auto-wrap",
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        default=False,
        help="Offload parameters to CPU (usually slower; can reduce VRAM)",
    )

    # Precision / memory
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable mixed precision training (AMP)",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision training (force FP32)",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--num-epoch",
        type=int,
        nargs="+",
        default=[20, 20, 20],  # 60 total epochs (fair comparison with CMNeXt)
        help="Number of epochs per training stage",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default="uavscenes_geminifusion",
        type=str,
        metavar="PATH",
        help="Checkpoint directory name",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Validation frequency (epochs)",
    )
    parser.add_argument(
        "--print-network",
        action="store_true",
        default=False,
        help="Print network parameters",
    )
    parser.add_argument(
        "--print-loss",
        action="store_true",
        default=False,
        help="Print loss during training",
    )
    parser.add_argument(
        "--save-image",
        type=int,
        default=10,
        help="Number of images to save during validation",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=["rgb", "depth"],
        type=str,
        nargs="+",
        help="Input modalities (rgb, depth/hag)",
    )

    # Model
    parser.add_argument("--backbone", default="mit_b2", type=str,
                       help="Backbone (mit_b2 for fair comparison)")
    parser.add_argument("--n_heads", default=8, type=int,
                       help="Number of cross-attention heads")
    parser.add_argument("--drop_rate", default=0.0, type=float,
                       help="Dropout rate")
    parser.add_argument("--dpr", default=0.1, type=float,
                       help="Drop path rate (0.1 for B2)")

    # Optimizer
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr_0", default=6e-5, type=float)
    parser.add_argument("--lr_1", default=3e-5, type=float)
    parser.add_argument("--lr_2", default=1.5e-5, type=float)
    parser.add_argument("--is_pretrain_finetune", action="store_true")

    # UAVScenes specific
    parser.add_argument("--hag-max-height", default=50.0, type=float,
                       help="Maximum HAG height for normalization (50m for fair comparison)")

    return parser.parse_args()


def create_segmenter(num_classes, gpu, backbone, n_heads, dpr, drop_rate):
    """Create segmentation model."""
    segmenter = WeTr(backbone, num_classes, n_heads, dpr, drop_rate)
    param_groups = segmenter.get_param_groups()
    assert torch.cuda.is_available()
    segmenter.to("cuda:" + str(gpu))
    return segmenter, param_groups


def _maybe_wrap_fsdp(segmenter: nn.Module, gpu: int, args: argparse.Namespace) -> nn.Module:
    if not args.fsdp:
        return segmenter
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        print_log("[FSDP] --fsdp set but distributed is not initialized; running single-GPU.")
        return segmenter
    if not FSDP_AVAILABLE:
        raise RuntimeError(
            "FSDP requested but not available in this PyTorch build. "
            "Install a PyTorch version with torch.distributed.fsdp support."
        )

    auto_wrap_policy = None
    if args.fsdp_auto_wrap == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(args.fsdp_min_params)
        )

    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None

    fsdp_segmenter = FSDP(
        segmenter,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        use_orig_params=True,
        device_id=torch.device(f"cuda:{gpu}"),
    )
    print_log(
        f"[FSDP] Enabled: world_size={dist.get_world_size()} auto_wrap={args.fsdp_auto_wrap} "
        f"min_params={args.fsdp_min_params} cpu_offload={args.fsdp_cpu_offload}"
    )
    return fsdp_segmenter


def _segmenter_state_dict_for_save(segmenter: nn.Module, args: argparse.Namespace) -> dict:
    if args.fsdp and dist.is_initialized() and dist.get_world_size() > 1:
        # Full (unsharded) state dict on rank0 only; offload to CPU to save VRAM.
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(segmenter, StateDictType.FULL_STATE_DICT, full_cfg):
            return segmenter.state_dict()
    if isinstance(segmenter, DDP):
        return segmenter.module.state_dict()
    return segmenter.state_dict()


def create_uavscenes_loaders(args, input_scale):
    """Create UAVScenes data loaders."""
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from utils.transforms import ToTensor

    input_names = ["rgb", "depth"]  # depth = HAG

    # Training transforms
    composed_trn = transforms.Compose([
        ToTensor(),
        RandomColorJitter(p=0.2),
        RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur((3, 3), p=0.2),
        RandomResizedCrop(input_scale, scale=(0.5, 2.0), seg_fill=255),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Validation transforms (resize for fast validation during training)
    composed_val = transforms.Compose([
        ToTensor(),
        Resize(input_scale),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Test transforms (full resolution for accurate slide mode evaluation)
    composed_test = transforms.Compose([
        ToTensor(),
        # No resize - keep full resolution for sliding window inference
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create datasets
    trainset = UAVScenesDataset(
        data_root=args.train_dir,
        split='train',
        transform=composed_trn,
        hag_max_height=args.hag_max_height,
    )

    validset = UAVScenesDataset(
        data_root=args.train_dir,
        split='val',  # Use validation set during training (not test!)
        transform=composed_val,
        hag_max_height=args.hag_max_height,
    )

    testset = UAVScenesDataset(
        data_root=args.train_dir,
        split='test',  # Test set only for final evaluation
        transform=composed_test,  # Full resolution for slide mode
        hag_max_height=args.hag_max_height,
    )

    print_log(f"Created train set {len(trainset)} examples, val set {len(validset)} examples, test set {len(testset)} examples")

    if len(trainset) == 0 or len(validset) == 0 or len(testset) == 0:
        raise ValueError(
            "UAVScenesDataset returned 0 samples. This usually means --train-dir is wrong or the dataset folder structure doesn't match. "
            f"Got data_root='{args.train_dir}'. Expected e.g. '{args.train_dir}/interval5_CAM_LIDAR/...' and '{args.train_dir}/interval5_CAM_label/...' and HAG under '{args.train_dir}/interval5_HAG_CSF/' (or 'interval5_HAG/')."
        )

    # Create samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(
            trainset, dist.get_world_size(), dist.get_rank(), shuffle=True
        )
    else:
        train_sampler = None

    if dist.is_initialized():
        val_sampler = DistributedSampler(
            validset, dist.get_world_size(), dist.get_rank(), shuffle=False
        )
        test_sampler = DistributedSampler(
            testset, dist.get_world_size(), dist.get_rank(), shuffle=False
        )
    else:
        val_sampler = None
        test_sampler = None

    # Create loaders
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    val_loader = DataLoader(
        validset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader, train_sampler


def load_ckpt(ckpt_path, ckpt_dict, is_pretrain_finetune=False):
    """Load checkpoint."""
    print("----------------")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new_segmenter_ckpt = dict()

    if is_pretrain_finetune:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            if "linear_pred" in ckpt_k:
                print(ckpt_k, " is Excluded!")
            else:
                if "module." in ckpt_k:
                    new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v
    else:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            new_segmenter_ckpt[ckpt_k] = ckpt_v
            if "module." in ckpt_k:
                new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v

    ckpt["segmenter"] = new_segmenter_ckpt

    for k, v in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
        else:
            print(v, " is missed!")

    best_val = ckpt.get("best_val", 0)
    epoch_start = ckpt.get("epoch_start", 0)

    if is_pretrain_finetune:
        print_log(
            f"Found [Pretrain] checkpoint at {ckpt_path} with best_val {best_val:.4f} at epoch {epoch_start}"
        )
        return 0, 0
    else:
        print_log(
            f"Found checkpoint at {ckpt_path} with best_val {best_val:.4f} at epoch {epoch_start}"
        )
        return best_val, epoch_start


def train(
    segmenter,
    input_types,
    train_loader,
    optimizer,
    epoch,
    segm_crit,
    freeze_bn,
    print_loss=False,
    amp_enabled=True,
    scaler=None,
):
    """Train for one epoch."""
    train_loader.dataset.set_stage("train")
    segmenter.train()

    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    did_log_once = getattr(train, "_did_log_once", False)

    is_main = _is_main_process()
    iterator = enumerate(train_loader)
    pbar = None
    if is_main:
        pbar = tqdm(iterator, total=len(train_loader))
        iterator = pbar

    for i, sample in iterator:
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample["mask"].cuda().long()

        if i == 0 and _is_main_process():
            try:
                lr_groups = [pg.get("lr", None) for pg in optimizer.param_groups]
                lr_groups_str = ", ".join(
                    [
                        f"g{idx}={lr:.2e}" if isinstance(lr, float) else f"g{idx}=?"
                        for idx, lr in enumerate(lr_groups)
                    ]
                )
                print_log(f"[LR] epoch={epoch} step0: {lr_groups_str}")
            except Exception:
                pass

        if (not did_log_once) and i == 0 and _is_main_process():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            print_log("[Debug] First train iteration tensor shapes:")
            for key_name, tensor in zip(input_types, inputs):
                print_log(
                    f"  - {key_name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
                )
            print_log(
                f"  - mask: shape={tuple(target.shape)} dtype={target.dtype} device={target.device}"
            )

            try:
                alloc = torch.cuda.memory_allocated()
                reserv = torch.cuda.memory_reserved()
                max_alloc = torch.cuda.max_memory_allocated()
                max_reserv = torch.cuda.max_memory_reserved()
                print_log(
                    "[Debug] CUDA memory before forward: "
                    f"allocated={_format_gb(alloc)}, reserved={_format_gb(reserv)}, "
                    f"max_alloc={_format_gb(max_alloc)}, max_reserved={_format_gb(max_reserv)}"
                )
            except Exception:
                pass

        # Forward pass
        try:
            with autocast(enabled=amp_enabled):
                outputs, masks = segmenter(inputs)
        except RuntimeError as e:
            # If the very first forward OOMs, emit memory stats to help diagnosis.
            if (not did_log_once) and _is_main_process() and "out of memory" in str(e).lower():
                try:
                    alloc = torch.cuda.memory_allocated()
                    reserv = torch.cuda.memory_reserved()
                    max_alloc = torch.cuda.max_memory_allocated()
                    max_reserv = torch.cuda.max_memory_reserved()
                    print_log(
                        "[Debug] CUDA memory at OOM: "
                        f"allocated={_format_gb(alloc)}, reserved={_format_gb(reserv)}, "
                        f"max_alloc={_format_gb(max_alloc)}, max_reserved={_format_gb(max_reserv)}"
                    )
                except Exception:
                    pass
            raise

        # Compute loss
        loss = 0
        for output in outputs:
            output = F.interpolate(
                output, size=target.size()[1:], mode="bilinear", align_corners=False
            )
            # Keep loss computation in FP32 for numerical stability.
            soft_output = F.log_softmax(output.float(), dim=1)
            loss += segm_crit(soft_output, target)

        # Backward pass
        optimizer.zero_grad()
        if amp_enabled:
            if scaler is None:
                raise ValueError("AMP enabled but GradScaler is None")
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if print_loss:
            print(f"step: {i:3d}: loss={loss:.2f}", flush=True)

        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        if pbar is not None:
            try:
                lr0 = optimizer.param_groups[0].get("lr", None)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "avg": f"{losses.avg:.3f}",
                        "lr": f"{lr0:.2e}" if isinstance(lr0, float) else "?",
                    }
                )
            except Exception:
                pass

        if (not did_log_once) and i == 0 and _is_main_process():
            try:
                alloc = torch.cuda.memory_allocated()
                reserv = torch.cuda.memory_reserved()
                max_alloc = torch.cuda.max_memory_allocated()
                max_reserv = torch.cuda.max_memory_reserved()
                print_log(
                    "[Debug] CUDA memory after step: "
                    f"allocated={_format_gb(alloc)}, reserved={_format_gb(reserv)}, "
                    f"max_alloc={_format_gb(max_alloc)}, max_reserved={_format_gb(max_reserv)}"
                )
            except Exception:
                pass

            train._did_log_once = True
            did_log_once = True

    if is_main:
        print_log(f"[Train] epoch={epoch} avg_loss={losses.avg:.4f}")


def sliding_window_inference(
    segmenter, inputs, num_classes, window_size=768, stride=512, amp_enabled=True
):
    """Sliding window inference for full resolution images."""
    B, C, H, W = inputs[0].shape
    device = inputs[0].device

    # Initialize output tensors
    output_sum = torch.zeros((B, num_classes, H, W), device=device)
    count_map = torch.zeros((B, 1, H, W), device=device)

    h_start = 0
    while h_start < H:
        h_end = min(h_start + window_size, H)
        if h_end - h_start < window_size and h_start > 0:
            h_start = H - window_size
            h_end = H

        w_start = 0
        while w_start < W:
            w_end = min(w_start + window_size, W)
            if w_end - w_start < window_size and w_start > 0:
                w_start = W - window_size
                w_end = W

            # Crop inputs
            crop_inputs = [x[:, :, h_start:h_end, w_start:w_end] for x in inputs]

            # Forward pass
            with autocast(enabled=amp_enabled):
                outputs, _ = segmenter(crop_inputs)
            crop_output = outputs[-1]  # Use ensemble output

            # Resize if needed
            if crop_output.shape[2:] != (h_end - h_start, w_end - w_start):
                crop_output = F.interpolate(
                    crop_output,
                    size=(h_end - h_start, w_end - w_start),
                    mode='bilinear',
                    align_corners=False
                )

            output_sum[:, :, h_start:h_end, w_start:w_end] += crop_output
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

            if w_end == W:
                break
            w_start += stride

        if h_end == H:
            break
        h_start += stride

    return output_sum / count_map


def validate(
    segmenter, input_types, val_loader, epoch, save_dir, num_classes=19, save_image=0,
    eval_mode='whole', amp_enabled=True, progress_mode: str = "rank0"
):
    """Validate model.

    Args:
        eval_mode: 'whole' for fast validation (resize), 'slide' for accurate (sliding window)
    """
    global best_miou
    val_loader.dataset.set_stage("val")
    segmenter.eval()

    conf_mat = []
    for _ in range(len(input_types) + 1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))

    rank = dist.get_rank() if dist.is_initialized() else 0
    should_print_header = (
        (progress_mode == "all")
        or (progress_mode == "rank0" and _is_main_process())
    )
    if should_print_header:
        print_log(f"Evaluating with mode: {eval_mode}")

    with torch.no_grad():
        all_times = 0
        num_images = 0
        num_saved = 0

        iterator = enumerate(val_loader)
        pbar = None
        show_pbar = (
            (progress_mode == "all")
            or (progress_mode == "rank0" and _is_main_process())
        )
        if show_pbar:
            try:
                desc = f"Val[r{rank}]" if (dist.is_initialized() and dist.get_world_size() > 1) else "Val"
                pbar = tqdm(
                    iterator,
                    total=len(val_loader),
                    desc=desc,
                    leave=False,
                    position=rank if progress_mode == "all" else 0,
                )
                iterator = pbar
            except Exception:
                pbar = None

        for i, sample in iterator:
            inputs = [sample[key].float().cuda() for key in input_types]
            target = sample["mask"]
            # target: (B, H, W)
            gt_batch = target.data.cpu().numpy().astype(np.uint8)

            start_time = time.time()

            if eval_mode == 'slide':
                # Sliding window inference (accurate but slow)
                ensemble_output = sliding_window_inference(
                    segmenter,
                    inputs,
                    num_classes,
                    window_size=768,
                    stride=512,
                    amp_enabled=amp_enabled,
                )
                # For slide mode, we only get ensemble output
                outputs_for_conf = [ensemble_output]
            else:
                # Whole image inference (fast, uses resized input from dataloader)
                with autocast(enabled=amp_enabled):
                    outputs, _ = segmenter(inputs)
                outputs_for_conf = outputs

            end_time = time.time()
            all_times += end_time - start_time

            if pbar is not None:
                try:
                    pbar.set_postfix({"imgs": num_images + gt_batch.shape[0], "lat": f"{(end_time-start_time):.3f}s"})
                except Exception:
                    pass

            if (progress_mode == "all" or _is_main_process()) and (i == 0 or (i + 1) % 50 == 0):
                try:
                    prefix = f"[Val r{rank}]" if (dist.is_initialized() and dist.get_world_size() > 1) else "[Val]"
                    print_log(f"{prefix} step={i+1}/{len(val_loader)} batch={gt_batch.shape[0]} total_imgs={num_images + gt_batch.shape[0]}")
                except Exception:
                    pass

            # Process outputs for confusion matrix
            if eval_mode == 'slide':
                # Only process ensemble output for slide mode
                label_size = target.shape[1:]
                # ensemble_output: (B, C, H, W)
                output_up = F.interpolate(
                    ensemble_output[:, :num_classes],
                    size=label_size,
                    mode="bilinear",
                    align_corners=False,
                )
                pred_batch = output_up.argmax(dim=1).cpu().numpy().astype(np.uint8)
                for b in range(pred_batch.shape[0]):
                    gt = gt_batch[b]
                    pred = pred_batch[b]
                    gt_idx = gt < num_classes  # Ignore labels >= num_classes
                    conf_mat[-1] += confusion_matrix(gt[gt_idx], pred[gt_idx], num_classes)
            else:
                label_size = target.shape[1:]
                last_pred_batch = None
                for idx, output in enumerate(outputs_for_conf):
                    if idx >= len(conf_mat):
                        break
                    output_up = F.interpolate(
                        output[:, :num_classes],
                        size=label_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_batch = output_up.argmax(dim=1).cpu().numpy().astype(np.uint8)
                    last_pred_batch = pred_batch

                    for b in range(pred_batch.shape[0]):
                        gt = gt_batch[b]
                        pred = pred_batch[b]
                        gt_idx = gt < num_classes  # Ignore labels >= num_classes
                        conf_mat[idx] += confusion_matrix(gt[gt_idx], pred[gt_idx], num_classes)

                if _is_main_process() and (save_image == -1 or num_saved < save_image):
                    # Save only the first item in the batch to keep output predictable.
                    b = 0
                    gt0 = gt_batch[b]
                    pred0 = (
                        last_pred_batch[b]
                        if last_pred_batch is not None
                        else gt0
                    )
                    img = make_validation_img(
                        inputs[0][b:b+1].data.cpu().numpy(),
                        inputs[1][b:b+1].data.cpu().numpy(),
                        sample["mask"][b:b+1].data.cpu().numpy(),
                        pred0[np.newaxis, :],
                    )
                    imgs_folder = os.path.join(save_dir, "imgs")
                    os.makedirs(imgs_folder, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(imgs_folder, f"validate_{num_saved}.png"),
                        img[:, :, ::-1],
                    )
                    num_saved += 1

            num_images += gt_batch.shape[0]

        latency = all_times / max(1, num_images)
        if should_print_header:
            print(f"all_times: {all_times}  num_images: {num_images}  latency: {latency}")

    # Aggregate confusion matrices across ranks for correct multi-GPU evaluation.
    if dist.is_initialized() and dist.get_world_size() > 1:
        for idx in range(len(conf_mat)):
            t = torch.as_tensor(conf_mat[idx], dtype=torch.int64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            conf_mat[idx] = t.cpu().numpy()

    # UAVScenes class names
    uavscenes_classes = [
        "background", "roof", "dirt_road", "paved_road", "river", "pool",
        "bridge", "container", "airstrip", "traffic_barrier", "green_field",
        "wild_field", "solar_panel", "umbrella", "transparent_roof", "car_park",
        "paved_walk", "sedan", "truck"
    ]

    ens_miou_out = 0.0
    if _is_main_process():
        for idx, input_type in enumerate(input_types + ["ens"]):
            glob, mean, iou = getScores(conf_mat[idx])
            miou = float(iou)  # percent
            best_note = ""
            if input_type == "ens":
                ens_miou_out = miou
                if miou > best_miou:
                    best_miou = miou
                    best_note = "    (best)"

            input_type_str = f"({input_type})"
            print_log(
                f"Epoch {epoch:<4d} {input_type_str:<7s}   glob_acc={glob:<5.2f}    mean_acc={mean:<5.2f}    mIoU={miou:<5.2f}        {best_note}"
            )

            if input_type == "ens":
                per_class_iou = getPerClassIoU(conf_mat[idx])
                print_log("\nPer-class IoU:")
                for class_idx, cls_name in enumerate(uavscenes_classes):
                    marker = "[D]" if class_idx >= 17 else "[S]"
                    print_log(
                        f"  {marker} {cls_name:<18} {per_class_iou[class_idx]:>6.2f}%"
                    )

        print_log("")

    # Make sure all ranks return the same scalar.
    if dist.is_initialized() and dist.get_world_size() > 1:
        miou_tensor = torch.tensor([ens_miou_out], dtype=torch.float32, device="cuda")
        dist.broadcast(miou_tensor, src=0)
        ens_miou_out = float(miou_tensor.item())

    return ens_miou_out


def main():
    global args, best_miou
    best_miou = 0.0
    args = get_arguments()

    # Dataset configuration
    if args.dataset == "uavscenes":
        args.num_classes = 19
        input_scale = [768, 768]  # Fair comparison with CMNeXt/DFormerv2
    elif args.dataset == "nyudv2":
        args.train_list = "data/nyudv2/train.txt"
        args.val_list = "data/nyudv2/val.txt"
        args.num_classes = 40
        input_scale = [480, 640]
    elif args.dataset == "sunrgbd":
        args.train_list = "data/sun/train.txt"
        args.val_list = "data/sun/test.txt"
        args.num_classes = 37
        input_scale = [480, 480]

    args.num_stages = 3
    gpu = setup_ddp()

    # Create checkpoint directory
    ckpt_dir = os.path.join("ckpt", args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Setup logger
    log_path = os.path.join(ckpt_dir, "log.txt") if _is_main_process() else os.devnull
    helpers.logger = open(log_path, "w+")
    print_log(" ".join(sys.argv))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Create model
    segmenter, param_groups = create_segmenter(
        args.num_classes,
        gpu,
        args.backbone,
        args.n_heads,
        args.dpr,
        args.drop_rate,
    )

    # AMP scaler (enabled/disabled via args.amp)
    scaler = GradScaler(enabled=args.amp)

    total_params = compute_params(segmenter)
    trainable_params = sum(p.numel() for p in segmenter.parameters() if p.requires_grad)
    print_log(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Calculate FLOPs
    if FVCORE_AVAILABLE:
        try:
            dummy_rgb = torch.zeros(1, 3, 768, 768).cuda()
            dummy_hag = torch.zeros(1, 3, 768, 768).cuda()
            segmenter.eval()
            # WeTr expects input as list [rgb, hag]
            flops = FlopCountAnalysis(segmenter, ([dummy_rgb, dummy_hag],))
            print_log(f"FLOPs: {flops.total() / 1e9:.2f}G")
            segmenter.train()
        except Exception as e:
            print_log(f"Could not calculate FLOPs: {e}")
    else:
        print_log("FLOPs: fvcore not installed (pip install fvcore)")

    # Resume from checkpoint
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(
                args.resume,
                {"segmenter": segmenter},
                is_pretrain_finetune=args.is_pretrain_finetune,
            )
        else:
            print_log(f"=> no checkpoint found at '{args.resume}'")
            return

    # Setup distributed wrapper (FSDP preferred; otherwise DDP)
    if args.fsdp:
        segmenter = _maybe_wrap_fsdp(segmenter, gpu, args)
    elif dist.is_initialized():
        segmenter = DDP(
            segmenter, device_ids=[gpu], output_device=gpu, find_unused_parameters=False
        )

    epoch_current = epoch_start

    # Loss function
    segm_crit = nn.NLLLoss(ignore_index=args.ignore_label).cuda()

    # Saver
    saver = Saver(
        args=vars(args),
        ckpt_dir=ckpt_dir,
        best_val=best_val,
        condition=lambda x, y: x > y,
    )

    # Learning rates for each stage
    lrs = [args.lr_0, args.lr_1, args.lr_2]

    print("-------------------------Optimizer Params--------------------")
    print(f"weight_decay: {args.weight_decay}")
    print(f"lrs: {lrs}")
    print("----------------------------------------------------------------")

    # Training loop
    for task_idx in range(args.num_stages):
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue

        start = time.time()
        torch.cuda.empty_cache()

        # Create data loaders
        if args.dataset == "uavscenes":
            train_loader, val_loader, test_loader, train_sampler = create_uavscenes_loaders(
                args, input_scale
            )
        else:
            # For NYUDv2/SUNRGBD, use original loader
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")

        # Calculate warmup_iter/max_iter using the real iters_per_epoch (fair comparison with CMNeXt)
        iters_per_epoch = len(train_loader)
        warmup_epochs = getattr(args, 'warmup_epochs', 3)
        warmup_iter = warmup_epochs * iters_per_epoch
        total_epochs = sum(args.num_epoch)  # 60 epochs (3 stages × 20 epochs)
        max_iter = total_epochs * iters_per_epoch
        power = getattr(args, 'power', 0.9)  # CMNeXt paper setting
        warmup_ratio = getattr(args, 'warmup_ratio', 0.1)  # CMNeXt paper setting
        if _is_main_process():
            print_log(
                f"[LR] stage={task_idx} iters_per_epoch={iters_per_epoch} warmup_epochs={warmup_epochs} "
                f"warmup_iter={warmup_iter} total_epochs={total_epochs} max_iter={max_iter} "
                f"warmup_ratio={warmup_ratio} power={power} base_lr={lrs[task_idx]:.2e}"
            )

        optimizer = PolyWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": lrs[task_idx],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": lrs[task_idx] * 10,
                    "weight_decay": args.weight_decay,
                },
            ],
            lr=lrs[task_idx],
            weight_decay=args.weight_decay,
            betas=[0.9, 0.999],
            warmup_iter=warmup_iter,
            max_iter=max_iter,
            warmup_ratio=warmup_ratio,
            power=power,
        )

        # Evaluation only
        if args.evaluate:
            try:
                iou = validate(
                    segmenter,
                    args.input,
                    val_loader,
                    0,
                    ckpt_dir,
                    num_classes=args.num_classes,
                    save_image=args.save_image,
                    amp_enabled=args.amp,
                    progress_mode=args.val_progress,
                )
                if _is_main_process():
                    print_log(f"Evaluation finished. mIoU={iou:.2f}")
                return
            finally:
                try:
                    helpers.logger.close()
                except Exception:
                    pass
                cleanup_ddp()

        print_log(f"Training Stage {task_idx}")

        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train(
                segmenter,
                args.input,
                train_loader,
                optimizer,
                epoch_current,
                segm_crit,
                args.freeze_bn,
                args.print_loss,
                amp_enabled=args.amp,
                scaler=scaler,
            )

            if (epoch + 1) % args.val_every == 0:
                miou = validate(
                    segmenter,
                    args.input,
                    val_loader,
                    epoch_current,
                    ckpt_dir,
                    args.num_classes,
                    amp_enabled=args.amp,
                    progress_mode=args.val_progress,
                )
                if args.fsdp and dist.is_initialized() and dist.get_world_size() > 1:
                    state_to_save = _segmenter_state_dict_for_save(segmenter, args)
                    if _is_main_process():
                        saver.save(
                            miou,
                            {"segmenter": state_to_save, "epoch_start": epoch_current},
                        )
                else:
                    if _is_main_process():
                        saver.save(
                            miou,
                            {
                                "segmenter": _segmenter_state_dict_for_save(segmenter, args),
                                "epoch_start": epoch_current,
                            },
                        )

            epoch_current += 1

        print_log(
            f"Stage {task_idx} finished, time spent {(time.time() - start) / 60.0:.3f}min\n"
        )

    print_log(f"All stages finished. Best Val is {saver.best_val:.3f}")

    # Final evaluation on test set
    print_log("\n" + "=" * 60)
    print_log("Final evaluation on TEST set")
    print_log("=" * 60)

    # Load best model
    best_ckpt = os.path.join(ckpt_dir, "ckpt.pth")
    if os.path.exists(best_ckpt):
        print_log(f"Loading best checkpoint: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        if "segmenter" in ckpt:
            if args.fsdp and dist.is_initialized() and dist.get_world_size() > 1:
                # FULL state dict load: all ranks must participate.
                full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FSDP.state_dict_type(segmenter, StateDictType.FULL_STATE_DICT, full_cfg):
                    segmenter.load_state_dict(ckpt["segmenter"], strict=False)
            elif isinstance(segmenter, DDP):
                segmenter.module.load_state_dict(ckpt["segmenter"], strict=False)
            else:
                segmenter.load_state_dict(ckpt["segmenter"], strict=False)

    # Evaluate on test set with slide mode using UAVScenesMetrics for detailed output
    from utils.metrics import UAVScenesMetrics

    segmenter.eval()
    test_metrics = UAVScenesMetrics(num_classes=args.num_classes, ignore_label=255)

    print_log("Running test evaluation with sliding window inference...")
    total_time = 0
    num_images = 0

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs = [sample[key].float().cuda() for key in args.input]
            target = sample["mask"]

            # Sliding window inference with timing
            torch.cuda.synchronize()
            start_time = time.time()

            ensemble_output = sliding_window_inference(
                segmenter,
                inputs,
                args.num_classes,
                window_size=768,
                stride=512,
                amp_enabled=args.amp,
            )

            torch.cuda.synchronize()
            total_time += time.time() - start_time
            num_images += 1

            pred = ensemble_output.argmax(dim=1).cpu().numpy()
            gt = target[0].data.cpu().numpy()

            test_metrics.update(pred, gt)

            if (i + 1) % 50 == 0 and _is_main_process():
                print_log(f'Test: {i + 1}/{len(test_loader)} samples')

    # Calculate inference speed
    avg_time_ms = (total_time / num_images) * 1000
    fps = num_images / total_time

    # Print detailed results
    if _is_main_process():
        test_metrics.print_results()

    # Print and save inference speed
    if _is_main_process():
        print_log(f"\nInference speed:")
        print_log(f"  Average time per image: {avg_time_ms:.1f}ms")
        print_log(f"  FPS: {fps:.2f}")

    # Save results to file
    if _is_main_process():
        results_dir = os.path.join(args.output_dir, 'results')
        test_metrics.save_results(results_dir, 'GeminiFusion', avg_time_ms, fps, num_images)

    helpers.logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
