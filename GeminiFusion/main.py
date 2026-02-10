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

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import utils.helpers as helpers
from utils.optimizer import PolyWarmupAdamW
from models.segformer import WeTr
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.augmentations_mm import *
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.uavscenes import UAVScenesDataset


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
    parser.add_argument(
        "--num-epoch",
        type=int,
        nargs="+",
        default=[100, 100, 100],
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

    # Validation transforms
    composed_val = transforms.Compose([
        ToTensor(),
        Resize(input_scale),
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
        split='test',
        transform=composed_val,
        hag_max_height=args.hag_max_height,
    )

    print_log(f"Created train set {len(trainset)} examples, val set {len(validset)} examples")

    # Create samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(
            trainset, dist.get_world_size(), dist.get_rank(), shuffle=True
        )
    else:
        train_sampler = None

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
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def load_ckpt(ckpt_path, ckpt_dict, is_pretrain_finetune=False):
    """Load checkpoint."""
    print("----------------")
    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample["mask"].cuda().long()

        # Forward pass
        outputs, masks = segmenter(inputs)

        # Compute loss
        loss = 0
        for output in outputs:
            output = F.interpolate(
                output, size=target.size()[1:], mode="bilinear", align_corners=False
            )
            soft_output = F.log_softmax(output, dim=1)
            loss += segm_crit(soft_output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if print_loss:
            print(f"step: {i:3d}: loss={loss:.2f}", flush=True)

        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)


def validate(
    segmenter, input_types, val_loader, epoch, save_dir, num_classes=19, save_image=0
):
    """Validate model."""
    global best_iou
    val_loader.dataset.set_stage("val")
    segmenter.eval()

    conf_mat = []
    for _ in range(len(input_types) + 1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))

    with torch.no_grad():
        all_times = 0
        count = 0

        for i, sample in enumerate(val_loader):
            inputs = [sample[key].float().cuda() for key in input_types]
            target = sample["mask"]
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes  # Ignore labels >= num_classes

            start_time = time.time()
            outputs, _ = segmenter(inputs)
            end_time = time.time()
            all_times += end_time - start_time

            for idx, output in enumerate(outputs):
                output = (
                    cv2.resize(
                        output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                        target.size()[1:][::-1],
                        interpolation=cv2.INTER_CUBIC,
                    )
                    .argmax(axis=2)
                    .astype(np.uint8)
                )
                # Compute IoU
                conf_mat[idx] += confusion_matrix(
                    gt[gt_idx], output[gt_idx], num_classes
                )

                if i < save_image or save_image == -1:
                    img = make_validation_img(
                        inputs[0].data.cpu().numpy(),
                        inputs[1].data.cpu().numpy(),
                        sample["mask"].data.cpu().numpy(),
                        output[np.newaxis, :],
                    )
                    imgs_folder = os.path.join(save_dir, "imgs")
                    os.makedirs(imgs_folder, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(imgs_folder, f"validate_{i}.png"),
                        img[:, :, ::-1],
                    )

            count += 1

        latency = all_times / count
        print(f"all_times: {all_times}  count: {count}  latency: {latency}")

    # Print results
    for idx, input_type in enumerate(input_types + ["ens"]):
        glob, mean, iou = getScores(conf_mat[idx])
        best_iou_note = ""
        if iou > best_iou:
            best_iou = iou
            best_iou_note = "    (best)"

        input_type_str = f"({input_type})"
        print_log(
            f"Epoch {epoch:<4d} {input_type_str:<7s}   glob_acc={glob:<5.2f}    mean_acc={mean:<5.2f}    IoU={iou:<5.2f}        {best_iou_note}"
        )

    print_log("")
    return iou


def main():
    global args, best_iou
    best_iou = 0
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
    helpers.logger = open(os.path.join(ckpt_dir, "log.txt"), "w+")
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

    print_log(
        f"Loaded Segmenter {args.backbone}, #PARAMS={compute_params(segmenter) / 1e6:3.2f}M"
    )

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

    # Setup DDP
    no_ddp_segmenter = segmenter
    if dist.is_initialized():
        segmenter = DDP(
            segmenter, device_ids=[gpu], output_device=0, find_unused_parameters=False
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
            warmup_iter=1500,
            max_iter=40000,
            warmup_ratio=1e-6,
            power=1.0,
        )

        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue

        start = time.time()
        torch.cuda.empty_cache()

        # Create data loaders
        if args.dataset == "uavscenes":
            train_loader, val_loader, train_sampler = create_uavscenes_loaders(
                args, input_scale
            )
        else:
            # For NYUDv2/SUNRGBD, use original loader
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")

        # Evaluation only
        if args.evaluate:
            return validate(
                no_ddp_segmenter,
                args.input,
                val_loader,
                0,
                ckpt_dir,
                num_classes=args.num_classes,
                save_image=args.save_image,
            )

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
            )

            if (epoch + 1) % args.val_every == 0:
                miou = validate(
                    no_ddp_segmenter,
                    args.input,
                    val_loader,
                    epoch_current,
                    ckpt_dir,
                    args.num_classes,
                )
                saver.save(
                    miou,
                    {"segmenter": segmenter.state_dict(), "epoch_start": epoch_current},
                )

            epoch_current += 1

        print_log(
            f"Stage {task_idx} finished, time spent {(time.time() - start) / 60.0:.3f}min\n"
        )

    print_log(f"All stages finished. Best Val is {saver.best_val:.3f}")
    helpers.logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
