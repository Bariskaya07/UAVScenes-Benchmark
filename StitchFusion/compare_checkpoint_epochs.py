#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import config
from dataloader.UAVScenesDataset import UAVScenesDataset
from dataloader.dataloader import ValPre
from models.builder import EncoderDecoder as segmodel
from utils.metric import compute_detailed_score, compute_score, hist_info
from utils.pyt_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare StitchFusion checkpoints on val/test split with per-class IoU."
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[20, 25, 30],
        help="Epoch checkpoints to compare.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for whole-image batched validation.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="stitchfusion_checkpoints",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def summarize_metrics(hist, correct, labeled):
    iou, mean_iou, _, freq_iou, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    detailed = compute_detailed_score(hist)
    return {
        "hist": hist,
        "iou": iou,
        "mean_iou": mean_iou * 100.0,
        "freq_iou": freq_iou * 100.0,
        "mean_pixel_acc": mean_pixel_acc * 100.0,
        "pixel_acc": pixel_acc * 100.0,
        "static_miou": detailed["static_mIoU"] * 100.0,
        "dynamic_miou": detailed["dynamic_mIoU"] * 100.0,
    }


def evaluate_checkpoints_one_pass(models_by_epoch, dataset, device, batch_size):
    for model in models_by_epoch.values():
        model.eval()

    state_by_epoch = {
        epoch: {
            "hist": np.zeros((config.num_classes, config.num_classes), dtype=np.float64),
            "correct": 0,
            "labeled": 0,
        }
        for epoch in models_by_epoch
    }

    mean = config.norm_mean.reshape(1, 1, 3).astype(np.float32)
    std = config.norm_std.reshape(1, 1, 3).astype(np.float32)

    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating checkpoints", leave=False):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_rgb = []
            batch_modal = []
            batch_labels = []

            for idx in range(batch_start, batch_end):
                dd = dataset[idx]
                img = dd["data"].astype(np.float32) / 255.0
                modal = dd["modal_x"].astype(np.float32)
                batch_rgb.append(((img - mean) / std).transpose(2, 0, 1))
                batch_modal.append(((modal - 0.5) / 0.5).transpose(2, 0, 1))
                batch_labels.append(dd["label"])

            rgb_t = torch.from_numpy(np.stack(batch_rgb)).float().to(device)
            modal_t = torch.from_numpy(np.stack(batch_modal)).float().to(device)
            for epoch, model in models_by_epoch.items():
                preds = model(rgb_t, modal_t).argmax(dim=1).cpu().numpy()
                state = state_by_epoch[epoch]
                for pred, label in zip(preds, batch_labels):
                    hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
                    state["hist"] += hist_tmp
                    state["correct"] += correct_tmp
                    state["labeled"] += labeled_tmp

    return {
        epoch: summarize_metrics(state["hist"], state["correct"], state["labeled"])
        for epoch, state in state_by_epoch.items()
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def main():
    args = parse_args()
    device = torch.device(args.device)

    data_setting = {"data_root": config.dataset_path}
    dataset = UAVScenesDataset(data_setting, args.split, ValPre(resize_to=(config.image_height, config.image_width)))

    ckpt_dir = Path(args.checkpoint_dir)
    models_by_epoch = {}

    for epoch in args.epochs:
        ckpt = ckpt_dir / f"stitchfusion_epoch_{epoch}.pth"
        if not ckpt.exists():
            print(f"[WARN] Missing checkpoint: {ckpt}")
            continue
        print(f"\nLoading {ckpt.name} on {device} ...")
        criterion = None
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d).to(device)
        load_model(model, str(ckpt))
        models_by_epoch[epoch] = model

    if not models_by_epoch:
        raise SystemExit("No requested checkpoints could be evaluated.")

    print(f"\nRunning one-pass comparison for epochs: {', '.join(map(str, models_by_epoch))}")
    results = evaluate_checkpoints_one_pass(models_by_epoch, dataset, device, args.batch_size)

    epochs = [epoch for epoch in args.epochs if epoch in results]

    print("\nSummary")
    print("=" * 104)
    print(
        f"{'Epoch':>5}  {'mIoU':>8}  {'Static':>8}  {'Dynamic':>8}  "
        f"{'freq_IoU':>10}  {'mean_pacc':>10}  {'pixel_acc':>10}"
    )
    print("-" * 104)
    for epoch in epochs:
        res = results[epoch]
        print(
            f"{epoch:>5}  {res['mean_iou']:>8.3f}  {res['static_miou']:>8.3f}  {res['dynamic_miou']:>8.3f}  "
            f"{res['freq_iou']:>10.3f}  {res['mean_pixel_acc']:>10.3f}  {res['pixel_acc']:>10.3f}"
        )

    base_epoch = epochs[0]
    print("\nPer-class IoU comparison")
    print("=" * 104)
    header = f"{'Class':<24} {base_epoch:>8}"
    for epoch in epochs[1:]:
        header += f"  {epoch:>8}  {'Δvs'+str(base_epoch):>10}"
    print(header)
    print("-" * 104)

    for idx, class_name in enumerate(dataset.class_names):
        row = f"{class_name:<24} {fmt(results[base_epoch]['iou'][idx] * 100.0):>8}"
        base_value = results[base_epoch]["iou"][idx] * 100.0
        for epoch in epochs[1:]:
            value = results[epoch]["iou"][idx] * 100.0
            delta = None
            if not math.isnan(base_value) and not math.isnan(value):
                delta = value - base_value
            row += f"  {fmt(value):>8}  {fmt(delta):>10}"
        print(row)


if __name__ == "__main__":
    main()
