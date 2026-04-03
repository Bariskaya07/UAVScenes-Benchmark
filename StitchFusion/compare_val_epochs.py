#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path


EPOCH_MIOU_RE = re.compile(r"Epoch\s+(\d+):\s+mIoU\s*=\s*([0-9.]+)")
SUMMARY_RE = re.compile(
    r"mean_IoU\s+([0-9.]+)%.*freq_IoU\s+([0-9.]+)%.*mean_pixel_acc\s+([0-9.]+)%.*pixel_acc\s+([0-9.]+)%",
)
CLASS_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s+((?:nan)|(?:\d+(?:\.\d+)?))%$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare StitchFusion validation results for selected epochs."
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to train_stitchfusion.log on the VM.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[20, 25, 30],
        help="Epoch numbers to compare.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional stitchfusion_checkpoints directory to inspect best.pth / *_best files.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    if value == "nan":
        return math.nan
    return float(value)


def parse_log(log_path: Path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    results = {}

    current_epoch = None
    for idx, line in enumerate(lines):
        match = EPOCH_MIOU_RE.search(line)
        if match:
            current_epoch = int(match.group(1))
            results[current_epoch] = {
                "epoch_miou": float(match.group(2)) * 100.0,
                "mean_iou": None,
                "freq_iou": None,
                "mean_pixel_acc": None,
                "pixel_acc": None,
                "classes": {},
                "line_no": idx + 1,
            }
            continue

        if current_epoch is None:
            continue

        summary_match = SUMMARY_RE.search(line)
        if summary_match:
            results[current_epoch]["mean_iou"] = float(summary_match.group(1))
            results[current_epoch]["freq_iou"] = float(summary_match.group(2))
            results[current_epoch]["mean_pixel_acc"] = float(summary_match.group(3))
            results[current_epoch]["pixel_acc"] = float(summary_match.group(4))
            continue

        class_match = CLASS_RE.match(line)
        if class_match:
            class_idx = int(class_match.group(1))
            class_name = class_match.group(2).strip()
            value = _to_float(class_match.group(3))
            results[current_epoch]["classes"][class_name] = {
                "index": class_idx,
                "iou": value,
            }

    return results


def inspect_checkpoint_dir(checkpoint_dir: Path):
    best_link = checkpoint_dir / "best.pth"
    last_link = checkpoint_dir / "last.pth"
    best_target = best_link.resolve().name if best_link.exists() or best_link.is_symlink() else None
    last_target = last_link.resolve().name if last_link.exists() or last_link.is_symlink() else None
    best_files = sorted(p.name for p in checkpoint_dir.glob("stitchfusion_epoch_*_best.pth"))
    return {
        "best_target": best_target,
        "last_target": last_target,
        "best_files": best_files,
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def main():
    args = parse_args()
    log_path = Path(args.log)
    results = parse_log(log_path)

    requested_epochs = args.epochs
    print("Validation summary")
    print("=" * 94)
    print(
        f"{'Epoch':>5}  {'Log mIoU':>9}  {'mean_IoU':>9}  {'freq_IoU':>9}  "
        f"{'mean_pacc':>10}  {'pixel_acc':>9}  {'log_line':>8}"
    )
    print("-" * 94)
    for epoch in requested_epochs:
        entry = results.get(epoch)
        if entry is None:
            print(f"{epoch:>5}  {'MISSING':>9}  {'-':>9}  {'-':>9}  {'-':>10}  {'-':>9}  {'-':>8}")
            continue
        print(
            f"{epoch:>5}  "
            f"{fmt(entry['epoch_miou']):>9}  "
            f"{fmt(entry['mean_iou']):>9}  "
            f"{fmt(entry['freq_iou']):>9}  "
            f"{fmt(entry['mean_pixel_acc']):>10}  "
            f"{fmt(entry['pixel_acc']):>9}  "
            f"{entry['line_no']:>8}"
        )

    available = [epoch for epoch in requested_epochs if epoch in results]
    if len(available) >= 2:
        base_epoch = available[0]
        print("\nPer-class IoU deltas vs epoch %d" % base_epoch)
        print("=" * 94)
        class_names = set()
        for epoch in available:
            class_names.update(results[epoch]["classes"].keys())
        header = f"{'Class':<24} {base_epoch:>8}"
        for epoch in available[1:]:
            header += f"  {epoch:>8}  {'Δvs'+str(base_epoch):>10}"
        print(header)
        print("-" * 94)
        for class_name in sorted(class_names, key=lambda name: results[base_epoch]["classes"].get(name, {}).get("index", 999)):
            row = f"{class_name:<24}"
            base_value = results[base_epoch]["classes"].get(class_name, {}).get("iou")
            row += f" {fmt(base_value):>8}"
            for epoch in available[1:]:
                value = results[epoch]["classes"].get(class_name, {}).get("iou")
                delta = None
                if value is not None and base_value is not None and not math.isnan(value) and not math.isnan(base_value):
                    delta = value - base_value
                row += f"  {fmt(value):>8}  {fmt(delta):>10}"
            print(row)

    if args.checkpoint_dir:
        info = inspect_checkpoint_dir(Path(args.checkpoint_dir))
        print("\nCheckpoint state")
        print("=" * 94)
        print(f"best.pth -> {info['best_target'] or 'MISSING'}")
        print(f"last.pth -> {info['last_target'] or 'MISSING'}")
        print("Available *_best files:")
        if info["best_files"]:
            for name in info["best_files"]:
                print(f"  - {name}")
        else:
            print("  - none")


if __name__ == "__main__":
    main()
