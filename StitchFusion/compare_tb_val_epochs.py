#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tensorboard is required for this script. Install it with `pip install tensorboard`."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read StitchFusion val_mIoU scalars from TensorBoard event files."
    )
    parser.add_argument(
        "--tb-dir",
        required=True,
        help="Path to StitchFusion TensorBoard directory, e.g. log_UAVScenes_mit_b2/tb",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[20, 25, 30],
        help="Epoch numbers to print.",
    )
    parser.add_argument(
        "--tag",
        default="val_mIoU",
        help="Scalar tag to inspect.",
    )
    return parser.parse_args()


def find_event_files(tb_dir: Path):
    return sorted(tb_dir.rglob("events.out.tfevents.*"))


def load_scalars(event_file: Path, tag: str):
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return acc.Scalars(tag)


def main():
    args = parse_args()
    tb_dir = Path(args.tb_dir)
    event_files = find_event_files(tb_dir)
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {tb_dir}")

    all_scalars = []
    for event_file in event_files:
        all_scalars.extend(load_scalars(event_file, args.tag))

    if not all_scalars:
        raise SystemExit(f"No scalar tag '{args.tag}' found under {tb_dir}")

    by_step = {}
    for item in all_scalars:
        by_step[int(item.step)] = float(item.value) * 100.0

    print(f"TensorBoard tag: {args.tag}")
    print(f"Event files scanned: {len(event_files)}")
    print("=" * 48)
    print(f"{'Epoch':>5}  {'val_mIoU (%)':>12}")
    print("-" * 48)
    for epoch in args.epochs:
        value = by_step.get(epoch)
        if value is None:
            print(f"{epoch:>5}  {'MISSING':>12}")
        else:
            print(f"{epoch:>5}  {value:>12.4f}")

    best_epoch = max(by_step, key=by_step.get)
    print("-" * 48)
    print(f"Best epoch in TB scalars: {best_epoch} ({by_step[best_epoch]:.4f}%)")


if __name__ == "__main__":
    main()
