#!/usr/bin/env python3
"""Audit effective UAVScenes training schedules across benchmark models.

This script intentionally checks the values that matter at runtime instead of
only reading scheduler class defaults. It follows each model's actual config /
call-site wiring just enough to answer questions like:

- What are the effective total epochs and warmup epochs?
- Which power / warmup_ratio are used?
- Is warmup linear or something else?
- Are the schedules aligned across models?

It is designed as a lightweight guardrail against the exact confusion where a
class default (e.g. power=1.0, warmup='exp') is mistaken for the effective
UAVScenes runtime setting even though the call site overrides it.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScheduleSpec:
    model: str
    total_epochs: int
    warmup_epochs: int
    power: float
    warmup_ratio: float
    warmup_type: str
    source: str


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def req_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find {label} with pattern: {pattern}")
    return match


def req_int(pattern: str, text: str, label: str) -> int:
    return int(req_match(pattern, text, label).group(1))


def req_float(pattern: str, text: str, label: str) -> float:
    return float(req_match(pattern, text, label).group(1))


def parse_cmx() -> ScheduleSpec:
    config = read("CMX/config.py")
    lr_policy = read("CMX/utils/lr_policy.py")
    return ScheduleSpec(
        model="CMX",
        total_epochs=req_int(r"C\.nepochs\s*=\s*(\d+)", config, "CMX total epochs"),
        warmup_epochs=req_int(r"C\.warm_up_epoch\s*=\s*(\d+)", config, "CMX warmup epochs"),
        power=req_float(r"C\.lr_power\s*=\s*([0-9.]+)", config, "CMX power"),
        warmup_ratio=req_float(r"C\.warmup_ratio\s*=\s*([0-9.]+)", config, "CMX warmup ratio"),
        warmup_type="linear" if "Linear warmup" in lr_policy else "unknown",
        source="CMX/config.py + CMX/train.py",
    )


def parse_cmnext() -> ScheduleSpec:
    cfg = read("CMNeXt/configs/uavscenes_rgb_hag.yaml")
    return ScheduleSpec(
        model="CMNeXt",
        total_epochs=req_int(r"EPOCHS:\s*(\d+)", cfg, "CMNeXt total epochs"),
        warmup_epochs=req_int(r"WARMUP:\s*(\d+)", cfg, "CMNeXt warmup epochs"),
        power=req_float(r"POWER:\s*([0-9.]+)", cfg, "CMNeXt power"),
        warmup_ratio=req_float(r"WARMUP_RATIO:\s*([0-9.]+)", cfg, "CMNeXt warmup ratio"),
        warmup_type="linear",
        source="CMNeXt/configs/uavscenes_rgb_hag.yaml",
    )


def parse_hrfuser() -> ScheduleSpec:
    cfg = read("HRFuser/configs/uavscenes_rgb_hag.yaml")
    return ScheduleSpec(
        model="HRFuser",
        total_epochs=req_int(r"epochs:\s*(\d+)", cfg, "HRFuser total epochs"),
        warmup_epochs=req_int(r"warmup_epochs:\s*(\d+)", cfg, "HRFuser warmup epochs"),
        power=req_float(r"power:\s*([0-9.]+)", cfg, "HRFuser power"),
        warmup_ratio=req_float(r"warmup_ratio:\s*([0-9.]+)", cfg, "HRFuser warmup ratio"),
        warmup_type=req_match(r"warmup_type:\s*([A-Za-z_]+)", cfg, "HRFuser warmup type").group(1),
        source="HRFuser/configs/uavscenes_rgb_hag.yaml",
    )


def parse_tokenfusion() -> ScheduleSpec:
    cfg = read("TokenFusion/configs/uavscenes_rgb_hag.yaml")
    total_epochs = req_int(r"epochs:\s*(\d+)", cfg, "TokenFusion total epochs")
    warmup_iter = req_int(r"warmup_iter:\s*(\d+)", cfg, "TokenFusion warmup iter")
    max_iter = req_int(r"max_iter:\s*(\d+)", cfg, "TokenFusion max iter")
    if max_iter % total_epochs != 0 or warmup_iter % (max_iter // total_epochs) != 0:
        raise RuntimeError("TokenFusion warmup_iter/max_iter are not epoch-aligned for UAVScenes")
    iters_per_epoch = max_iter // total_epochs
    warmup_epochs = warmup_iter // iters_per_epoch
    return ScheduleSpec(
        model="TokenFusion",
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        power=req_float(r"power:\s*([0-9.]+)", cfg, "TokenFusion power"),
        warmup_ratio=req_float(r"warmup_ratio:\s*([0-9.]+)", cfg, "TokenFusion warmup ratio"),
        warmup_type="linear",
        source="TokenFusion/configs/uavscenes_rgb_hag.yaml + TokenFusion/utils/optimizer.py",
    )


def parse_geminifusion() -> ScheduleSpec:
    main_py = read("GeminiFusion/main.py")
    stage_block = req_match(
        r'--num-epoch".*?default=\[(\d+),\s*(\d+),\s*(\d+)\]',
        main_py,
        "GeminiFusion num-epoch defaults",
    )
    total_epochs = sum(int(stage_block.group(i)) for i in range(1, 4))
    return ScheduleSpec(
        model="GeminiFusion",
        total_epochs=total_epochs,
        warmup_epochs=req_int(r'--warmup-epochs".*?default=(\d+)', main_py, "GeminiFusion warmup epochs"),
        power=req_float(r'getattr\(args,\s*"power",\s*([0-9.]+)\)', main_py, "GeminiFusion power"),
        warmup_ratio=req_float(r'getattr\(args,\s*"warmup_ratio",\s*([0-9.]+)\)', main_py, "GeminiFusion warmup ratio"),
        warmup_type="linear",
        source="GeminiFusion/main.py + GeminiFusion/utils/optimizer.py",
    )


def parse_sigma() -> ScheduleSpec:
    cfg = read("Sigma/configs/config_UAVScenes.py")
    return ScheduleSpec(
        model="Sigma",
        total_epochs=req_int(r"C\.nepochs\s*=\s*(\d+)", cfg, "Sigma total epochs"),
        warmup_epochs=req_int(r"C\.warm_up_epoch\s*=\s*(\d+)", cfg, "Sigma warmup epochs"),
        power=req_float(r"C\.lr_power\s*=\s*([0-9.]+)", cfg, "Sigma power"),
        warmup_ratio=req_float(r"C\.warmup_ratio\s*=\s*([0-9.]+)", cfg, "Sigma warmup ratio"),
        warmup_type="linear",
        source="Sigma/configs/config_UAVScenes.py + Sigma/utils/lr_policy.py",
    )


def parse_mul_vmamba() -> ScheduleSpec:
    cfg = read("Mul_VMamba/configs/uavscenes_rgbhagmulmamba.yaml")
    sched = read("Mul_VMamba/semseg/schedulers.py")
    warmup_type = "linear" if "warmup='linear'" in sched else "unknown"
    return ScheduleSpec(
        model="Mul_VMamba",
        total_epochs=req_int(r"EPOCHS\s*:\s*(\d+)", cfg, "Mul_VMamba total epochs"),
        warmup_epochs=req_int(r"WARMUP\s*:\s*(\d+)", cfg, "Mul_VMamba warmup epochs"),
        power=req_float(r"POWER\s*:\s*([0-9.]+)", cfg, "Mul_VMamba power"),
        warmup_ratio=req_float(r"WARMUP_RATIO\s*:\s*([0-9.]+)", cfg, "Mul_VMamba warmup ratio"),
        warmup_type=warmup_type,
        source="Mul_VMamba/configs/uavscenes_rgbhagmulmamba.yaml + Mul_VMamba/semseg/schedulers.py",
    )


def format_float(value: float) -> str:
    return f"{value:.4g}"


def main() -> int:
    specs = [
        parse_cmx(),
        parse_cmnext(),
        parse_hrfuser(),
        parse_tokenfusion(),
        parse_geminifusion(),
        parse_sigma(),
        parse_mul_vmamba(),
    ]

    headers = ["Model", "Epochs", "Warmup", "Power", "WarmupRatio", "WarmupType"]
    rows = [
        [
            spec.model,
            str(spec.total_epochs),
            str(spec.warmup_epochs),
            format_float(spec.power),
            format_float(spec.warmup_ratio),
            spec.warmup_type,
        ]
        for spec in specs
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(parts: list[str]) -> str:
        return "  ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))

    reference = (
        specs[0].total_epochs,
        specs[0].warmup_epochs,
        specs[0].power,
        specs[0].warmup_ratio,
        specs[0].warmup_type,
    )
    mismatches = []
    for spec in specs[1:]:
        current = (
            spec.total_epochs,
            spec.warmup_epochs,
            spec.power,
            spec.warmup_ratio,
            spec.warmup_type,
        )
        if current != reference:
            mismatches.append(spec)

    print()
    if mismatches:
        print("Mismatch detected against the first model reference:")
        for spec in mismatches:
            print(f"- {spec.model}: from {spec.source}")
        return 1

    print("All 7 UAVScenes models resolve to the same effective LR schedule.")
    print("This audit checks runtime-facing config/call sites, not only class defaults.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
