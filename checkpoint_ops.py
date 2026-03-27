from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


DEFAULT_BUCKET_URI = "gs://thesis-uavscenes/uavscenes007/checkpoints_30"


def checkpoint_extension(path: os.PathLike[str] | str) -> str:
    suffixes = Path(path).suffixes
    return "".join(suffixes) if suffixes else ".pth"


def epoch_checkpoint_name(model_slug: str, epoch: int, extension: str = ".pth") -> str:
    return f"{model_slug}_epoch_{epoch}{extension}"


def best_checkpoint_name(model_slug: str, epoch: int, extension: str = ".pth") -> str:
    return f"{model_slug}_epoch_{epoch}_best{extension}"


def materialize_epoch_checkpoint(
    source_path: os.PathLike[str] | str,
    model_slug: str,
    epoch: int,
) -> Path:
    src = Path(source_path)
    dst = src.parent / epoch_checkpoint_name(model_slug, epoch, checkpoint_extension(src))
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def promote_best_checkpoint(
    source_path: os.PathLike[str] | str,
    model_slug: str,
    epoch: int,
) -> Path:
    src = Path(source_path)
    ext = checkpoint_extension(src)
    dst = src.parent / best_checkpoint_name(model_slug, epoch, ext)
    for old_best in src.parent.glob(f"{model_slug}_epoch_*_best{ext}"):
        if old_best.exists() and old_best != dst:
            old_best.unlink()
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def maybe_sync_checkpoint_dir(
    checkpoint_dir: os.PathLike[str] | str,
    bucket_uri: str = DEFAULT_BUCKET_URI,
    logger: Optional[Callable[[str], None]] = None,
) -> bool:
    def _log(message: str) -> None:
        if logger is not None:
            logger(message)
        else:
            print(message)

    if str(os.getenv("UAVSCENES_SKIP_CKPT_BUCKET_SYNC", "")).lower() in {"1", "true", "yes"}:
        _log("[CheckpointSync] Skipping bucket sync because UAVSCENES_SKIP_CKPT_BUCKET_SYNC is set.")
        return False

    local_dir = Path(checkpoint_dir)
    if not local_dir.exists():
        _log(f"[CheckpointSync] Checkpoint directory not found, skipping sync: {local_dir}")
        return False

    gsutil = shutil.which("gsutil")
    gcloud = shutil.which("gcloud")

    if gsutil:
        cmd = [gsutil, "-m", "cp", "-r", str(local_dir), bucket_uri]
    elif gcloud:
        cmd = [gcloud, "storage", "cp", "--recursive", str(local_dir), bucket_uri]
    else:
        _log("[CheckpointSync] Neither gsutil nor gcloud was found; skipping bucket sync.")
        return False

    _log(f"[CheckpointSync] Syncing {local_dir} -> {bucket_uri}")
    try:
        subprocess.run(cmd, check=True)
        _log("[CheckpointSync] Bucket sync completed.")
        return True
    except Exception as exc:  # pragma: no cover - best effort sync
        _log(f"[CheckpointSync] Bucket sync failed: {exc}")
        return False
