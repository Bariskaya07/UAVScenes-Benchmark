from __future__ import annotations

import os
from pathlib import Path


def _unique_existing_paths(paths):
    seen = set()
    resolved_paths = []
    for path in paths:
        candidate = Path(path).expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            resolved_paths.append(candidate)
    return resolved_paths


def get_pretrained_roots(project_root: str | os.PathLike | None = None):
    roots = []

    env_pretrained_root = os.environ.get("UAVSCENES_PRETRAINED_ROOT")
    if env_pretrained_root:
        roots.append(Path(env_pretrained_root).expanduser())

    env_shared_root = os.environ.get("UAVSCENES_SHARED_ROOT")
    if env_shared_root:
        shared_root = Path(env_shared_root).expanduser()
        roots.extend([
            shared_root / "pretrained",
            shared_root / "uavscenes007" / "pretrained",
        ])

    if project_root:
        project_root = Path(project_root).resolve()
        benchmark_root = project_root.parent
        roots.extend([
            project_root / "pretrained",
            project_root / "checkpoints" / "pretrained",
            benchmark_root / "pretrained",
            benchmark_root / "checkpoints" / "pretrained",
        ])

    roots.extend([
        Path.home() / "thesis-uavscenes" / "uavscenes007" / "pretrained",
        Path.home() / "uavscenes007" / "pretrained",
    ])

    return _unique_existing_paths(roots)


def resolve_pretrained_path(path_value: str | os.PathLike | None, project_root: str | os.PathLike | None = None) -> str:
    """Resolve a pretrained weight path against repo-local and shared VM layouts.

    Resolution order:
    1. Absolute path as-is
    2. Relative path inside the current project
    3. Same relative path under shared pretrained roots
    4. Basename match under shared pretrained roots
    5. Recursive basename search under shared pretrained roots
    6. Fallback to project-relative path string
    """
    if not path_value:
        return ""

    raw_path = Path(os.path.expanduser(str(path_value)))
    if raw_path.is_absolute():
        return str(raw_path)

    project_root_path = Path(project_root).resolve() if project_root else None
    if project_root_path is not None:
        project_candidate = (project_root_path / raw_path).resolve()
        if project_candidate.exists():
            return str(project_candidate)
    else:
        project_candidate = raw_path.resolve()
        if project_candidate.exists():
            return str(project_candidate)

    basename = raw_path.name
    for root in get_pretrained_roots(project_root_path):
        for candidate in (root / raw_path, root / basename):
            if candidate.exists():
                return str(candidate.resolve())

        matches = sorted(root.rglob(basename))
        if matches:
            return str(matches[0].resolve())

    if project_root_path is not None:
        return str((project_root_path / raw_path).resolve())
    return str(raw_path)
