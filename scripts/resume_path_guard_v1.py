"""Read-only path preflight, not an atomic launch reservation or GPU lock."""
import os
from pathlib import Path


def require_independent_absent_paths(*, checkpoint, output, log, pid, exit_code):
    paths = [Path(p) for p in (checkpoint, output, log, pid, exit_code)]
    if any(not p.is_absolute() for p in paths):
        raise ValueError("absolute registered paths required")
    checkpoint, *new = paths
    if not checkpoint.is_dir():
        raise ValueError("checkpoint directory missing")
    checkpoint = checkpoint.resolve(strict=True)
    normalized = [p.resolve(strict=False) for p in new]
    if len(set(normalized)) != len(normalized):
        raise ValueError("new paths alias each other")
    for path, resolved in zip(new, normalized):
        if os.path.lexists(path):
            raise FileExistsError(f"preserve existing path: {path}")
        if resolved == checkpoint or checkpoint in resolved.parents or resolved in checkpoint.parents:
            raise ValueError("new path overlaps checkpoint evidence")
        # Avoid redirection through any existing symlink parent.
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError("symlink parent forbidden")
    output = normalized[0]
    if any(output == p or output in p.parents for p in normalized[1:]):
        raise ValueError("log/PID/exit marker must remain outside training output")
    return tuple(normalized)
