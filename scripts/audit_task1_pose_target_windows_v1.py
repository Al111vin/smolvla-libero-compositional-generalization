#!/usr/bin/env python3
"""Audit proposed closure/lift sampling windows without modifying data.

The script reads edge-pinch v2 HDF5 demonstrations, detects the first stable
close command and the first subsequent stable reopen command, and reports the
proposed [close-pre, close+post] window. It never rewrites the source files or
trains a model.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np


def first_sustained(values: np.ndarray, predicate, length: int) -> int | None:
    mask = predicate(values)
    for i in range(0, len(mask) - length + 1):
        if bool(mask[i : i + length].all()):
            return i
    return None


def inspect_file(path: str, pre: int, post: int, multiplier: int) -> dict:
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["data/demo_0/actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path}: expected actions shape (T,7), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{path}: actions contain NaN/Inf")
    grip = actions[:, 6]
    close = first_sustained(grip, lambda x: x > 0.5, 3)
    if close is None:
        raise ValueError(f"{path}: no stable close event")
    reopen = first_sustained(grip[close + 1 :], lambda x: x < -0.5, 3)
    reopen = None if reopen is None else close + 1 + reopen
    start = max(0, close - pre)
    end = min(len(actions), close + post + 1)
    return {
        "source": str(path),
        "frames": int(len(actions)),
        "close_step": int(close),
        "reopen_step": None if reopen is None else int(reopen),
        "window_start": int(start),
        "window_end_exclusive": int(end),
        "window_frames": int(end - start),
        "sampling_multiplier": int(multiplier),
        "source_sha256_required_for_training": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pre", type=int, default=10)
    parser.add_argument("--post", type=int, default=20)
    parser.add_argument("--multiplier", type=int, default=3)
    args = parser.parse_args()
    if args.pre < 0 or args.post < 0 or args.multiplier < 1:
        raise ValueError("pre/post must be non-negative and multiplier must be >= 1")
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(args.input_glob)
    records = [inspect_file(p, args.pre, args.post, args.multiplier) for p in paths]
    summary = {
        "schema_version": 1,
        "read_only": True,
        "num_demos": len(records),
        "pre": args.pre,
        "post": args.post,
        "sampling_multiplier": args.multiplier,
        "total_source_frames": int(sum(r["frames"] for r in records)),
        "total_window_frames": int(sum(r["window_frames"] for r in records)),
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"POSE_TARGET_WINDOW_AUDIT_OK demos={len(records)} output={out}")


if __name__ == "__main__":
    main()
