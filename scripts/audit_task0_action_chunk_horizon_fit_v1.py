#!/usr/bin/env python3
"""Compare the complete predicted 50-step action chunk with expert actions.

This is an in-sample, read-only inference diagnostic on recorded custom-task-0
training observations. It is not training, a closed-loop rollout, or benchmark
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from eval_v3_task0 import image_to_tensor


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-windows", type=int, default=None)
    args = parser.parse_args()

    if args.output_json.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_json}")
    if args.stride < 1 or args.chunk_size < 1:
        raise ValueError("stride and chunk-size must be >= 1")
    if args.limit_windows is not None and args.limit_windows < 1:
        raise ValueError("limit-windows must be >= 1 when specified")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    checkpoint_file = args.checkpoint / "model.safetensors"
    if not checkpoint_file.is_file() or not args.hdf5.is_file():
        raise FileNotFoundError("checkpoint model.safetensors and HDF5 are required")
    hdf5_hash = sha256(args.hdf5)
    if hdf5_hash != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 sha256: {hdf5_hash}")

    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint))
    policy.to("cuda")
    policy.eval()
    if int(policy.config.chunk_size) != args.chunk_size:
        raise AssertionError(
            f"requested chunk_size={args.chunk_size}, checkpoint config={policy.config.chunk_size}"
        )
    policy.config.n_action_steps = args.chunk_size
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    all_errors: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    demos: list[dict[str, object]] = []
    total_windows = 0
    try:
        with h5py.File(args.hdf5, "r") as source:
            demo_names = sorted(source["data"].keys())
            if len(demo_names) != 5:
                raise AssertionError(f"expected 5 demos, got {len(demo_names)}")
            remaining = args.limit_windows
            for demo_name in demo_names:
                demo = source[f"data/{demo_name}"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                obs = demo["obs"]
                count = len(actions)
                required = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "ee_pos", "ee_ori", "gripper_states")
                lengths = [len(obs[key]) for key in required]
                if any(length != count for length in lengths):
                    raise AssertionError(f"{demo_name}: observation/action length mismatch {lengths}")
                if count < args.chunk_size:
                    raise AssertionError(f"{demo_name}: episode shorter than action chunk")
                starts = list(range(0, count - args.chunk_size + 1, args.stride))
                final_start = count - args.chunk_size
                if starts[-1] != final_start:
                    starts.append(final_start)
                if remaining is not None:
                    starts = starts[:remaining]
                demo_errors: list[np.ndarray] = []
                demo_targets: list[np.ndarray] = []
                for start in starts:
                    state = np.concatenate((
                        np.asarray(obs["joint_states"][start], dtype=np.float32),
                        np.asarray(obs["ee_pos"][start], dtype=np.float32),
                        np.asarray(obs["ee_ori"][start], dtype=np.float32),
                        np.asarray(obs["gripper_states"][start], dtype=np.float32),
                    ))
                    if state.shape != (15,) or not np.isfinite(state).all():
                        raise AssertionError(f"{demo_name}/{start}: invalid 15-D state")
                    frame = {
                        "observation.images.agentview": image_to_tensor(np.asarray(obs["agentview_rgb"][start], dtype=np.uint8)),
                        "observation.images.wrist": image_to_tensor(np.asarray(obs["eye_in_hand_rgb"][start], dtype=np.uint8)),
                        "observation.state": torch.from_numpy(state),
                        "task": args.instruction,
                    }
                    policy.reset()
                    with torch.inference_mode():
                        normalized_chunk = policy.predict_action_chunk(pre(frame))
                        physical_chunk = post(normalized_chunk)
                    if isinstance(physical_chunk, torch.Tensor):
                        physical_chunk = physical_chunk.detach().cpu().numpy()
                    physical_chunk = np.asarray(physical_chunk, dtype=np.float32)
                    if physical_chunk.shape != (1, args.chunk_size, 7):
                        raise AssertionError(
                            f"unexpected postprocessed chunk shape {physical_chunk.shape}"
                        )
                    predicted = physical_chunk[0]
                    target = actions[start : start + args.chunk_size]
                    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
                        raise AssertionError(f"{demo_name}/{start}: non-finite chunk or target")
                    demo_errors.append(np.abs(predicted - target))
                    demo_targets.append(np.abs(target))
                if not starts:
                    continue
                err = np.stack(demo_errors)
                tgt = np.stack(demo_targets)
                all_errors.append(err)
                all_targets.append(tgt)
                total_windows += len(starts)
                demos.append({
                    "demo": demo_name,
                    "windows": len(starts),
                    "source_episode_success": bool(demo.attrs.get("success", False)),
                    "source_formal_action_replay_exact": bool(demo.attrs.get("formal_action_replay_exact", False)),
                    "mae_first_10_steps": float(err[:, :10].mean()),
                    "mae_last_10_steps": float(err[:, -10:].mean()),
                })
                if remaining is not None:
                    remaining -= len(starts)
                    if remaining <= 0:
                        break

        errors = np.concatenate(all_errors, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        horizon_mae = errors.mean(axis=(0, 2))
        horizon_p95 = np.quantile(errors.mean(axis=2), 0.95, axis=0)
        horizon_target_mean_abs = targets.mean(axis=(0, 2))
        summary = {
            "schema_version": 1,
            "diagnostic_only": True,
            "formal_benchmark_evidence": False,
            "training_or_checkpoint_modified": False,
            "task_id": 0,
            "layout_id": 1,
            "instruction": args.instruction,
            "checkpoint_sha256": sha256(checkpoint_file),
            "hdf5_sha256": hdf5_hash,
            "sampling": {
                "episodes": len(demos),
                "stride": args.stride,
                "windows": total_windows,
                "chunk_size": args.chunk_size,
                "inference_seed": args.seed,
                "limit_windows": args.limit_windows,
                "comparison": "postprocessed full predicted chunk against expert actions starting at the same recorded observation; overlapping windows are retained",
            },
            "overall_action_mae": float(errors.mean()),
            "horizon_mae": horizon_mae.tolist(),
            "horizon_p95_mean_channel_mae": horizon_p95.tolist(),
            "target_mean_absolute_by_horizon": horizon_target_mean_abs.tolist(),
            "horizon_mae_over_target_mean_absolute": (horizon_mae / np.maximum(horizon_target_mean_abs, 1e-12)).tolist(),
            "mae_first_10_steps": float(errors[:, :10].mean()),
            "mae_last_10_steps": float(errors[:, -10:].mean()),
            "mae_by_channel": errors.mean(axis=(0, 1)).tolist(),
            "demos": demos,
            "interpretation_limit": "Teacher-forced chunk error is an in-sample diagnostic; it is not a closed-loop success metric and does not establish causality.",
            "fold02": "LOCKED",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        del policy, pre, post
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
