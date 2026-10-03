#!/usr/bin/env python3
"""Read-only sparse teacher-forced action-fit audit for custom task 0.

Uses recorded HDF5 observations and expert action labels from the five training
episodes. It measures one-step action prediction fit at sampled frames; it is
not a rollout, training run, or formal benchmark evaluation.
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
    args = parser.parse_args()

    if args.output_json.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_json}")
    if args.stride < 1:
        raise ValueError("stride must be >= 1")
    checkpoint_file = args.checkpoint / "model.safetensors"
    if not checkpoint_file.is_file() or not args.hdf5.is_file():
        raise FileNotFoundError("checkpoint model.safetensors and HDF5 are required")
    hdf5_hash = sha256(args.hdf5)
    if hdf5_hash != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 sha256: {hdf5_hash}")

    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint))
    policy.to("cuda")
    policy.eval()
    policy.config.n_action_steps = 1
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    demos: list[dict[str, object]] = []
    all_abs_errors: list[np.ndarray] = []
    try:
        with h5py.File(args.hdf5, "r") as source:
            demo_names = sorted(source["data"].keys())
            if len(demo_names) != 5:
                raise AssertionError(f"expected 5 demos, got {len(demo_names)}")
            for demo_name in demo_names:
                demo = source[f"data/{demo_name}"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                observations = demo["obs"]
                count = len(actions)
                lengths = [len(observations[key]) for key in (
                    "agentview_rgb", "eye_in_hand_rgb", "joint_states", "ee_pos",
                    "ee_ori", "gripper_states",
                )]
                if any(length != count for length in lengths):
                    raise AssertionError(f"{demo_name}: observation/action length mismatch {lengths}")
                indices = list(range(0, count, args.stride))
                if indices[-1] != count - 1:
                    indices.append(count - 1)
                errors: list[np.ndarray] = []
                for index in indices:
                    state = np.concatenate((
                        np.asarray(observations["joint_states"][index], dtype=np.float32),
                        np.asarray(observations["ee_pos"][index], dtype=np.float32),
                        np.asarray(observations["ee_ori"][index], dtype=np.float32),
                        np.asarray(observations["gripper_states"][index], dtype=np.float32),
                    ))
                    if state.shape != (15,) or not np.isfinite(state).all():
                        raise AssertionError(f"{demo_name}/{index}: invalid 15-D state")
                    frame = {
                        "observation.images.agentview": image_to_tensor(
                            np.asarray(observations["agentview_rgb"][index], dtype=np.uint8)
                        ),
                        "observation.images.wrist": image_to_tensor(
                            np.asarray(observations["eye_in_hand_rgb"][index], dtype=np.uint8)
                        ),
                        "observation.state": torch.from_numpy(state),
                        "task": args.instruction,
                    }
                    policy.reset()
                    with torch.inference_mode():
                        predicted = post(policy.select_action(pre(frame)))
                    if isinstance(predicted, torch.Tensor):
                        predicted = predicted.detach().cpu().numpy()
                    predicted = np.asarray(predicted, dtype=np.float32).squeeze()
                    if predicted.shape != (7,) or not np.isfinite(predicted).all():
                        raise AssertionError(f"{demo_name}/{index}: invalid predicted action")
                    target = actions[index]
                    if target.shape != (7,) or not np.isfinite(target).all():
                        raise AssertionError(f"{demo_name}/{index}: invalid target action")
                    errors.append(np.abs(predicted - target))
                error_array = np.stack(errors)
                all_abs_errors.append(error_array)
                demos.append({
                    "demo": demo_name,
                    "source_episode_success": bool(demo.attrs.get("success", False)),
                    "source_formal_action_replay_exact": bool(
                        demo.attrs.get("formal_action_replay_exact", False)
                    ),
                    "frames": count,
                    "sampled_frames": len(indices),
                    "stride": args.stride,
                    "sampled_action_mae_by_channel": error_array.mean(axis=0).tolist(),
                    "sampled_action_mae_overall": float(error_array.mean()),
                })

        pooled = np.concatenate(all_abs_errors, axis=0)
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
                "sampled_observation_count": int(pooled.shape[0]),
                "prediction_mode": "reset policy state at each recorded observation and compare the first predicted action after fixed inverse-normalization to the aligned expert action",
            },
            "absolute_action_error_mae_by_channel": pooled.mean(axis=0).tolist(),
            "absolute_action_error_mae_overall": float(pooled.mean()),
            "absolute_action_error_p95_by_channel": np.quantile(pooled, 0.95, axis=0).tolist(),
            "demos": demos,
            "interpretation_limit": "Sparse teacher-forced action fit is not a closed-loop success metric and does not establish training causality.",
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
