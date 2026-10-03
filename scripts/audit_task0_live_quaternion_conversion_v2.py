#!/usr/bin/env python3
"""Replay Task0 experts with the original reset/settle protocol and compare orientation converters.

Diagnostic only: no policy inference, optimizer, dataset, or checkpoint writes.
The replay starts from the full simulator state saved with each HDF5 episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from robosuite.utils import transform_utils

from eval_v3_task0 import quat_xyzw_to_axis_angle
from scripts import validate_libero_36_envs as reset_validator


OBS_KEYS = {
    "joint_states": "robot0_joint_pos",
    "ee_pos": "robot0_eef_pos",
    "gripper_states": "robot0_gripper_qpos",
}
COLLECTION_SETTLE_STEPS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diff_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "max_abs_diff": float(array.max()) if array.size else 0.0,
        "mean_abs_diff": float(array.mean()) if array.size else 0.0,
        "p99_abs_diff": float(np.quantile(array, 0.99)) if array.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--layout-spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames-per-episode", type=int)
    parser.add_argument("--state-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_json}")
    if args.state_tolerance <= 0:
        raise ValueError("state-tolerance must be positive")
    for value, name in (
        (args.max_episodes, "max-episodes"),
        (args.max_frames_per_episode, "max-frames-per-episode"),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")

    hdf5_hash = sha256(args.hdf5)
    if hdf5_hash != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 SHA256: {hdf5_hash}")
    rows = reset_validator.read_layout_spec(args.layout_spec)
    matches = [
        row
        for row in rows
        if int(row["task_id"]) == 0 and int(row["layout_id"]) == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task0/layout1 row, got {len(matches)}")
    bddl_path = Path(matches[0]["bddl_path"])
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)

    with h5py.File(args.hdf5, "r") as source:
        demo_names = sorted(source["data"].keys())
        if args.max_episodes is not None:
            demo_names = demo_names[: args.max_episodes]
        env = reset_validator.make_environment(bddl_path)
        episode_rows: list[dict[str, Any]] = []
        try:
            for demo_name in demo_names:
                demo = source[f"data/{demo_name}"]
                if not bool(demo.attrs.get("passed", False)):
                    raise AssertionError(f"source demo did not pass QC: {demo_name}")
                saved_init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
                seed = int(demo.attrs["seed"])
                actions = np.asarray(demo["actions"], dtype=np.float32)
                frame_count = len(actions)
                if args.max_frames_per_episode is not None:
                    frame_count = min(frame_count, args.max_frames_per_episode)

                # Reproduce the collector protocol exactly. Restoring the saved
                # MuJoCo state alone may omit controller-internal state.
                obs, reset_attempts, _camera_record = reset_validator.safe_reset(env, seed, 100)
                for settle_step in range(COLLECTION_SETTLE_STEPS):
                    obs, _reward, done, _info = env.step(np.zeros(7, dtype=np.float32))
                    if done:
                        raise RuntimeError(
                            f"environment terminated during settle at {demo_name}/{settle_step}"
                        )
                replayed_init_state = np.asarray(env.get_sim_state(), dtype=np.float64)
                init_state_max_abs_diff = float(
                    np.max(np.abs(saved_init_state - replayed_init_state))
                )

                stored_state_diffs: dict[str, list[float]] = {
                    key: [] for key in (*OBS_KEYS, "ee_ori")
                }
                converter_diffs: list[float] = []
                quaternion_norm_errors: list[float] = []
                reward_diffs: list[float] = []
                environment_done_true_steps: list[int] = []
                for frame in range(frame_count):
                    for stored_key, live_key in OBS_KEYS.items():
                        stored = np.asarray(demo[f"obs/{stored_key}"][frame], dtype=np.float64).reshape(-1)
                        live = np.asarray(obs[live_key], dtype=np.float64).reshape(-1)
                        if stored.shape != live.shape:
                            raise AssertionError(
                                f"{demo_name}/{frame}/{stored_key}: shape mismatch {stored.shape} vs {live.shape}"
                            )
                        stored_state_diffs[stored_key].append(
                            float(np.max(np.abs(stored - live)))
                        )

                    quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
                    quaternion_norm_errors.append(abs(float(np.linalg.norm(quaternion)) - 1.0))
                    train_axis_angle = np.asarray(
                        transform_utils.quat2axisangle(quaternion.copy()), dtype=np.float64
                    )
                    eval_axis_angle = np.asarray(
                        quat_xyzw_to_axis_angle(quaternion.copy()), dtype=np.float64
                    )
                    stored_orientation = np.asarray(
                        demo["obs/ee_ori"][frame], dtype=np.float64
                    ).reshape(-1)
                    stored_state_diffs["ee_ori"].append(
                        float(np.max(np.abs(stored_orientation - train_axis_angle)))
                    )
                    converter_diffs.append(
                        float(np.max(np.abs(train_axis_angle - eval_axis_angle)))
                    )

                    # The collector continues the entire recorded action trace,
                    # including terminal hold frames, even after env.step reports done.
                    obs, replay_reward, done, _info = env.step(actions[frame])
                    stored_reward = float(demo["rewards"][frame])
                    reward_diffs.append(abs(float(replay_reward) - stored_reward))
                    if done:
                        environment_done_true_steps.append(frame)

                episode_summary = {
                    "demo": demo_name,
                    "seed": seed,
                    "recorded_frames": len(actions),
                    "replayed_frames": frame_count,
                    "reset_attempts": reset_attempts,
                    "source_reset_attempts": int(demo.attrs["reset_attempts"]),
                    "collection_settle_steps": COLLECTION_SETTLE_STEPS,
                    "saved_vs_replayed_initial_sim_state_max_abs_diff": init_state_max_abs_diff,
                    "source_demo_passed_qc": True,
                    "environment_done_true_action_steps": environment_done_true_steps,
                    "replay_vs_stored_reward_abs_diff": diff_stats(reward_diffs),
                    "live_vs_stored_observation_max_abs_diff_by_channel": {
                        key: diff_stats(values)
                        for key, values in stored_state_diffs.items()
                    },
                    "training_vs_evaluation_axisangle_max_abs_diff": diff_stats(converter_diffs),
                    "live_quaternion_norm_error": diff_stats(quaternion_norm_errors),
                }
                episode_summary["state_replay_within_tolerance"] = all(
                    stats["max_abs_diff"] <= args.state_tolerance
                    for stats in episode_summary[
                        "live_vs_stored_observation_max_abs_diff_by_channel"
                    ].values()
                ) and init_state_max_abs_diff <= args.state_tolerance
                episode_rows.append(episode_summary)
        finally:
            env.close()

    all_replays_match = bool(episode_rows) and all(
        row["state_replay_within_tolerance"] for row in episode_rows
    )
    all_frames = sum(row["replayed_frames"] for row in episode_rows)
    all_converter_diffs = [
        row["training_vs_evaluation_axisangle_max_abs_diff"]["max_abs_diff"]
        for row in episode_rows
    ]
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "formal_benchmark_evidence": False,
        "policy_inference_performed": False,
        "training_or_checkpoint_modified": False,
        "dataset_modified": False,
        "hdf5_sha256": hdf5_hash,
        "bddl_path": str(bddl_path),
        "bddl_sha256": sha256(bddl_path),
        "episodes": len(episode_rows),
        "total_replayed_observations": all_frames,
        "initialization_protocol": "same recorded seed + collector safe_reset/frozen-camera setup + 20 zero-action settle steps",
        "action_replay_protocol": "all rows are observed before their corresponding stored action; the action is then stepped, and env done flags are recorded but do not truncate the trace, matching the source collector",
        "state_tolerance": args.state_tolerance,
        "all_action_replays_match_stored_observations": all_replays_match,
        "training_vs_evaluation_axisangle_max_abs_diff_over_episodes": max(
            all_converter_diffs, default=0.0
        ),
        "episodes_detail": episode_rows,
        "interpretation_rule": "Only interpret converter differences as train/eval representation evidence if the replayed joint, position, gripper, and stored training-side axis-angle states all match within tolerance. Any replay mismatch invalidates the converter comparison for that episode.",
        "fold02": "LOCKED",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, indent=2))


if __name__ == "__main__":
    main()
