#!/usr/bin/env python3
"""Diagnostic-only paired rollout on the five custom task-0 demo starts.

This deliberately does not modify or replace eval_libero_strict_v1.py. It uses
the same strict inference path on the same custom BDDL/instruction, but starts
from the five HDF5 demonstration initial states rather than held-out states.
Results are in-sample diagnostics and are not benchmark evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from eval_v3_task0 import action_to_numpy, observation_to_frame
from libero_36_camera import apply_frozen_agentview_camera


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--stagee1-checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--expected-bddl-sha256", required=True)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--max-demos", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_dir}")
    for path in (args.hdf5, args.bddl):
        if not path.is_file():
            raise FileNotFoundError(path)
    for checkpoint in (args.historical_checkpoint, args.stagee1_checkpoint):
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
    hdf5_sha, bddl_sha = sha256(args.hdf5), sha256(args.bddl)
    if hdf5_sha != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 sha256: {hdf5_sha}")
    if bddl_sha != args.expected_bddl_sha256:
        raise AssertionError(f"unexpected BDDL sha256: {bddl_sha}")

    args.output_dir.mkdir(parents=True)
    outcomes: list[dict[str, object]] = []
    models = (
        ("recovered_historical_v3", args.historical_checkpoint),
        ("stageE1_custom_task0_10k", args.stagee1_checkpoint),
    )
    try:
        for model_name, checkpoint in models:
            policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
            policy.to("cuda")
            policy.eval()
            policy.config.n_action_steps = args.n_action_steps
            pre, post = make_pre_post_processors(
                policy.config,
                pretrained_path=str(checkpoint),
                preprocessor_overrides={"device_processor": {"device": "cuda"}},
            )
            env = OffScreenRenderEnv(
                bddl_file_name=str(args.bddl),
                camera_heights=128,
                camera_widths=128,
                horizon=1000,
            )
            try:
                with h5py.File(args.hdf5, "r") as source:
                    demo_names = sorted(source["data"].keys())
                    if len(demo_names) != 5:
                        raise AssertionError(f"expected five demos, got {len(demo_names)}")
                    for demo_index, demo_name in enumerate(demo_names[: args.max_demos]):
                        state = np.asarray(
                            source[f"data/{demo_name}/states"][0], dtype=np.float64
                        )
                        policy_seed = 2000 + demo_index
                        np.random.seed(1000 + demo_index)
                        random.seed(policy_seed)
                        torch.manual_seed(policy_seed)
                        torch.cuda.manual_seed_all(policy_seed)

                        env.reset()
                        env.set_init_state(state)
                        obs, _ = apply_frozen_agentview_camera(env)
                        policy.reset()
                        for _ in range(args.wait_steps):
                            obs, _, done, _ = env.step(
                                np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                            )
                            if done:
                                break

                        success = False
                        total_reward = 0.0
                        steps_run = 0
                        for step in range(args.steps):
                            with torch.inference_mode():
                                action = action_to_numpy(
                                    post(policy.select_action(pre(observation_to_frame(obs, args.instruction))))
                                )
                            action = np.asarray(action, dtype=np.float32).squeeze()
                            if action.shape != (7,) or not np.isfinite(action).all():
                                raise AssertionError(f"invalid action at {model_name}/{demo_name}/{step}")
                            obs, reward, done, info = env.step(np.clip(action, -1, 1))
                            total_reward += float(reward)
                            steps_run = step + 1
                            success = bool(getattr(env, "check_success", lambda: False)()) or total_reward > 0 or bool(info.get("success", False))
                            if success or done:
                                break

                        row = {
                            "model": model_name,
                            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                            "demo_index": demo_index,
                            "source_episode": demo_name,
                            "policy_seed": policy_seed,
                            "success": success,
                            "total_reward": total_reward,
                            "steps": steps_run,
                            "max_steps": args.steps,
                            "wait_steps": args.wait_steps,
                            "n_action_steps": args.n_action_steps,
                            "task_id": 0,
                            "layout_id": 1,
                            "bddl_sha256": bddl_sha,
                            "hdf5_sha256": hdf5_sha,
                        }
                        outcomes.append(row)
                        model_dir = args.output_dir / model_name
                        model_dir.mkdir(exist_ok=True)
                        (model_dir / f"demo_{demo_index:02d}.json").write_text(
                            json.dumps(row, indent=2) + "\n"
                        )
            finally:
                env.close()
                del policy, pre, post
                torch.cuda.empty_cache()

        summary = {
            "schema_version": 1,
            "diagnostic_only": True,
            "formal_benchmark_evidence": False,
            "training_demo_initial_states_used": True,
            "task_id": 0,
            "layout_id": 1,
            "instruction": args.instruction,
            "hdf5_sha256": hdf5_sha,
            "bddl_sha256": bddl_sha,
            "protocol": {
                "strict_deterministic_algorithms": True,
                "steps": args.steps,
                "wait_steps": args.wait_steps,
                "n_action_steps": args.n_action_steps,
                "max_demos_per_model": args.max_demos,
                "policy_seed_rule": "2000 + demo_index",
            },
            "models": {
                name: {
                    "rollouts": [x for x in outcomes if x["model"] == name],
                    "successes": sum(bool(x["success"]) for x in outcomes if x["model"] == name),
                    "mean_reward": float(np.mean([x["total_reward"] for x in outcomes if x["model"] == name])),
                }
                for name, _ in models
            },
            "interpretation_limit": "These are training-start diagnostics, not held-out performance or a formal benchmark result.",
            "official_benchmark_changed": False,
            "fold02": "LOCKED",
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        if not (args.output_dir / "summary.json").exists():
            (args.output_dir / "partial_outcomes.json").write_text(
                json.dumps(outcomes, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
