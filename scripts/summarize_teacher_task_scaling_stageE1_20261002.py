#!/usr/bin/env python3
"""Validate and summarize the bounded Stage E1 task-0 training/evaluation."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output}")

    preflight = json.loads(args.preflight.read_text())
    if not preflight.get("passed") or preflight.get("training_started"):
        raise AssertionError("no-gradient sampler preflight is absent or invalid")
    if preflight["requested_task_ids"] != [0] or preflight["expected_samples_per_task"] != 80000:
        raise AssertionError("preflight does not match the registered Stage E1 design")

    log = args.training_log.read_text(errors="replace")
    marker = re.search(r"TEACHER_STAGE_E1_EXIT_CODE=(\d+)", log)
    exit_code = int(marker.group(1)) if marker else None
    if exit_code != 0:
        raise AssertionError(f"training exit marker is absent or nonzero: {exit_code}")
    if re.search(r"(?:loss|grad_norm)[^\n]*(?:\bnan\b|\binf(?:inity)?\b)", log, re.I):
        raise AssertionError("training log contains non-finite loss/gradient output")

    for step in (2500, 5000, 7500, 10000):
        model = args.training_output / "checkpoints" / f"{step:06d}" / "pretrained_model" / "model.safetensors"
        if not model.is_file():
            raise FileNotFoundError(f"missing checkpoint: {model}")

    # eval_libero_strict_v1 writes one summary at the output root and stores
    # per-state records under checkpoint_<step>/; there is no task_000 layer.
    summary_path = args.evaluation_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint = summary.get("checkpoints", {}).get("010000")
    if summary.get("records") != 50 or not checkpoint or checkpoint.get("num_rollouts") != 50:
        raise AssertionError("strict evaluation does not contain all 50 task0 initial states")
    success_indices = sorted(int(x) for x in checkpoint["success_indices"])
    successes = int(checkpoint["num_success"])
    if len(success_indices) != successes:
        raise AssertionError("success count and per-initialization rows disagree")

    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": "teacher_directed_custom_task0_same_source_single_task_control",
        "training": {
            "status": "completed",
            "exit_code": exit_code,
            "optimizer_updates": 10000,
            "base_initialization": "/root/smolvla-training-prep/models/smolvla_base_libero",
            "dataset_root": preflight["dataset_root"],
            "dataset_repo_id": preflight["dataset_repo_id"],
            "episodes": preflight["episodes_per_requested_task"]["0"],
            "task0_frames": preflight["frames_per_requested_task"]["0"],
            "expected_task0_sample_draws": preflight["expected_samples_per_task"],
            "preflight_no_gradient_passed": True
        },
        "evaluation": {
            "status": "completed_validated_50_of_50",
            "task_id": 0,
            "layout_id": 1,
            "initial_state_manifest_sha256": summary["states_manifest_sha256"],
            "bddl_sha256": summary["bddl_sha256"],
            "protocol": {"strict_determinism": True, "steps": 280, "wait_steps": 10, "n_action_steps": 25},
            "successes": successes,
            "rollouts": 50,
            "success_rate": round(successes / 50, 4),
            "wilson_95_ci": wilson(successes, 50),
            "success_indices": success_indices,
            "mean_reward": checkpoint["mean_reward"],
            "checkpoint_sha256": json.loads((args.evaluation_root / "checkpoint_010000" / "state_000.json").read_text())["checkpoint_sha256"]
        },
        "decision": {
            "predeclared_gate": "at least 28/50 success with no evaluator or deterministic-algorithm errors",
            "stageE1_gate_passed": successes >= 28,
            "stageE2_four_task_run_authorized_by_gate": successes >= 28,
            "official_benchmark_changed": False,
            "fold02": "LOCKED"
        },
        "evidence_paths": {
            "preflight": str(args.preflight),
            "training_log": str(args.training_log),
            "training_output": str(args.training_output),
            "evaluation_root": str(args.evaluation_root)
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
