#!/usr/bin/env python3
"""Validate and summarize fixed-init repeats across task-0 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN_ID = "teacher_native_task0_base_init_current_recipe_10k_20261002_v1"
CHECKPOINT_STEPS = (2500, 5000, 7500)
EXPECTED_REPEATS = 5


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def load_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}; got {len(rows)}")
    row = rows[0]
    for key in ("task_id", "init_index", "seed", "wait_steps", "n_action_steps", "steps"):
        row[key] = int(row[key])
    row["success"] = row["success"].lower() == "true"
    row["total_reward"] = float(row["total_reward"])
    if row["task_id"] != 0 or row["init_source"] != "benchmark":
        raise ValueError(f"Unexpected task or init source in {path}: {row}")
    if row["init_index"] != 3 or row["seed"] != 12348:
        raise ValueError(f"Fixed initialization/seed mismatch in {path}: {row}")
    if row["wait_steps"] != 10 or row["n_action_steps"] != 25:
        raise ValueError(f"Protocol mismatch in {path}: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--current-final-summary", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_step: dict[int, list[dict]] = {step: [] for step in CHECKPOINT_STEPS}
    all_csv = sorted(args.eval_root.rglob("*_summary.csv"))
    for path in all_csv:
        match = re.search(r"checkpoint_(\d{4,6})", str(path))
        if not match:
            raise ValueError(f"Cannot determine checkpoint step from {path}")
        step = int(match.group(1))
        if step not in rows_by_step:
            raise ValueError(f"Unexpected checkpoint step {step} in {path}")
        row = load_row(path)
        if Path(row["checkpoint"]).as_posix().split("/checkpoints/")[-1].split("/")[0] != f"{step:06d}":
            raise ValueError(f"CSV checkpoint path does not match parent step in {path}")
        rows_by_step[step].append(row)

    for step, rows in rows_by_step.items():
        if len(rows) != EXPECTED_REPEATS:
            raise ValueError(f"Expected 5 repeats at {step}; found {len(rows)}")

    current = json.loads(args.current_final_summary.read_text(encoding="utf-8"))
    final_fixed = current["fixed_initialization_repeats"]
    if final_fixed["n"] != EXPECTED_REPEATS:
        raise ValueError("The existing 10k reference must contain exactly 5 fixed repeats")
    log = args.training_log.read_text(encoding="utf-8", errors="replace")
    if "TEACHER_BASE_INIT_TRAIN_EXIT_CODE=0" not in log:
        raise ValueError("Training success marker missing")

    checkpoints = {}
    for step, rows in rows_by_step.items():
        successes = sum(row["success"] for row in rows)
        checkpoints[str(step)] = {
            "n": len(rows),
            "successes": successes,
            "success_seeds": [row["seed"] for row in rows if row["success"]],
            "rewards": [row["total_reward"] for row in rows],
            "steps": [row["steps"] for row in rows],
            "wilson_95_ci": wilson(successes, len(rows)),
        }

    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "audit_type": "diagnostic_checkpoint_progression_fixed_initialization",
        "evaluation_status": "complete_validated_15_of_15",
        "protocol": {
            "task_id": 0,
            "init_source": "benchmark",
            "init_index": 3,
            "cli_seed": 12345,
            "effective_seed": 12348,
            "wait_steps": 10,
            "n_action_steps": 25,
            "max_steps": 300,
            "repeats_per_checkpoint": EXPECTED_REPEATS,
        },
        "training": {"status": "completed", "exit_code": 0, "success_marker_verified": True},
        "intermediate_checkpoints": checkpoints,
        "existing_final_checkpoint_reference": {
            "step": 10000,
            "n": final_fixed["n"],
            "successes": final_fixed["successes"],
            "rewards": final_fixed["rewards"],
            "steps": final_fixed["steps"],
            "source_summary": str(args.current_final_summary),
        },
        "decision": {
            "single_task_gate_changed": False,
            "task_scaling_authorized": False,
            "fold02": "LOCKED",
            "interpretation": "Fixed-init checkpoint sweep is diagnostic only; it cannot replace the paired 20-init gate or authorize task-count scaling.",
        },
        "sources": {
            "eval_root": str(args.eval_root),
            "current_final_summary": str(args.current_final_summary),
            "training_log": str(args.training_log),
        },
        "limitations": [
            "Only one benchmark initialization was tested; results are checkpoint-selection diagnostics, not a generalization estimate.",
            "A checkpoint with any success in five repeats would be a candidate for a paired-init follow-up, not a pass by itself.",
            "This evaluation does not alter the registered benchmark protocol or Fold 02 gate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "checkpoints": {step: value["successes"] for step, value in checkpoints.items()},
        "final_10000_reference": final_fixed["successes"],
        "fold02": "LOCKED",
    }, indent=2))


if __name__ == "__main__":
    main()
