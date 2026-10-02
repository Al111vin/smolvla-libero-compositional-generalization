#!/usr/bin/env python3
"""Read-only descriptive audit of the 20 paired task-0 rollout action traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


RUN_ID = "teacher_native_task0_base_init_current_recipe_10k_20261002_v1"
EXPECTED_INDICES = set(range(20))
FIRST_WINDOW = 60
GRIPPER_DEADBAND = 0.05
CLIP_EPSILON = 1e-3


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def cliffs_delta(success: list[float], failure: list[float]) -> float:
    if not success or not failure:
        return 0.0
    wins = sum(a > b for a in success for b in failure)
    losses = sum(a < b for a in success for b in failure)
    return (wins - losses) / (len(success) * len(failure))


def read_action_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty action trace: {path}")
    required = {
        *(f"raw_action_{i}" for i in range(7)),
        *(f"processed_action_{i}" for i in range(7)),
        *(f"applied_action_{i}" for i in range(7)),
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing action columns in {path}: {sorted(missing)}")
    return rows


def action_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    applied = [
        [float(row[f"applied_action_{i}"]) for i in range(7)] for row in rows
    ]
    raw = [[float(row[f"raw_action_{i}"]) for i in range(7)] for row in rows]
    gripper = [row[6] for row in applied]
    signs = [
        -1 if value < -GRIPPER_DEADBAND else 1 if value > GRIPPER_DEADBAND else 0
        for value in gripper
    ]
    sign_changes = sum(a * b == -1 for a, b in zip(signs, signs[1:]))
    transitions = max(1, len(rows) - 1)

    def channel_group_delta(start: int, end: int) -> float:
        deltas = [
            math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(start, end)))
            for a, b in zip(applied, applied[1:])
        ]
        return mean(deltas)

    return {
        "steps": float(len(rows)),
        "gripper_sign_changes_per_100_steps": sign_changes / transitions * 100,
        "applied_action_dim_0_2_step_delta_l2_mean": channel_group_delta(0, 3),
        "applied_action_dim_3_5_step_delta_l2_mean": channel_group_delta(3, 6),
        "applied_gripper_mean": mean(gripper),
        "applied_gripper_negative_fraction_lt_minus_0_5": sum(v < -0.5 for v in gripper) / len(gripper),
        "applied_gripper_positive_fraction_gt_0_5": sum(v > 0.5 for v in gripper) / len(gripper),
        "applied_any_dimension_at_clip_boundary_fraction": sum(
            any(abs(value) >= 1.0 - CLIP_EPSILON for value in row) for row in applied
        ) / len(applied),
        "raw_action_outside_minus1_1_element_fraction": sum(
            abs(value) > 1.0 for row in raw for value in row
        ) / (len(raw) * 7),
    }


def group_summary(records: list[dict], key: str, outcome: bool) -> dict:
    values = [r[key] for r in records if r["success"] is outcome]
    return {"n": len(values), "mean": round(mean(values), 6), "median": round(median(values), 6)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    outcomes = {
        int(row["init_index"]): row
        for row in evaluation["paired_20_initializations"]["rows"]
    }
    if set(outcomes) != EXPECTED_INDICES:
        raise ValueError("Evaluation summary must contain paired init indices 0..19")

    by_index: dict[int, Path] = {}
    for path in args.action_root.glob("*_actions.csv"):
        match = re.search(r"_init(\d+)_n25_", path.name)
        if not match:
            raise ValueError(f"Unexpected action filename: {path.name}")
        index = int(match.group(1))
        if index in by_index:
            raise ValueError(f"Duplicate action trace for init index {index}")
        by_index[index] = path
    if set(by_index) != EXPECTED_INDICES:
        raise ValueError(f"Expected action traces 0..19, found {sorted(by_index)}")

    records = []
    for index in sorted(EXPECTED_INDICES):
        outcome = outcomes[index]
        rows = read_action_rows(by_index[index])
        if len(rows) != int(outcome["steps"]):
            raise ValueError(
                f"Step-count mismatch at init {index}: trace={len(rows)}, "
                f"summary={outcome['steps']}"
            )
        record = {
            "init_index": index,
            "effective_seed": int(outcome["effective_seed"]),
            "success": bool(outcome["success"]),
            "steps": len(rows),
            "full_rollout": action_metrics(rows),
            "first_60_steps": action_metrics(rows[:FIRST_WINDOW]),
        }
        if len(rows) < FIRST_WINDOW:
            raise ValueError(f"Rollout {index} is shorter than the shared 60-step window")
        records.append(record)

    metric_keys = [
        "gripper_sign_changes_per_100_steps",
        "applied_action_dim_0_2_step_delta_l2_mean",
        "applied_action_dim_3_5_step_delta_l2_mean",
        "applied_gripper_mean",
        "applied_gripper_negative_fraction_lt_minus_0_5",
        "applied_gripper_positive_fraction_gt_0_5",
        "applied_any_dimension_at_clip_boundary_fraction",
        "raw_action_outside_minus1_1_element_fraction",
    ]
    aggregates = {}
    for window in ("first_60_steps", "full_rollout"):
        aggregates[window] = {}
        for key in metric_keys:
            success = [r[window][key] for r in records if r["success"]]
            failure = [r[window][key] for r in records if not r["success"]]
            aggregates[window][key] = {
                "success": {"n": len(success), "mean": round(mean(success), 6), "median": round(median(success), 6)},
                "failure": {"n": len(failure), "mean": round(mean(failure), 6), "median": round(median(failure), 6)},
                "cliffs_delta_success_minus_failure": round(cliffs_delta(success, failure), 6),
            }

    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "audit_type": "read_only_descriptive_action_trace_audit",
        "validation": {
            "num_paired_action_traces": len(records),
            "num_success": sum(r["success"] for r in records),
            "num_failure": sum(not r["success"] for r in records),
            "all_trace_lengths_match_evaluation_summary": True,
            "indices_exactly_0_to_19": True,
        },
        "method": {
            "first_shared_window_steps": FIRST_WINDOW,
            "gripper_sign_deadband": GRIPPER_DEADBAND,
            "clip_boundary_tolerance": CLIP_EPSILON,
            "applied_action_components_used_for_jitter": "dimensions 0-2 and 3-5 separately",
            "no_hypothesis_test_or_causal_claim": True,
        },
        "group_comparison": aggregates,
        "per_initialization": records,
        "interpretation_limits": [
            "Only 20 single rollouts from one checkpoint were available (9 successes, 11 failures).",
            "Full-rollout statistics are length-confounded because successful episodes terminate early while failures generally run to the cap.",
            "The first 60 steps provide an equal-duration comparison; similar early action smoothness does not prove later oscillation causes failure.",
            "Raw/applied command statistics are descriptive and do not establish that a training or evaluator bug is present.",
        ],
        "sources": {
            "evaluation_summary": str(args.evaluation_summary),
            "action_root": str(args.action_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "num_success": result["validation"]["num_success"],
        "num_failure": result["validation"]["num_failure"],
        "first_60_gripper_flip_rate_success": aggregates["first_60_steps"]["gripper_sign_changes_per_100_steps"]["success"]["mean"],
        "first_60_gripper_flip_rate_failure": aggregates["first_60_steps"]["gripper_sign_changes_per_100_steps"]["failure"]["mean"],
        "full_gripper_flip_rate_success": aggregates["full_rollout"]["gripper_sign_changes_per_100_steps"]["success"]["mean"],
        "full_gripper_flip_rate_failure": aggregates["full_rollout"]["gripper_sign_changes_per_100_steps"]["failure"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
