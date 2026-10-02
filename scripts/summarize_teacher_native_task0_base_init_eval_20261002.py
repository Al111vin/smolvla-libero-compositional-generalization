#!/usr/bin/env python3
"""Validate and summarize the fixed + paired task0 base-init evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RUN_NAME = "teacher_native_task0_base_init_current_recipe_10k_20261002_v1"
EXPECTED_FIXED_REPEATS = 5
EXPECTED_PAIRED_INIT_INDICES = set(range(20))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial test for the two discordant paired counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 4)


def load_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one rollout row in {path}; found {len(rows)}")
    row = rows[0]
    row["init_index"] = int(row["init_index"])
    row["task_id"] = int(row["task_id"])
    row["wait_steps"] = int(row["wait_steps"])
    row["n_action_steps"] = int(row["n_action_steps"])
    row["steps"] = int(row["steps"])
    row["seed"] = int(row["seed"])
    row["success"] = row["success"].strip().lower() == "true"
    row["total_reward"] = float(row["total_reward"])
    if row["task_id"] != 0 or row["init_source"] != "benchmark":
        raise ValueError(f"Unexpected task or init source in {path}: {row}")
    if row["wait_steps"] != 10 or row["n_action_steps"] != 25:
        raise ValueError(f"Protocol mismatch in {path}: {row}")
    return row


def rows_under(directory: Path) -> list[tuple[Path, dict]]:
    files = sorted(directory.rglob("*_summary.csv"))
    return [(p, load_csv(p)) for p in files]


def read_reference(path: Path) -> dict[int, bool]:
    ref = json.loads(path.read_text(encoding="utf-8"))
    rows = ref["evaluation"]["paired_20_initializations"]["rows"]
    result = {int(r["init_index"]): bool(r["success"]) for r in rows}
    if set(result) != EXPECTED_PAIRED_INIT_INDICES:
        raise ValueError("Reference result must contain init indices 0..19 exactly once")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--historical-reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--training-log", type=Path)
    ap.add_argument("--run-id", default=RUN_NAME)
    ap.add_argument("--training-success-marker", default="TEACHER_BASE_INIT_TRAIN_EXIT_CODE=0")
    args = ap.parse_args()

    all_rows = rows_under(args.eval_root)
    fixed_rows = [(p, r) for p, r in all_rows if any(part.startswith("fixed_init3_repeat_") for part in p.parts)]
    paired_rows = [(p, r) for p, r in all_rows if "paired_20_init" in p.parts]
    if len(fixed_rows) != EXPECTED_FIXED_REPEATS:
        raise ValueError(f"Expected 5 fixed-repeat summary CSVs; found {len(fixed_rows)}")
    if len(paired_rows) != 20:
        raise ValueError(f"Expected 20 paired-init summary CSVs; found {len(paired_rows)}")
    for path, row in fixed_rows:
        if row["init_index"] != 3 or row["seed"] != 12348:
            raise ValueError(f"Fixed-repeat condition mismatch in {path}: {row}")

    paired_by_index: dict[int, dict] = {}
    for path, row in paired_rows:
        idx = row["init_index"]
        if idx in paired_by_index:
            raise ValueError(f"Duplicate paired init index {idx}: {path}")
        if row["seed"] != 12345 + 2 * idx:
            raise ValueError(f"Paired seed mismatch in {path}: {row}")
        paired_by_index[idx] = row
    if set(paired_by_index) != EXPECTED_PAIRED_INIT_INDICES:
        raise ValueError("Paired-init indices must be exactly 0..19")

    ref = read_reference(args.historical_reference)
    candidate_successes = {i for i, r in paired_by_index.items() if r["success"]}
    baseline_successes = {i for i, success in ref.items() if success}
    b = len(baseline_successes - candidate_successes)
    c = len(candidate_successes - baseline_successes)
    fixed_successes = sum(row["success"] for _, row in fixed_rows)
    paired_n = len(paired_by_index)
    paired_k = len(candidate_successes)

    training = {"status": "not_included"}
    if args.training_log:
        log = args.training_log.read_text(encoding="utf-8", errors="replace")
        marker = args.training_success_marker
        if marker not in log:
            raise ValueError("Training success marker missing from the supplied training log")
        training = {"status": "completed", "exit_code": 0, "success_marker_verified": True}

    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "evaluation_status": "complete_validated_25_of_25",
        "protocol": {
            "task_id": 0,
            "wait_steps": 10,
            "n_action_steps": 25,
            "max_steps": 300,
            "fixed_init_index": 3,
            "fixed_effective_seed": 12348,
            "paired_seed_rule_effective": "12345 + 2 * init_index",
        },
        "training": training,
        "fixed_initialization_repeats": {
            "n": len(fixed_rows),
            "successes": fixed_successes,
            "rewards": [r["total_reward"] for _, r in fixed_rows],
            "steps": [r["steps"] for _, r in fixed_rows],
        },
        "paired_20_initializations": {
            "n": paired_n,
            "successes": paired_k,
            "mean_reward": round(sum(r["total_reward"] for r in paired_by_index.values()) / paired_n, 4),
            "wilson_95_ci": wilson_interval(paired_k, paired_n),
            "success_indices": sorted(candidate_successes),
            "historical_success_indices": sorted(baseline_successes),
            "paired_outcomes": {
                "historical_success_candidate_failure_indices": sorted(baseline_successes - candidate_successes),
                "historical_failure_candidate_success_indices": sorted(candidate_successes - baseline_successes),
                "both_success_indices": sorted(candidate_successes & baseline_successes),
                "both_failure_indices": sorted(EXPECTED_PAIRED_INIT_INDICES - candidate_successes - baseline_successes),
                "exact_two_sided_mcnemar_p": exact_mcnemar_p(b, c),
            },
            "rows": [
                {
                    "init_index": i,
                    "effective_seed": paired_by_index[i]["seed"],
                    "success": paired_by_index[i]["success"],
                    "reward": paired_by_index[i]["total_reward"],
                    "steps": paired_by_index[i]["steps"],
                }
                for i in sorted(paired_by_index)
            ],
        },
        "decision": {
            "fixed_gate_met": fixed_successes == EXPECTED_FIXED_REPEATS,
            "paired_gate_met": paired_k >= 11,
            "single_task_gate_met": fixed_successes == EXPECTED_FIXED_REPEATS and paired_k >= 11,
            "task_scaling_authorized": False,
            "fold02": "LOCKED",
        },
        "sources": {
            "eval_root": str(args.eval_root),
            "historical_reference": str(args.historical_reference),
            "training_log": str(args.training_log) if args.training_log else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "fixed_successes": fixed_successes,
        "paired_successes": paired_k,
        "paired_wilson_95_ci": result["paired_20_initializations"]["wilson_95_ci"],
        "single_task_gate_met": result["decision"]["single_task_gate_met"],
        "fold02": "LOCKED",
    }, indent=2))


if __name__ == "__main__":
    main()
