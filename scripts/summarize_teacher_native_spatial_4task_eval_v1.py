"""Validate and summarize the registered four-task native Spatial evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--baseline-eval-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv_tree(root: Path):
    rows = []
    for path in sorted(root.rglob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        if len(parsed) != 1:
            raise ValueError(f"Expected one summary row in {path}, found {len(parsed)}")
        row = parsed[0]
        row["_path"] = str(path)
        rows.append(row)
    return rows


def as_bool(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Unrecognized success value: {value!r}")


def wilson(successes: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - radius, 4), round(center + radius, 4)]


def exact_mcnemar_p(a: int, b: int):
    discordant = a + b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a, b) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def validate_protocol(row, *, task_id: int, init_index: int, seed: int, checkpoint: str):
    expected = {
        "suite": "libero_spatial",
        "task_id": str(task_id),
        "init_source": "benchmark",
        "init_index": str(init_index),
        "seed": str(seed),
        "wait_steps": "10",
        "n_action_steps": "25",
        "checkpoint": checkpoint,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Protocol mismatch {key}: {row.get(key)!r} != {value!r} ({row['_path']})")


def clean_row(row):
    steps = int(row["steps"])
    if not 1 <= steps <= 300:
        raise ValueError(f"Rollout step count out of range: {steps} ({row['_path']})")
    return {
        "task_id": int(row["task_id"]),
        "init_index": int(row["init_index"]),
        "seed": int(row["seed"]),
        "success": as_bool(row["success"]),
        "reward": float(row["total_reward"]),
        "steps": steps,
        "csv": row["_path"],
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    candidate_rows = read_csv_tree(args.eval_root)
    checkpoint = str(args.checkpoint_root / "040000" / "pretrained_model")

    paired = {task_id: [] for task_id in range(4)}
    fixed = []
    probes = []
    for row in candidate_rows:
        path = Path(row["_path"])
        record = clean_row(row)
        if "paired_20_init" in path.parts:
            task_id = record["task_id"]
            if record["init_index"] not in range(20):
                raise ValueError(f"Unexpected paired init index: {record}")
            validate_protocol(row, task_id=task_id, init_index=record["init_index"], seed=12345 + 2 * record["init_index"], checkpoint=checkpoint)
            paired[task_id].append(record)
        elif "fixed_task0_init3" in path.parts:
            validate_protocol(row, task_id=0, init_index=3, seed=12351, checkpoint=checkpoint)
            fixed.append(record)
        elif "checkpoint_probe" in path.parts:
            step = int(next(part.split("_")[-1] for part in path.parts if part.startswith("step_")))
            if step not in (10000, 20000, 30000, 40000):
                raise ValueError(f"Unexpected checkpoint probe step: {step}")
            expected_checkpoint = str(args.checkpoint_root / f"{step:06d}" / "pretrained_model")
            validate_protocol(row, task_id=0, init_index=3, seed=12351, checkpoint=expected_checkpoint)
            probes.append({**record, "checkpoint_step": step})
        else:
            raise ValueError(f"Unexpected evaluation output location: {path}")

    for task_id in range(4):
        indices = [row["init_index"] for row in paired[task_id]]
        if len(indices) != 20 or sorted(indices) != list(range(20)):
            raise ValueError(f"Task {task_id}: expected exactly 20 unique paired init rows, got {indices}")
    if len(fixed) != 5 or len(probes) != 4:
        raise ValueError(f"Expected 5 fixed repeats + 4 checkpoint probes, found {len(fixed)} + {len(probes)}")
    if [row["checkpoint_step"] for row in sorted(probes, key=lambda r: r["checkpoint_step"])] != [10000, 20000, 30000, 40000]:
        raise ValueError("Checkpoint-probe coverage is incomplete")

    baseline_rows = read_csv_tree(args.baseline_eval_root)
    baseline_by_init = {}
    for row in baseline_rows:
        if int(row["task_id"]) != 0:
            continue
        init = int(row["init_index"])
        seed = 12345 + 2 * init
        if row["seed"] != str(seed):
            raise ValueError(f"Baseline seed mismatch for init {init}: {row['seed']}")
        baseline_by_init[init] = as_bool(row["success"])
    if set(baseline_by_init) != set(range(20)):
        raise ValueError(f"Task-0 baseline must contain paired init 0..19, got {sorted(baseline_by_init)}")

    per_task = {}
    for task_id, rows in paired.items():
        successes = sum(row["success"] for row in rows)
        per_task[str(task_id)] = {
            "n": 20,
            "successes": successes,
            "success_rate": successes / 20,
            "wilson_95_ci": wilson(successes, 20),
            "mean_reward": sum(row["reward"] for row in rows) / 20,
            "success_init_indices": [row["init_index"] for row in rows if row["success"]],
            "rows": sorted(rows, key=lambda row: row["init_index"]),
        }

    task0 = {row["init_index"]: row["success"] for row in paired[0]}
    a = sum(baseline_by_init[i] and not task0[i] for i in range(20))
    b = sum(not baseline_by_init[i] and task0[i] for i in range(20))
    fixed_successes = sum(row["success"] for row in fixed)
    gate = {
        "task0_at_least_8_of_20": per_task["0"]["successes"] >= 8,
        "task0_fixed_init3_five_of_five": fixed_successes == 5,
        "task1_at_least_10_of_20": per_task["1"]["successes"] >= 10,
        "task2_at_least_10_of_20": per_task["2"]["successes"] >= 10,
        "task3_at_least_10_of_20": per_task["3"]["successes"] >= 10,
    }
    result = {
        "schema_version": 1,
        "status": "complete_validated",
        "run_id": args.run_id,
        "evaluation_protocol": {
            "evaluator": "eval_v3_task0_state_capture_v1.py",
            "suite": "libero_spatial",
            "init_indices": list(range(20)),
            "paired_seed_rule": "12345 + 2*init_index",
            "wait_steps": 10,
            "n_action_steps": 25,
            "max_steps": 300,
            "note": "Same native V3 rollout protocol as the single-task baseline; not a strict-deterministic-algorithms protocol."
        },
        "paired_20_initializations": per_task,
        "task0_baseline_comparison": {
            "baseline_source": str(args.baseline_eval_root),
            "baseline_successes": sum(baseline_by_init.values()),
            "candidate_successes": per_task["0"]["successes"],
            "baseline_only_successes": a,
            "candidate_only_successes": b,
            "exact_two_sided_mcnemar_p": round(exact_mcnemar_p(a, b), 4),
        },
        "task0_fixed_init3_repeats": {
            "n": 5,
            "successes": fixed_successes,
            "rows": fixed,
        },
        "checkpoint_probes_task0_init3": sorted(probes, key=lambda row: row["checkpoint_step"]),
        "progression_gate": gate,
        "progression_gate_passed": all(gate.values()),
        "next_stage": "design_and_preflight_the_8-task stage only" if all(gate.values()) else "do_not_scale task count; compare taskwise single-task controls against joint-4 to identify whether any failure is task-specific or interference",
        "fold02": "LOCKED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "per_task_successes": {k: v["successes"] for k, v in per_task.items()}, "fixed_init3": f"{fixed_successes}/5", "gate_passed": result["progression_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
