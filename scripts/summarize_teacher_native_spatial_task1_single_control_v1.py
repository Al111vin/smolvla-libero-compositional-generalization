#!/usr/bin/env python3
"""Validate and summarize the registered task-1-only control evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def rows_under(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.rglob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        if len(parsed) != 1:
            raise ValueError(f"Expected one CSV row in {path}, found {len(parsed)}")
        row = parsed[0]
        row["_path"] = str(path)
        rows.append(row)
    return rows


def as_bool(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid success value {value!r}")


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - radius, 4), round(center + radius, 4)]


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if not n:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(discordant_a, discordant_b) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def validate(row: dict[str, str], *, init: int, seed: int, checkpoint: str) -> None:
    expected = {
        "suite": "libero_spatial",
        "task_id": "1",
        "init_source": "benchmark",
        "init_index": str(init),
        "seed": str(seed),
        "wait_steps": "10",
        "n_action_steps": "25",
        "checkpoint": checkpoint,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Protocol mismatch {key}={row.get(key)!r}, expected {value!r}: {row['_path']}")
    steps = int(row["steps"])
    if not 1 <= steps <= 300:
        raise ValueError(f"Invalid rollout step count {steps}: {row['_path']}")


def clean(row: dict[str, str]) -> dict:
    return {
        "init_index": int(row["init_index"]),
        "seed": int(row["seed"]),
        "success": as_bool(row["success"]),
        "reward": float(row["total_reward"]),
        "steps": int(row["steps"]),
        "csv": row["_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--joint4-eval-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    final_checkpoint = str(args.checkpoint_root / "040000" / "pretrained_model")

    candidate = rows_under(args.eval_root)
    paired_rows, repeat_rows = {}, []
    for row in candidate:
        path = Path(row["_path"])
        if "paired_20_init" in path.parts:
            init = int(row["init_index"])
            if init not in range(20) or init in paired_rows:
                raise ValueError(f"Duplicate/invalid paired initialization: {row['_path']}")
            validate(row, init=init, seed=12345 + 2 * init, checkpoint=final_checkpoint)
            paired_rows[init] = clean(row)
        elif "fixed_task1_init3" in path.parts:
            validate(row, init=3, seed=12351, checkpoint=final_checkpoint)
            repeat_rows.append(clean(row))
        else:
            raise ValueError(f"Unexpected result location: {row['_path']}")
    if set(paired_rows) != set(range(20)) or len(repeat_rows) != 4:
        raise ValueError(f"Expected 20 paired rows and 4 additional repeats, got {len(paired_rows)} and {len(repeat_rows)}")

    joint_rows = {}
    joint4_checkpoint = "/root/smolvla-training-prep/results/training/teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2/checkpoints/040000/pretrained_model"
    for row in rows_under(args.joint4_eval_root):
        if int(row["task_id"]) != 1:
            continue
        init = int(row["init_index"])
        if init not in range(20) or init in joint_rows:
            raise ValueError(f"Duplicate/invalid joint4 init row: {row['_path']}")
        seed = 12345 + 2 * init
        validate(row, init=init, seed=seed, checkpoint=joint4_checkpoint)
        joint_rows[init] = clean(row)
    if set(joint_rows) != set(range(20)):
        raise ValueError(f"Joint4 task1 reference missing rows: {sorted(set(range(20)) - set(joint_rows))}")

    single_n = sum(r["success"] for r in paired_rows.values())
    joint_n = sum(r["success"] for r in joint_rows.values())
    single_only = sum(paired_rows[i]["success"] and not joint_rows[i]["success"] for i in range(20))
    joint_only = sum(not paired_rows[i]["success"] and joint_rows[i]["success"] for i in range(20))
    fixed = [paired_rows[3], *repeat_rows]
    fixed_successes = sum(r["success"] for r in fixed)
    train_text = args.training_log.read_text(encoding="utf-8", errors="replace")
    marker = "TEACHER_NATIVE_SPATIAL_TASK1_SINGLE_40K_TRAIN_EXIT_CODE=0"
    if marker not in train_text:
        raise ValueError("Successful training exit marker not found")
    error_tokens = ["Traceback (most recent call last)", "RuntimeError:", "TRAIN_FAILED_EXIT="]
    if any(token in train_text for token in error_tokens):
        raise ValueError("Training log contains a failure marker")

    result = {
        "schema_version": 1,
        "date": "2026-10-03",
        "run_id": args.run_id,
        "status": "executed_complete",
        "stage": "native_spatial_task1_single_task_control",
        "training": {"steps": 40000, "batch_size": 2, "sample_draws": 80000, "seed": 1000, "successful_exit_marker": marker},
        "evaluation_protocol": {
            "suite": "libero_spatial",
            "task_id": 1,
            "initialization_indices": list(range(20)),
            "effective_seed_rule": "12345 + 2 * init_index (CSV-recorded evaluator seed)",
            "wait_steps": 10,
            "n_action_steps": 25,
            "max_steps": 300,
            "deterministic_algorithms_strict": False,
            "checkpoint": final_checkpoint,
        },
        "paired_20_initializations": {
            "single_task_successes": single_n,
            "joint4_successes": joint_n,
            "single_task_wilson_95_ci": wilson(single_n, 20),
            "joint4_wilson_95_ci": wilson(joint_n, 20),
            "single_task_only_successes": single_only,
            "joint4_only_successes": joint_only,
            "exact_two_sided_mcnemar_p": round(exact_mcnemar(single_only, joint_only), 6),
            "single_task_rows": [paired_rows[i] for i in range(20)],
            "joint4_rows": [joint_rows[i] for i in range(20)],
        },
        "fixed_init3_repeats": {
            "effective_seed": 12351,
            "n": len(fixed),
            "successes": fixed_successes,
            "success_pattern": [r["success"] for r in fixed],
            "steps": [r["steps"] for r in fixed],
            "note": "Includes the paired init3 observation plus four additional repeats; fixed-condition check only, not a generalization estimate.",
        },
        "interpretation": {
            "signal_threshold": "At least 10/20 paired and 5/5 fixed-init3 is a positive diagnostic signal only.",
            "causal_limit": "Batch size differs from joint4 (2 vs 8); task1-only vs joint4 is diagnostic, not a clean causal estimate of interference.",
            "fold02": "LOCKED",
            "8_task_scaling": "not authorized automatically; review progression gates and matched-exposure plan first",
            "formal_benchmark_changed": False,
        },
        "execution_audit": {
            "paired_summary_csvs": 20,
            "additional_fixed_summary_csvs": 4,
            "training_log": str(args.training_log),
            "evaluation_root": str(args.eval_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "single_task_successes": single_n, "joint4_successes": joint_n, "fixed_init3_successes": fixed_successes}, indent=2))


if __name__ == "__main__":
    main()
