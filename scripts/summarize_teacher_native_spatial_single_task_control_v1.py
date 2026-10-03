#!/usr/bin/env python3
"""Validate and summarize native Spatial task0 single-task matched-recipe control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_rows(root: Path) -> list[dict[str, str]]:
    result = []
    for path in sorted(root.rglob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"Expected one CSV row in {path}, found {len(rows)}")
        row = rows[0]
        row["_path"] = str(path)
        result.append(row)
    return result


def as_bool(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid success value {value!r}")


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - rad, 4), round(center + rad, 4)]


def mcnemar_exact(single_only: int, joint_only: int) -> float:
    n = single_only + joint_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(single_only, joint_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def validate_row(row: dict[str, str], *, task_id: int, init: int, seed: int, checkpoint: str) -> None:
    expected = {
        "suite": "libero_spatial",
        "task_id": str(task_id),
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
        raise ValueError(f"Invalid step count {steps}: {row['_path']}")


def clean(row: dict[str, str]) -> dict:
    return {
        "init_index": int(row["init_index"]),
        "seed": int(row["seed"]),
        "success": as_bool(row["success"]),
        "reward": float(row["total_reward"]),
        "steps": int(row["steps"]),
        "csv": row["_path"],
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--joint4-eval-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.task_id != 0:
        raise ValueError("This registered control is specifically LIBERO Spatial task0")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    final_checkpoint = str(args.checkpoint_root / "040000" / "pretrained_model")
    candidate_rows = read_rows(args.eval_root)
    paired, repeats, probes = {}, [], {}
    for row in candidate_rows:
        path = Path(row["_path"])
        if "paired_20_init" in path.parts:
            init = int(row["init_index"])
            if init not in range(20) or init in paired:
                raise ValueError(f"Duplicate or invalid paired row: {row['_path']}")
            validate_row(row, task_id=0, init=init, seed=12345 + 2 * init, checkpoint=final_checkpoint)
            paired[init] = clean(row)
        elif "fixed_task0_init3" in path.parts:
            validate_row(row, task_id=0, init=3, seed=12351, checkpoint=final_checkpoint)
            repeats.append(clean(row))
        elif "checkpoint_probe" in path.parts:
            step = int(next(part.removeprefix("step_") for part in path.parts if part.startswith("step_")))
            if step not in (10000, 20000, 30000) or step in probes:
                raise ValueError(f"Unexpected/duplicate checkpoint probe: {row['_path']}")
            expected_checkpoint = str(args.checkpoint_root / f"{step:06d}" / "pretrained_model")
            validate_row(row, task_id=0, init=3, seed=12351, checkpoint=expected_checkpoint)
            probes[step] = clean(row)
        else:
            raise ValueError(f"Unexpected evaluation result location: {row['_path']}")
    if set(paired) != set(range(20)) or len(repeats) != 4 or set(probes) != {10000, 20000, 30000}:
        raise ValueError(f"Incomplete rows: paired={len(paired)}, repeats={len(repeats)}, probes={sorted(probes)}")

    joint_checkpoint = "/root/smolvla-training-prep/results/training/teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2/checkpoints/040000/pretrained_model"
    joint_rows = {}
    for row in read_rows(args.joint4_eval_root):
        if int(row["task_id"]) != 0:
            continue
        init = int(row["init_index"])
        if init not in range(20) or init in joint_rows:
            raise ValueError(f"Duplicate/invalid joint4 row: {row['_path']}")
        validate_row(row, task_id=0, init=init, seed=12345 + 2 * init, checkpoint=joint_checkpoint)
        joint_rows[init] = clean(row)
    if set(joint_rows) != set(range(20)):
        raise ValueError(f"Joint4 baseline lacks paired rows: {sorted(set(range(20)) - set(joint_rows))}")

    single_successes = sum(row["success"] for row in paired.values())
    joint_successes = sum(row["success"] for row in joint_rows.values())
    single_only = sum(paired[i]["success"] and not joint_rows[i]["success"] for i in range(20))
    joint_only = sum(not paired[i]["success"] and joint_rows[i]["success"] for i in range(20))
    fixed = [paired[3], *repeats]
    fixed_successes = sum(row["success"] for row in fixed)
    training_text = args.training_log.read_text(encoding="utf-8", errors="replace")
    marker = "TEACHER_NATIVE_SPATIAL_TASK0_SINGLE_40K_TRAIN_EXIT_CODE=0"
    if marker not in training_text:
        raise ValueError("Successful training exit marker not found")
    for token in ("Traceback (most recent call last)", "RuntimeError:", "TRAIN_FAILED_EXIT="):
        if token in training_text:
            raise ValueError(f"Training log contains failure marker {token!r}")
    for path in args.eval_root.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("Traceback (most recent call last)", "RuntimeError:", "CUDA error"):
            if token in text:
                raise ValueError(f"Evaluation log contains failure marker {token!r}: {path}")

    checkpoint_hashes = {}
    for step in (10000, 20000, 30000, 40000):
        path = args.checkpoint_root / f"{step:06d}" / "pretrained_model" / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_hashes[str(step)] = sha256(path)

    fixed_gate = fixed_successes == 5
    result = {
        "schema_version": 1,
        "date": "2026-10-03",
        "status": "executed_complete",
        "stage": "native_spatial_task0_single_task_matched_recipe_control",
        "run_id": args.run_id,
        "training": {
            "steps": 40000,
            "batch_size": 2,
            "task0_sample_draws": 80000,
            "seed": 1000,
            "learning_rate": 1e-4,
            "checkpoint_model_safetensors_sha256": checkpoint_hashes,
            "training_log": str(args.training_log),
        },
        "evaluation_protocol": {
            "task_id": 0,
            "suite": "libero_spatial",
            "effective_seed_rule": "12345 + 2*init_index (CSV-recorded evaluator seed)",
            "wait_steps": 10,
            "n_action_steps": 25,
            "max_steps": 300,
            "strict_deterministic_algorithms": False,
            "final_checkpoint": final_checkpoint,
            "checkpoint_probe_steps": [10000, 20000, 30000],
        },
        "paired_20_initializations": {
            "single_task_successes": single_successes,
            "single_task_wilson_95_ci": wilson(single_successes, 20),
            "joint4_successes": joint_successes,
            "joint4_wilson_95_ci": wilson(joint_successes, 20),
            "single_task_only_successes": single_only,
            "joint4_only_successes": joint_only,
            "exact_two_sided_mcnemar_p": round(mcnemar_exact(single_only, joint_only), 6),
            "single_task_rows": [paired[i] for i in range(20)],
            "joint4_rows": [joint_rows[i] for i in range(20)],
        },
        "fixed_init3_repeats": {
            "effective_seed": 12351,
            "n": len(fixed),
            "successes": fixed_successes,
            "success_pattern": [row["success"] for row in fixed],
            "steps": [row["steps"] for row in fixed],
            "note": "Includes the paired init3 row plus four additional repeats; fixed-condition check only.",
        },
        "checkpoint_learning_curve_probes": {
            str(step): probes[step] for step in (10000, 20000, 30000)
        },
        "interpretation": {
            "positive_diagnostic_signal": single_successes >= 10 and fixed_gate,
            "formal_acceptance": False,
            "causal_limit": "Single task batch2 vs joint4 batch8 changes gradient composition; task0 exposure, optimizer updates, base, LR and pooled normalization stats are matched.",
            "if_positive": "Do not automatically scale to 8 tasks; review the registered 4-task gate and matched-exposure progression design.",
            "fold02": "LOCKED",
            "official_benchmark_changed": False,
        },
        "execution_audit": {
            "paired_rows": 20,
            "additional_fixed_repeats": 4,
            "checkpoint_probes": 3,
            "evaluation_root": str(args.eval_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "single_task_successes": single_successes,
        "joint4_successes": joint_successes,
        "fixed_init3_successes": fixed_successes,
        "mcnemar_p": result["paired_20_initializations"]["exact_two_sided_mcnemar_p"],
    }, indent=2))


if __name__ == "__main__":
    main()
