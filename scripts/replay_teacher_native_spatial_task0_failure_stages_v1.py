#!/usr/bin/env python3
"""Instrumented, policy-free replay of registered native task-0 action traces.

This is a diagnostic replay, not a policy evaluation and not formal benchmark
evidence. The input manifest points to private summary/action CSVs. The script
refuses to overwrite an output root, validates the complete trace inventory
before starting, and stops at the first replay/observer parity failure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


TRACE_COUNT = 24
PAIRED_COUNT = 20
FIXED_REPEAT_COUNT = 4
TASK_ID = 0
TASK_SUITE = "libero_spatial"
N_ACTION_STEPS = 25
WAIT_STEPS = 10
MAX_STEPS = 300
STATE_ATOL = 1e-6
REWARD_ATOL = 1e-7
APPROACH_THRESHOLD_M = 0.05
LIFT_THRESHOLD_M = 0.03
PLATE_XY_THRESHOLD_M = 0.08
TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

SUMMARY_FIELDS = {
    "suite",
    "task_id",
    "init_source",
    "init_index",
    "language",
    "success",
    "total_reward",
    "steps",
    "wait_steps",
    "n_action_steps",
    "seed",
    "checkpoint",
}
ACTION_FIELDS = {"step", "reward"}
ACTION_FIELDS.update(
    f"applied_action_{index}" for index in range(7)
)
ACTION_FIELDS.update(f"state_{index}" for index in range(15))
MANIFEST_FIELDS = {
    "trace_id",
    "group",
    "init_index",
    "cli_seed",
    "effective_seed",
    "expected_success",
    "summary_csv",
    "summary_sha256",
    "actions_csv",
    "actions_sha256",
}
TASK0_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


class ProtocolError(RuntimeError):
    """Raised when inputs or replay evidence fail a registered check."""


def parse_bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ProtocolError(f"Invalid boolean in {field}: {value!r}")


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"Invalid number in {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ProtocolError(f"Non-finite value in {field}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_unreached_stage(events: dict[str, bool]) -> str | None:
    """Return the first missing preregistered stage proxy, not a causal label."""
    for stage in ("approach", "grasp", "lift", "transport", "placement_terminal"):
        if not bool(events.get(stage, False)):
            return stage
    return None


def validate_manifest_shape(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("schema_version") != 1:
        raise ProtocolError("Unsupported manifest schema_version")
    if data.get("experiment_id") != "teacher_native_spatial_task0_failure_stage_replay_v1_20261004":
        raise ProtocolError("Unexpected experiment_id")
    if data.get("task_suite") != TASK_SUITE or int(data.get("task_id", -1)) != TASK_ID:
        raise ProtocolError("Manifest must target LIBERO Spatial task 0")
    records = data.get("traces")
    if not isinstance(records, list) or len(records) != TRACE_COUNT:
        raise ProtocolError(f"Expected exactly {TRACE_COUNT} manifest records")
    trace_ids: set[str] = set()
    paired: dict[int, int] = {}
    repeats = 0
    paired_successes = 0
    repeat_successes = 0
    for record in records:
        if not isinstance(record, dict) or not MANIFEST_FIELDS.issubset(record):
            raise ProtocolError("Trace manifest row is missing required fields")
        trace_id = str(record["trace_id"])
        if not TRACE_ID_RE.fullmatch(trace_id) or trace_id in trace_ids:
            raise ProtocolError(f"Invalid or duplicate trace_id: {trace_id!r}")
        trace_ids.add(trace_id)
        group = record["group"]
        init_index = int(record["init_index"])
        if group == "paired":
            if not 0 <= init_index < 20:
                raise ProtocolError(f"Invalid paired init_index: {init_index}")
            paired[init_index] = paired.get(init_index, 0) + 1
        elif group == "fixed_init3_repeat":
            if init_index != 3:
                raise ProtocolError("All additional fixed repeats must use init_index 3")
            repeats += 1
        else:
            raise ProtocolError(f"Unknown trace group: {group!r}")
        cli_seed = int(record["cli_seed"])
        effective_seed = int(record["effective_seed"])
        if effective_seed != cli_seed + init_index:
            raise ProtocolError(f"Seed rule mismatch for {trace_id}")
        if not isinstance(record["expected_success"], bool):
            raise ProtocolError(f"expected_success must be a JSON boolean for {trace_id}")
        if group == "paired" and record["expected_success"]:
            paired_successes += 1
        if group == "fixed_init3_repeat" and record["expected_success"]:
            repeat_successes += 1
    if paired != {index: 1 for index in range(PAIRED_COUNT)}:
        raise ProtocolError("Paired traces must contain each init_index 0..19 exactly once")
    if repeats != FIXED_REPEAT_COUNT:
        raise ProtocolError(f"Expected {FIXED_REPEAT_COUNT} additional init3 repeats")
    if paired_successes != 11 or repeat_successes != 3:
        raise ProtocolError(
            "Source outcome inventory differs from the registered final-checkpoint result "
            f"(paired={paired_successes}/20, extra_init3={repeat_successes}/4)"
        )
    return sorted(records, key=lambda row: (row["group"] != "paired", int(row["init_index"]), row["trace_id"]))


def _read_one_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ProtocolError(f"Empty {label}: {path}")
    return fields, rows


def validate_source_records(
    manifest_path: Path,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_root = manifest_path.parent
    validated = []
    unique_files: set[Path] = set()
    for record in records:
        item = dict(record)
        paths = {}
        for key in ("summary_csv", "actions_csv"):
            path = Path(str(item[key])).expanduser()
            if not path.is_absolute():
                path = (manifest_root / path).resolve()
            else:
                path = path.resolve()
            if path in unique_files:
                raise ProtocolError(f"Input file reused by multiple traces: {path}")
            unique_files.add(path)
            paths[key] = path
        summary_fields, summaries = _read_one_csv(paths["summary_csv"], "summary CSV")
        if len(summaries) != 1 or not SUMMARY_FIELDS.issubset(summary_fields):
            raise ProtocolError(f"Invalid summary schema/row count: {paths['summary_csv']}")
        summary = summaries[0]
        actions_fields, action_rows = _read_one_csv(paths["actions_csv"], "action CSV")
        if not ACTION_FIELDS.issubset(actions_fields):
            raise ProtocolError(f"Invalid action schema: {paths['actions_csv']}")
        expected_index = int(item["init_index"])
        if summary["suite"] != TASK_SUITE or int(summary["task_id"]) != TASK_ID:
            raise ProtocolError(f"Wrong suite/task in {paths['summary_csv']}")
        if summary["language"].strip() != TASK0_TEXT:
            raise ProtocolError(f"Wrong task language in {paths['summary_csv']}")
        if summary["init_source"] != "benchmark" or int(summary["init_index"]) != expected_index:
            raise ProtocolError(f"Initialization mismatch in {paths['summary_csv']}")
        if int(summary["seed"]) != int(item["effective_seed"]):
            raise ProtocolError(f"Recorded effective seed mismatch in {paths['summary_csv']}")
        if int(summary["wait_steps"]) != WAIT_STEPS or int(summary["n_action_steps"]) != N_ACTION_STEPS:
            raise ProtocolError(f"Evaluation protocol mismatch in {paths['summary_csv']}")
        steps = int(summary["steps"])
        if steps != len(action_rows) or not 1 <= steps <= MAX_STEPS:
            raise ProtocolError(f"Step count mismatch in {paths['summary_csv']}")
        checkpoint = str(summary["checkpoint"])
        if "040000" not in checkpoint and "40000" not in checkpoint:
            raise ProtocolError(f"Trace is not identified as final step-40000 checkpoint: {checkpoint}")
        if "teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1" not in checkpoint:
            raise ProtocolError(f"Trace is not from the registered single-task run: {checkpoint}")
        source_success = parse_bool(summary["success"], "summary.success")
        if source_success != item["expected_success"]:
            raise ProtocolError(f"Manifest outcome differs from summary for {item['trace_id']}")
        total_reward = finite_float(summary["total_reward"], "summary.total_reward")
        states = []
        actions = []
        rewards = []
        for expected_step, row in enumerate(action_rows):
            if int(row["step"]) != expected_step:
                raise ProtocolError(f"Non-contiguous source action steps in {paths['actions_csv']}")
            state = [finite_float(row[f"state_{i}"], f"state_{i}") for i in range(15)]
            action = [finite_float(row[f"applied_action_{i}"], f"applied_action_{i}") for i in range(7)]
            if any(value < -1.000001 or value > 1.000001 for value in action):
                raise ProtocolError(f"Applied action outside [-1,1] in {paths['actions_csv']}")
            states.append(state)
            actions.append(action)
            rewards.append(finite_float(row["reward"], "action.reward"))
        summary_sha = sha256_file(paths["summary_csv"])
        actions_sha = sha256_file(paths["actions_csv"])
        if str(item["summary_sha256"]).lower() != summary_sha:
            raise ProtocolError(f"Summary CSV SHA256 mismatch for {item['trace_id']}")
        if str(item["actions_sha256"]).lower() != actions_sha:
            raise ProtocolError(f"Action CSV SHA256 mismatch for {item['trace_id']}")
        item.update(
            {
                "summary_path": str(paths["summary_csv"]),
                "actions_path": str(paths["actions_csv"]),
                "summary_sha256": summary_sha,
                "actions_sha256": actions_sha,
                "source_success": source_success,
                "source_total_reward": total_reward,
                "source_steps": steps,
                "checkpoint": checkpoint,
                "states": states,
                "actions": actions,
                "rewards": rewards,
            }
        )
        validated.append(item)
    inventory = {
        "trace_count": len(validated),
        "paired_count": sum(item["group"] == "paired" for item in validated),
        "fixed_init3_repeat_count": sum(item["group"] == "fixed_init3_repeat" for item in validated),
        "input_files": [
            {
                "trace_id": item["trace_id"],
                "summary_sha256": item["summary_sha256"],
                "actions_sha256": item["actions_sha256"],
                "source_success": item["source_success"],
                "source_steps": item["source_steps"],
            }
            for item in validated
        ],
    }
    return validated, inventory


def _resolve_collision_geoms(env, object_name: str) -> list[str]:
    inner = env.env
    if object_name not in inner.obj_body_id:
        raise ProtocolError(f"Object root body unavailable: {object_name}")
    model = inner.sim.model
    root = int(inner.obj_body_id[object_name])
    parents = [int(value) for value in model.body_parentid]
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if body_id not in descendants and parent_id in descendants:
                descendants.add(body_id)
                changed = True
    geom_body_ids = [int(value) for value in model.geom_bodyid]
    contype = [int(value) for value in model.geom_contype]
    conaffinity = [int(value) for value in model.geom_conaffinity]
    names = []
    for geom_id, body_id in enumerate(geom_body_ids):
        if body_id in descendants and (contype[geom_id] != 0 or conaffinity[geom_id] != 0):
            name = model.geom_id2name(geom_id)
            if name:
                names.append(str(name))
    names = sorted(set(names))
    if not names:
        raise ProtocolError(f"No collision geoms for {object_name}")
    return names


def _vec3(obs: dict[str, Any], key: str) -> list[float]:
    value = [finite_float(item, key) for item in list(obs[key])]
    if len(value) != 3:
        raise ProtocolError(f"Expected {key} to have 3 values")
    return value


def _robot_state(obs: dict[str, Any]) -> list[float]:
    from eval_v3_task0 import quat_xyzw_to_axis_angle

    import numpy as np

    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    ori = quat_xyzw_to_axis_angle(obs["robot0_eef_quat"])
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    state = np.concatenate([joint, eef, ori, gripper]).astype(np.float64)
    if state.shape != (15,) or not all(math.isfinite(float(x)) for x in state):
        raise ProtocolError("Invalid 15-D robot state")
    return [float(x) for x in state]


def _events_from_rows(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "approach": any(bool(row["approach"]) for row in rows),
        "grasp": any(bool(row["grasp_contact_proxy"]) for row in rows),
        "lift": any(bool(row["lifted"]) for row in rows),
        "transport": any(bool(row["within_plate_xy_threshold"]) for row in rows),
        "placement_terminal": any(bool(row["official_success"]) for row in rows),
    }


def _first_event_steps(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    stage_fields = {
        "approach": "approach",
        "grasp": "grasp_contact_proxy",
        "lift": "lifted",
        "transport": "within_plate_xy_threshold",
        "placement_terminal": "official_success",
    }
    return {
        stage: next(
            (int(row["step"]) for row in rows if bool(row[field])),
            None,
        )
        for stage, field in stage_fields.items()
    }


def _stage_sequence_is_monotone(first_steps: dict[str, int | None]) -> bool:
    ordered = ("approach", "grasp", "lift", "transport", "placement_terminal")
    observed = [(stage, first_steps.get(stage)) for stage in ordered if first_steps.get(stage) is not None]
    last_step = -1
    prior_stages = set()
    for stage, step in observed:
        if step is None or step < last_step:
            return False
        stage_position = ordered.index(stage)
        if any(ordered.index(previous) > stage_position for previous in prior_stages):
            return False
        if any(first_steps.get(previous) is None for previous in ordered[:stage_position]):
            return False
        last_step = step
        prior_stages.add(stage)
    return True


def classify_stage_proxies(
    events: dict[str, bool], first_steps: dict[str, int | None]
) -> tuple[str | None, bool]:
    """Classify only monotone proxy sequences; keep non-monotone traces ambiguous."""
    monotone = _stage_sequence_is_monotone(first_steps)
    if not monotone:
        return "ambiguous_nonmonotone_sequence", False
    return first_unreached_stage(events), True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_failure(root: Path, trace_id: str, error: BaseException, completed: list[dict[str, Any]]) -> None:
    _atomic_json(
        root / "failure.json",
        {
            "status": "stopped_with_preserved_partial_outputs",
            "failed_trace_id": trace_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_trace_ids": [row["trace_id"] for row in completed],
            "timestamp_utc_epoch": time.time(),
            "automatic_retry": False,
        },
    )


def validate_output_root(output_root: Path, cases: list[dict[str, Any]]) -> None:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise ProtocolError(f"Refusing existing output root: {output_root}")
    source_paths = [Path(case[key]).resolve() for case in cases for key in ("summary_path", "actions_path")]
    if any(path.is_relative_to(output_root) for path in source_paths):
        raise ProtocolError("Output root must not contain any source trace")


def require_execution_authorization(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "authorize_execution", False)):
        raise ProtocolError(
            "Simulator replay is disabled unless --authorize-execution is explicitly supplied"
        )


def _replay_trace(env, task, initial_states, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    init_index = int(case["init_index"])
    initial_state = np.asarray(initial_states[init_index], dtype=np.float64)
    obs = env.reset()
    updated = env.set_init_state(initial_state)
    if updated is not None:
        obs = updated
    zero_action = np.zeros(7, dtype=np.float32)
    wait_executed = 0
    for _ in range(WAIT_STEPS):
        obs, _, done, _ = env.step(zero_action)
        wait_executed += 1
        if done:
            break
    np.random.seed(int(case["effective_seed"]))
    torch.manual_seed(int(case["effective_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(case["effective_seed"]))

    inner = env.env
    bowl_name = "akita_black_bowl_1"
    plate_name = "plate_1"
    for name in (bowl_name, plate_name):
        if name not in inner.object_states_dict or f"{name}_pos" not in obs:
            raise ProtocolError(f"Required task-0 object missing: {name}")
    ramekin_name = "ramekin_1"
    ramekin_available = ramekin_name in inner.object_states_dict and f"{ramekin_name}_pos" in obs
    bowl_geoms = _resolve_collision_geoms(env, bowl_name)
    important_geoms = inner.robots[0].gripper.important_geoms
    left_geoms = list(important_geoms.get("left_fingerpad", []))
    right_geoms = list(important_geoms.get("right_fingerpad", []))
    if not left_geoms or not right_geoms:
        raise ProtocolError("Could not resolve both finger-pad geometry sets")
    bowl_state = inner.object_states_dict[bowl_name]
    plate_state = inner.object_states_dict[plate_name]
    ramekin_geoms = _resolve_collision_geoms(env, ramekin_name) if ramekin_available else []
    initial_bowl = _vec3(obs, f"{bowl_name}_pos")
    baseline_z = initial_bowl[2]
    rows = []
    total_reward = 0.0
    first_success_step = None
    for index, action_values in enumerate(case["actions"]):
        if index >= MAX_STEPS:
            raise ProtocolError("Source trace exceeds registered maximum step cap")
        state_now = _robot_state(obs)
        source_state = case["states"][index]
        state_max_abs_diff = max(abs(a - b) for a, b in zip(state_now, source_state))
        if state_max_abs_diff > STATE_ATOL:
            raise ProtocolError(
                f"Pre-action robot-state replay mismatch at step {index}: "
                f"max_abs_diff={state_max_abs_diff:.9g} > {STATE_ATOL}"
            )
        pre_eef = _vec3(obs, "robot0_eef_pos")
        pre_bowl = _vec3(obs, f"{bowl_name}_pos")
        pre_plate = _vec3(obs, f"{plate_name}_pos")
        pre_eef_to_bowl = math.dist(pre_eef, pre_bowl)
        approach = pre_eef_to_bowl <= APPROACH_THRESHOLD_M
        action = np.asarray(action_values, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        reward = float(reward)
        source_reward = case["rewards"][index]
        if abs(reward - source_reward) > REWARD_ATOL:
            raise ProtocolError(
                f"Reward replay mismatch at step {index}: {reward} != {source_reward}"
            )
        if (bool(done) or reward > 0.0) and index != len(case["actions"]) - 1:
            raise ProtocolError(
                f"Replay terminated before final source row at step {index}"
            )
        total_reward += reward
        official_success = bool(env.check_success())
        info_success = bool(info.get("success", info.get("is_success", False)))
        if official_success and first_success_step is None:
            first_success_step = index + 1
        eef = _vec3(obs, "robot0_eef_pos")
        bowl = _vec3(obs, f"{bowl_name}_pos")
        plate = _vec3(obs, f"{plate_name}_pos")
        ramekin = _vec3(obs, f"{ramekin_name}_pos") if ramekin_available else None
        eef_to_bowl = math.dist(eef, bowl)
        bowl_to_plate_xy = math.dist(bowl[:2], plate[:2])
        left_contact = bool(inner.check_contact(left_geoms, bowl_geoms))
        right_contact = bool(inner.check_contact(right_geoms, bowl_geoms))
        grasp_proxy = bool(inner._check_grasp(inner.robots[0].gripper, bowl_geoms))
        if grasp_proxy != (left_contact and right_contact):
            raise ProtocolError(
                f"Grasp/contact proxy disagreement at step {index + 1}"
            )
        ontop = bool(plate_state.check_ontop(bowl_state))
        if ontop != official_success:
            raise ProtocolError(
                f"Native placement predicate disagrees with official success at step {index + 1}"
            )
        bowl_contacts_plate = bool(bowl_state.check_contact(plate_state))
        ramekin_contact = (
            bool(bowl_state.check_contact(inner.object_states_dict[ramekin_name]))
            if ramekin_available
            else None
        )
        rows.append(
            {
                "step": index + 1,
                "source_step_zero_based": index,
                "source_robot_state_max_abs_diff": float(state_max_abs_diff),
                "reward": reward,
                "done": bool(done),
                "info_success": info_success,
                "official_success": official_success,
                "pre_action_eef_xyz": pre_eef,
                "pre_action_bowl_xyz": pre_bowl,
                "pre_action_plate_xyz": pre_plate,
                "pre_action_eef_to_bowl_distance_m": pre_eef_to_bowl,
                "eef_xyz": eef,
                "bowl_xyz": bowl,
                "plate_xyz": plate,
                "ramekin_xyz": ramekin,
                "eef_to_bowl_distance_m": eef_to_bowl,
                "bowl_lift_from_post_wait_baseline_m": bowl[2] - baseline_z,
                "bowl_to_plate_xy_distance_m": bowl_to_plate_xy,
                "approach": approach,
                "left_fingerpad_bowl_contact": left_contact,
                "right_fingerpad_bowl_contact": right_contact,
                "grasp_contact_proxy": grasp_proxy,
                "lifted": bowl[2] - baseline_z >= LIFT_THRESHOLD_M,
                "within_plate_xy_threshold": bowl_to_plate_xy <= PLATE_XY_THRESHOLD_M,
                "bowl_contact_plate": bowl_contacts_plate,
                "bowl_contact_ramekin": ramekin_contact,
                "native_plate_ontop": ontop,
            }
        )
    final_info = rows[-1]["info_success"] if rows else False
    replay_success = bool(env.check_success()) or final_info or total_reward > 0.0
    success_parity = replay_success == bool(case["source_success"])
    steps_parity = len(rows) == int(case["source_steps"])
    reward_parity = abs(total_reward - float(case["source_total_reward"])) <= REWARD_ATOL
    if not steps_parity or not reward_parity or not success_parity:
        raise ProtocolError(
            "Episode summary replay mismatch: "
            f"steps={len(rows)}/{case['source_steps']}, "
            f"reward={total_reward}/{case['source_total_reward']}, "
            f"success={replay_success}/{case['source_success']}"
        )
    events = _events_from_rows(rows)
    first_event_steps = _first_event_steps(rows)
    stage_label, sequence_monotone = classify_stage_proxies(events, first_event_steps)
    return (
        {
            "trace_id": case["trace_id"],
            "group": case["group"],
            "init_index": init_index,
            "effective_seed": int(case["effective_seed"]),
            "source_success": bool(case["source_success"]),
            "replay_success": replay_success,
            "source_steps": int(case["source_steps"]),
            "replay_steps": len(rows),
            "source_total_reward": float(case["source_total_reward"]),
            "replay_total_reward": total_reward,
            "state_max_abs_diff": max(row["source_robot_state_max_abs_diff"] for row in rows),
            "wait_steps_executed": wait_executed,
            "stage_events": events,
            "stage_first_observed_step": first_event_steps,
            "stage_sequence_monotone": sequence_monotone,
            "first_unreached_stage_proxy": None if replay_success else stage_label,
            "stage_proxy_classification_valid": sequence_monotone,
            "causal_attribution": False,
        },
        rows,
    )


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_manifest_shape(data)
    cases, inventory = validate_source_records(manifest_path, records)
    output_root = Path(args.results_dir).expanduser().resolve()
    validate_output_root(output_root, cases)
    require_execution_authorization(args)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    completed: list[dict[str, Any]] = []
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": data["experiment_id"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "policy_inference": False,
        "training": False,
        "dataset_modified": False,
        "checkpoint_modified": False,
        "formal_benchmark_evidence": False,
        "task_count_scaling": "LOCKED",
        "fold02": "LOCKED",
        "inventory": inventory,
        "output_root": str(output_root),
    }
    _atomic_json(output_root / "manifest.json", run_manifest)
    try:
        import numpy as np

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark(TASK_SUITE)()
        task = suite.get_task(TASK_ID)
        initial_states = suite.get_task_init_states(TASK_ID)
        if initial_states is None or len(initial_states) < PAIRED_COUNT:
            raise ProtocolError("Task-0 benchmark initial-state list is incomplete")
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl_path.is_file():
            raise FileNotFoundError(f"Task-0 BDDL missing: {bddl_path}")
        run_manifest["bddl_sha256"] = sha256_file(bddl_path)
        run_manifest["language"] = str(task.language)
        run_manifest["bddl_path"] = str(bddl_path)
        _atomic_json(output_root / "manifest.json", run_manifest)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path), camera_heights=128, camera_widths=128
        )
        try:
            successful = next(case for case in cases if case["expected_success"])
            failed = next(case for case in cases if not case["expected_success"])
            smoke_ids = {successful["trace_id"], failed["trace_id"]}
            ordered = [case for case in cases if case["trace_id"] in smoke_ids]
            ordered.extend(case for case in cases if case["trace_id"] not in smoke_ids)
            for case in ordered:
                trace_path = output_root / f"{case['trace_id']}.jsonl.gz"
                try:
                    summary, rows = _replay_trace(env, task, initial_states, case)
                    with gzip.open(trace_path, "wt", encoding="utf-8", newline="") as file:
                        for row in rows:
                            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    summary["trace_path"] = trace_path.name
                    summary["trace_sha256"] = sha256_file(trace_path)
                    completed.append(summary)
                    print(
                        f"completed {case['trace_id']} success={summary['replay_success']} "
                        f"steps={summary['replay_steps']} stage={summary['first_unreached_stage_proxy']}"
                    )
                except Exception as error:
                    _write_failure(output_root, case["trace_id"], error, completed)
                    raise
        finally:
            env.close()
    except Exception as error:
        if not (output_root / "failure.json").exists():
            _write_failure(output_root, "preflight_or_environment", error, completed)
        run_manifest["status"] = "stopped_partial"
        _atomic_json(output_root / "manifest.json", run_manifest)
        raise
    outcomes = {
        "traces_completed": len(completed),
        "source_successes": sum(bool(item["source_success"]) for item in completed),
        "source_failures": sum(not bool(item["source_success"]) for item in completed),
        "monotone_proxy_classifications": sum(bool(item["stage_proxy_classification_valid"]) for item in completed),
        "ambiguous_nonmonotone_sequences": sum(not bool(item["stage_sequence_monotone"]) for item in completed),
        "first_unreached_stage_counts": {},
    }
    for item in completed:
        stage = item["first_unreached_stage_proxy"]
        if stage is not None:
            outcomes["first_unreached_stage_counts"][stage] = outcomes["first_unreached_stage_counts"].get(stage, 0) + 1
    run_manifest["status"] = "completed_diagnostic_not_formal_evidence"
    run_manifest["completed_traces"] = completed
    run_manifest["outcomes"] = outcomes
    run_manifest["interpretation_limit"] = (
        "Descriptive action-trace replay only. Replay parity is required; stage proxies do not establish causality or generalization."
    )
    _atomic_json(output_root / "manifest.json", run_manifest)
    print(json.dumps(outcomes, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Private 24-trace input manifest JSON")
    parser.add_argument("--results-dir", required=True, help="New, absent private output root")
    parser.add_argument("--validate-only", action="store_true", help="Validate manifest and CSVs without creating output or importing LIBERO")
    parser.add_argument(
        "--authorize-execution",
        action="store_true",
        help="Required fail-closed acknowledgement before creating output or starting simulator replay",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_manifest_shape(data)
    cases, inventory = validate_source_records(manifest_path, records)
    validate_output_root(Path(args.results_dir), cases)
    if args.validate_only:
        print(json.dumps({"inventory": inventory, "results_root_absent": True}, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
