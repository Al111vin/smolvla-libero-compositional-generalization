#!/usr/bin/env python3
"""Replay pinned, saved task-0 actions in CPU-only ControlEnv; never loads a policy."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import stat
import subprocess
import sys
import time
import time
from typing import Any

from build_fresh_seed_stage_manifest_v1 import (
    CHECKPOINT_SHA256,
    EVALUATOR_SHA256,
    ManifestError,
    build_manifest,
    expected_rows,
)
from capture_reset_render_provenance_v1 import capture_model
from controlled_cpu_replay_support_v1 import ENVIRONMENT_SEED, headless_kwargs, initialize_replay
from first_decision_provenance_v1 import digest_tree


EXPERIMENT_ID = "teacher_native_spatial_fresh_seed_stage_cpu_v1_20261012"
SOURCE_ROOT = Path("/root/smolvla-eval-prep/results/fresh_seed_pair_eval_v1_20261011")
MANIFEST_PATH = Path(
    "/root/smolvla-eval-prep/recovery/teacher_native_spatial_fresh_seed_stage_cpu_manifest_v1_20261012.json"
)
OUTPUT_ROOT = Path(
    "/root/smolvla-eval-prep/results/teacher_native_spatial_fresh_seed_stage_cpu_v1_20261012"
)
GPU_LOCK = Path("/tmp/smolvla_gpu_eval.lock")
STATE_ATOL = 1e-6
REWARD_ATOL = 1e-7
LIFT_THRESHOLD_M = 0.03
PLATE_XY_THRESHOLD_M = 0.08
MAX_RUNTIME_SECONDS = 3600


class ReplayError(RuntimeError):
    pass


def deadline(_signum: int, _frame: Any) -> None:
    raise TimeoutError(f"Registered CPU stage-audit wall budget {MAX_RUNTIME_SECONDS}s exhausted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(data: dict[str, Any], field: str) -> str:
    copied = dict(data)
    copied.pop(field, None)
    encoded = json.dumps(copied, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_under(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        raise ReplayError(f"Expected absolute registered path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ReplayError(f"Registered source escapes source root: {path}")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ReplayError(f"Symlinked source path not allowed: {cursor}")
        cursor = cursor.parent
    return resolved


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if manifest_path.resolve() != MANIFEST_PATH:
        raise ReplayError("Manifest path differs from the frozen registration")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("experiment_id") != EXPERIMENT_ID:
        raise ReplayError("Unexpected experiment id")
    if data.get("source_root") != str(SOURCE_ROOT):
        raise ReplayError("Unexpected source root")
    if data.get("source_inventory_count") != 40 or len(data.get("traces", [])) != 40:
        raise ReplayError("Expected exactly 40 pinned traces")
    if data.get("evaluator_sha256") != EVALUATOR_SHA256:
        raise ReplayError("Frozen evaluator identity mismatch")
    if data.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ReplayError("Frozen checkpoint identity mismatch")
    if _canonical_sha(data, "manifest_sha256") != data.get("manifest_sha256"):
        raise ReplayError("Manifest self-hash mismatch")
    fresh = build_manifest(SOURCE_ROOT)
    if fresh.get("manifest_sha256") != data.get("manifest_sha256"):
        raise ReplayError("Live source files or manifest contents changed after pinning")

    expected = expected_rows()
    expected_by_key = {row["key"]: row for row in expected}
    if {row.get("key") for row in data["traces"]} != set(expected_by_key):
        raise ReplayError("Trace keys differ from the frozen 40-row schedule")
    for trace in data["traces"]:
        for name, path_text in trace["paths"].items():
            path = _safe_under(Path(path_text), SOURCE_ROOT)
            if sha256_file(path) != trace["sha256"].get(name):
                raise ReplayError(f"Pinned input hash mismatch: {trace['key']} {name}")
    return data, expected


def require_gpu_idle() -> dict[str, Any]:
    try:
        compute = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
        utilization = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).splitlines()
    except Exception as error:
        raise ReplayError(f"Could not verify GPU is idle: {error}") from error
    active = [line.strip() for line in compute.splitlines() if line.strip()]
    values: list[int] = []
    for line in utilization:
        try:
            values.append(int(line.strip()))
        except ValueError as error:
            raise ReplayError(f"Unparseable GPU utilization: {line!r}") from error
    if active or not values or any(value != 0 for value in values):
        raise ReplayError(f"GPU is not idle: compute_pids={active}, utilization={values}")
    return {"compute_pids": [], "utilization_percent": values}


def _robot_state(obs: dict[str, Any]) -> list[float]:
    import numpy as np

    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ReplayError("Invalid end-effector quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        orientation = np.zeros(3, dtype=np.float32)
    else:
        q = q / norm
        w = float(np.clip(q[3], -1.0, 1.0))
        angle = 2.0 * math.acos(w)
        scale = math.sqrt(max(1.0 - w * w, 0.0))
        axis = np.asarray([1.0, 0.0, 0.0]) if scale < 1e-8 else q[:3] / scale
        orientation = (axis * angle).astype(np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    result = np.concatenate([joint, eef, orientation, gripper]).astype(np.float64)
    if result.shape != (15,) or not np.isfinite(result).all():
        raise ReplayError("Invalid 15-D pre-action robot state")
    return result.tolist()


def _read_source(trace: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]], dict[str, Any]]:
    paths = trace["paths"]
    for name, path_text in paths.items():
        if sha256_file(Path(path_text)) != trace["sha256"].get(name):
            raise ReplayError(f"Pinned input changed immediately before replay: {trace['key']} {name}")
    with Path(paths["summary_csv"]).open(newline="", encoding="utf-8-sig") as stream:
        summaries = list(csv.DictReader(stream))
    with Path(paths["actions_csv"]).open(newline="", encoding="utf-8-sig") as stream:
        actions = list(csv.DictReader(stream))
    protocol = json.loads(Path(paths["protocol_json"]).read_text(encoding="utf-8"))
    capture = json.loads(Path(paths["first_input_json"]).read_text(encoding="utf-8"))
    if len(summaries) != 1 or len(actions) != int(summaries[0]["steps"]):
        raise ReplayError(f"Malformed pinned action evidence: {trace['key']}")
    return summaries[0], actions, {"protocol": protocol, "first_input": capture}


def replay_trace(env: Any, initial_states: Any, trace: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    summary, actions, metadata = _read_source(trace)
    capture = metadata["first_input"]
    init = int(row["init"])
    if int(summary["init_index"]) != init or int(summary["seed"]) != int(row["effective_seed"]):
        raise ReplayError(f"Source summary protocol mismatch for {trace['key']}")

    obs, waits = initialize_replay(
        env,
        np.asarray(initial_states[init], dtype=np.float64),
        int(row["effective_seed"]),
        np,
    )
    if waits != 10:
        raise ReplayError(f"Wait-step mismatch for {trace['key']}: {waits}")

    inner = env.env
    model_arrays = capture_model(env.sim.model._model)
    simulator_state = digest_tree(env.sim.get_state().flatten())
    if model_arrays != capture["model_arrays"]:
        raise ReplayError(f"Compiled model-array mismatch before replay: {trace['key']}")
    if simulator_state != capture["simulator_state"]:
        raise ReplayError(f"Simulator-state mismatch before replay: {trace['key']}")

    bowl_name, plate_name = "akita_black_bowl_1", "plate_1"
    if bowl_name not in inner.object_states_dict or plate_name not in inner.object_states_dict:
        raise ReplayError("Required task-0 bowl/plate object state is absent")
    bowl_state, plate_state = inner.object_states_dict[bowl_name], inner.object_states_dict[plate_name]
    baseline_z = float(obs[f"{bowl_name}_pos"][2])
    first_lift = first_transport = first_terminal = None
    max_state_error = 0.0
    reward_total = 0.0
    terminal_native_matches = True

    for index, source in enumerate(actions):
        if int(source["step"]) != index:
            raise ReplayError(f"Non-contiguous source step at {trace['key']}:{index}")
        current = _robot_state(obs)
        reference = [float(source[f"state_{j}"]) for j in range(15)]
        state_error = max(abs(a - b) for a, b in zip(current, reference))
        if not math.isfinite(state_error) or state_error > STATE_ATOL:
            raise ReplayError(f"Pre-action robot-state parity failed at {trace['key']}:{index}")
        max_state_error = max(max_state_error, state_error)
        action = np.asarray([float(source[f"applied_action_{j}"]) for j in range(7)], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(action < -1.0) or np.any(action > 1.0):
            raise ReplayError(f"Invalid applied action at {trace['key']}:{index}")
        obs, reward, done, _info = env.step(action)
        reward = float(reward)
        source_reward = float(source["reward"])
        if not math.isfinite(reward) or not math.isfinite(source_reward) or abs(reward - source_reward) > REWARD_ATOL:
            raise ReplayError(f"Reward parity failed at {trace['key']}:{index}")
        if (bool(done) or reward > 0.0) and index != len(actions) - 1:
            raise ReplayError(f"Replay terminated before source trace end: {trace['key']}:{index}")
        reward_total += reward

        bowl = np.asarray(obs[f"{bowl_name}_pos"], dtype=np.float64)
        plate = np.asarray(obs[f"{plate_name}_pos"], dtype=np.float64)
        if bowl.shape != (3,) or plate.shape != (3,) or not np.isfinite(bowl).all() or not np.isfinite(plate).all():
            raise ReplayError(f"Invalid bowl/plate pose at {trace['key']}:{index}")
        if first_lift is None and float(bowl[2]) - baseline_z >= LIFT_THRESHOLD_M:
            first_lift = index + 1
        if first_transport is None and float(np.linalg.norm(bowl[:2] - plate[:2])) <= PLATE_XY_THRESHOLD_M:
            first_transport = index + 1
        official = bool(env.check_success())
        native = bool(plate_state.check_ontop(bowl_state))
        if native != official:
            terminal_native_matches = False
            raise ReplayError(f"Native ontop/official success disagreement at {trace['key']}:{index}")
        if official and first_terminal is None:
            first_terminal = index + 1

    steps = len(actions)
    source_success = summary["success"] == "True"
    replay_success = bool(env.check_success()) or reward_total > 0.0
    source_total = float(summary["total_reward"])
    if replay_success != source_success or steps != int(summary["steps"]) or abs(reward_total - source_total) > REWARD_ATOL:
        raise ReplayError(f"Episode summary parity failed for {trace['key']}")
    if source_success and (first_lift is None or first_transport is None or first_terminal is None):
        raise ReplayError(f"Positive control missed a frozen stage predicate: {trace['key']}")
    return {
        "key": trace["key"], "model": trace["model"], "init": init,
        "source_success": source_success, "replay_success": replay_success,
        "source_steps": int(summary["steps"]), "replay_steps": steps,
        "source_total_reward": source_total, "replay_total_reward": reward_total,
        "max_pre_action_state_abs_error": max_state_error,
        "first_lift_step": first_lift, "first_transport_step": first_transport,
        "first_terminal_placement_step": first_terminal,
        "native_ontop_matches_official_each_step": terminal_native_matches,
        "causal_attribution": False,
    }


def _write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "mujoco", "robosuite", "libero"):  # policy/trainer stack deliberately omitted
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-discoverable"
    return result


def run(manifest_path: Path, output_root: Path, authorize_execution: bool) -> dict[str, Any]:
    if not authorize_execution:
        raise ReplayError("Explicit --authorize-execution is required")
    if manifest_path.resolve() != MANIFEST_PATH or output_root.resolve() != OUTPUT_ROOT:
        raise ReplayError("Paths differ from the frozen unique registration")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing output path: {output_root}")
    data, rows = validate_manifest(manifest_path)
    if output_root.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(output_root):
        raise ReplayError("Output path overlaps the raw source tree")
    if not GPU_LOCK.is_file() or GPU_LOCK.is_symlink():
        raise ReplayError(f"Existing regular GPU lock required: {GPU_LOCK}")

    lock_fd = os.open(GPU_LOCK, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        os.close(lock_fd)
        raise ReplayError("GPU lock path must be an existing regular file")
    env = None
    completed: list[dict[str, Any]] = []
    run_manifest: dict[str, Any] = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "status": "preflight", "source_manifest_sha256": data["manifest_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_root": str(SOURCE_ROOT), "output_root": str(output_root),
        "gpu_lock": str(GPU_LOCK), "policy_calls": 0, "training": False,
        "new_rollouts": False, "gpu_budget": 0,
        "raw_actions_or_trajectories_copied": False,
        "fold02": "LOCKED", "eight_task_expansion": "LOCKED",
        "historical_v3_invalidation": "unchanged",
    }
    started_at = time.monotonic()

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_before = require_gpu_idle()
        output_root.mkdir(parents=False, exist_ok=False)
        run_manifest.update(status="running", gpu_before=gpu_before,
                            runtime_versions=_versions())
        _write_json_exclusive(output_root / "manifest.json", run_manifest)

        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv

        suite = benchmark.get_benchmark("libero_spatial")()
        task = suite.get_task(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(f"Official task-0 BDDL is missing: {bddl}")
        bddl_sha = sha256_file(bddl)
        random.seed(ENVIRONMENT_SEED)
        np.random.seed(ENVIRONMENT_SEED)
        env = ControlEnv(**headless_kwargs(bddl))
        initial_states = suite.get_task_init_states(0)
        if initial_states is None or len(initial_states) < 20:
            raise ReplayError("Official task-0 benchmark initial-state list is incomplete")
        # Smoke: one known positive and its paired timeout; then all positives;
        # only after their validation, inspect remaining source-negative traces.
        by_key = {trace["key"]: trace for trace in data["traces"]}
        order = ["single_task0_paired_8_r0_160k", "joint_task0_paired_8_r0_160k"]
        smoke = set(order)
        positives = [trace["key"] for trace in data["traces"]
                     if trace["expected_success"] and trace["key"] not in smoke]
        negatives = [trace["key"] for trace in data["traces"]
                     if not trace["expected_success"] and trace["key"] not in smoke]
        order.extend(positives)
        order.extend(negatives)

        for key in order:
            require_gpu_idle()
            trace = by_key[key]
            row = next(item for item in rows if item["key"] == key)
            try:
                result = replay_trace(env, initial_states, trace, row)
                completed.append(result)
                _write_json_exclusive(output_root / f"{key}.json", result)
            except BaseException as error:
                failure = {
                    "status": "stopped_partial_preserved",
                    "failed_key": key,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_keys": [item["key"] for item in completed],
                    "automatic_retry": False,
                }
                _write_json_exclusive(output_root / "failure.json", failure)
                raise
        gpu_after = require_gpu_idle()
        run_manifest.update(status="completed_diagnostic_not_formal_evidence",
                            bddl_path=str(bddl), bddl_sha256=bddl_sha,
                            gpu_after=gpu_after,
                            completed_trace_count=len(completed),
                            positive_controls=sum(item["source_success"] for item in completed),
                            source_negative_count=sum(not item["source_success"] for item in completed),
                            results=completed,
                            interpretation="Descriptive only; no causal attribution or progression-gate change.")
        env.close()
        env = None
        require_gpu_idle()
        _write_json_exclusive(output_root / "completion.json", run_manifest)
        _write_json_exclusive(output_root / "exit_code.json", {"status": "complete", "exit_code": 0})
        # Initial manifest is immutable; completion is authoritative.
        return run_manifest
    except BaseException as error:
        if output_root.exists():
            if not (output_root / "failure.json").exists():
                _write_json_exclusive(output_root / "failure.json", {
                    "status": "stopped_partial_preserved", "failed_key": "preflight_or_runtime",
                    "error_type": type(error).__name__, "error": str(error),
                    "completed_keys": [item["key"] for item in completed],
                    "automatic_retry": False,
                })
            if not (output_root / "exit_code.json").exists():
                _write_json_exclusive(output_root / "exit_code.json", {
                    "status": "error", "exit_code": 1,
                    "error_type": type(error).__name__,
                })
        raise
    finally:
        elapsed = time.monotonic() - started_at
        if elapsed > MAX_RUNTIME_SECONDS and output_root.exists() and not (output_root / "failure.json").exists():
            _write_json_exclusive(output_root / "failure.json", {
                "status": "stopped_partial_preserved", "failed_key": "wall_budget",
                "elapsed_seconds": elapsed, "automatic_retry": False,
            })
        try:
            if env is not None:
                env.close()
        finally:
            os.close(lock_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    args = parser.parse_args()
    if not hasattr(signal, "SIGALRM"):
        raise ReplayError("A signal alarm is required for bounded CPU replay")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(MAX_RUNTIME_SECONDS)
    try:
        result = run(args.manifest, args.results_dir, args.authorize_execution)
    finally:
        signal.alarm(0)
    print(json.dumps({"status": result["status"], "completed_trace_count": result["completed_trace_count"],
                      "gpu_budget": result["gpu_budget"], "policy_calls": result["policy_calls"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
