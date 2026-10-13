#!/usr/bin/env python3
"""V3: two-trace policy-free init-8 replay smoke, one clean worker per trace."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import stat
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_child_signals_v4 import run_with_signal_cleanup
from replay_fresh_seed_cpu_stage_v1 import (
    GPU_LOCK as SOURCE_GPU_LOCK,
    MANIFEST_PATH as SOURCE_MANIFEST,
    ReplayError,
    expected_rows,
    require_gpu_idle,
    replay_trace,
    sha256_file,
    validate_manifest as validate_source_manifest,
)
from controlled_cpu_replay_support_v1 import ENVIRONMENT_SEED, headless_kwargs


EXPERIMENT_ID = "teacher_native_spatial_fresh_seed_init8_cpu_reset_v3_20261013"
MANIFEST_PATH = Path(
    "/root/smolvla-eval-prep/recovery/teacher_native_spatial_fresh_seed_init8_cpu_reset_manifest_v3_20261013.json"
)
OUTPUT_ROOT = Path(
    "/root/smolvla-eval-prep/results/teacher_native_spatial_fresh_seed_init8_cpu_reset_v3_20261013"
)
TRACE_KEYS = (
    "single_task0_paired_8_r0_160k",
    "joint_task0_paired_8_r0_160k",
)
MAX_TOTAL_SECONDS = 300
WORKER_TIMEOUT_SECONDS = 120
GRACE_SECONDS = 10
REGISTRATION_MANIFEST_SHA = "d65063fe57c1554f25b8760437100f98c6c935f9775e0e9ecd55e33a0d928b88"


def _code_paths() -> dict[str, Path]:
    scripts = Path(__file__).resolve().parent
    return {
        "runner": Path(__file__).resolve(),
        "stage_replay": scripts / "replay_fresh_seed_cpu_stage_v1.py",
        "manifest_builder": scripts / "build_fresh_seed_stage_manifest_v1.py",
        "reset_helpers": scripts / "controlled_cpu_replay_support_v1.py",
        "render_provenance": scripts / "capture_reset_render_provenance_v1.py",
        "decision_provenance": scripts / "first_decision_provenance_v1.py",
        "child_supervisor": scripts / "bounded_child_signals_v4.py",
        "eval_schedule": scripts / "fresh_seed_eval_schedule_v1.py",
        "protocol_schedule": scripts / "scaling_eval_schedule_v1.py",
        "unit_tests": scripts.parent / "tests" / "test_fresh_seed_init8_cpu_reset_v3.py",
    }


def _code_hashes() -> dict[str, str]:
    paths = _code_paths()
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ReplayError(f"Required code/test inputs are missing or symlinked: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def canonical_sha(value: dict[str, Any], field: str) -> str:
    copied = dict(value)
    copied.pop(field, None)
    encoded = json.dumps(copied, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_trace_keys() -> tuple[str, str]:
    return TRACE_KEYS


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _force_cpu_only() -> None:
    # Workers use headless MuJoCo only; make CUDA device discovery impossible.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _source_rows() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data, rows = validate_source_manifest(SOURCE_MANIFEST)
    if data.get("manifest_sha256") != REGISTRATION_MANIFEST_SHA:
        raise ReplayError("Registered source manifest SHA-256 changed")
    by_trace = {trace["key"]: trace for trace in data["traces"]}
    by_row = {row["key"]: row for row in rows}
    if any(key not in by_trace or key not in by_row for key in TRACE_KEYS):
        raise ReplayError("Registered init-8 pair is missing from the pinned 40-trace source")
    return data, {key: {"trace": by_trace[key], "row": by_row[key]} for key in TRACE_KEYS}


def _load_run_manifest(path: Path) -> dict[str, Any]:
    if path.resolve() != MANIFEST_PATH:
        raise ReplayError("Manifest path differs from the unique registration")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment_id") != EXPERIMENT_ID:
        raise ReplayError("Experiment identity mismatch")
    if data.get("source_manifest_path") != str(SOURCE_MANIFEST):
        raise ReplayError("Pinned source manifest path mismatch")
    if data.get("source_manifest_sha256") != REGISTRATION_MANIFEST_SHA:
        raise ReplayError("Pinned source manifest identity mismatch")
    if tuple(data.get("trace_keys", ())) != TRACE_KEYS:
        raise ReplayError("Trace scope differs from the frozen init-8 pair")
    if data.get("gpu_budget") != 0 or data.get("policy_calls") != 0 or data.get("new_rollouts") != 0:
        raise ReplayError("The registered scope must remain CPU-only and policy-free")
    if canonical_sha(data, "manifest_sha256") != data.get("manifest_sha256"):
        raise ReplayError("Run-manifest self-hash mismatch")
    actual_code_hashes = _code_hashes()
    if actual_code_hashes != data.get("code_sha256"):
        raise ReplayError("Runner, helper, schedule, or unit-test SHA-256 differs from its manifest")
    if actual_code_hashes["runner"] != data.get("runner_sha256"):
        raise ReplayError("Runner SHA-256 differs from its manifest")
    _source_rows()
    return data


def _check_lock(lock_path: Path) -> int:
    if lock_path != SOURCE_GPU_LOCK or not lock_path.is_file() or lock_path.is_symlink():
        raise ReplayError("Expected existing regular shared GPU lock")
    fd = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ReplayError("GPU lock is not a regular file")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(fd)
        raise ReplayError("Shared GPU lock is already held") from error
    return fd


def worker(trace_key: str, output_root: Path, manifest_path: Path) -> dict[str, Any]:
    if trace_key not in TRACE_KEYS:
        raise ReplayError("Worker key is outside the exact registered two-trace scope")
    _load_run_manifest(manifest_path)
    if output_root.resolve() != OUTPUT_ROOT.resolve() or not output_root.is_dir():
        raise ReplayError("Worker output root differs from the unique registered output")
    gpu_before = require_gpu_idle()
    ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, timeout=10)
    conflict_tokens = ("replay_fresh_seed_cpu_stage_v1.py",
                       "teacher_native_spatial_task0_freshseed2001",
                       "teacher_native_spatial_4task_freshseed2001")
    for line in ps.splitlines()[1:]:
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != os.getpid() and any(
                token in fields[1] for token in conflict_tokens):
            raise ReplayError(f"Conflicting train/replay process detected: PID {fields[0]}")
    source_data, selected = _source_rows()
    item = selected[trace_key]

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    # Each worker is a new process; reset both global RNGs before suite/env construction.
    random.seed(ENVIRONMENT_SEED)
    np.random.seed(ENVIRONMENT_SEED)
    suite = benchmark.get_benchmark("libero_spatial")()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl.is_file():
        raise ReplayError("Official task-0 BDDL is missing")
    initial_states = suite.get_task_init_states(0)
    if initial_states is None or len(initial_states) < 20:
        raise ReplayError("Official task-0 benchmark initial-state list is incomplete")

    env = ControlEnv(**headless_kwargs(bddl))
    try:
        # replay_trace verifies model arrays and simulator state before first action.
        result = replay_trace(env, initial_states, item["trace"], item["row"])
        gpu_after = require_gpu_idle()
        result.update({
            "experiment_id": EXPERIMENT_ID,
            "fresh_worker_per_trace": True,
            "preconstruction_python_numpy_seed": ENVIRONMENT_SEED,
            "source_manifest_sha256": source_data["manifest_sha256"],
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "policy_calls": 0,
            "new_rollouts": 0,
            "training_updates": 0,
            "causal_attribution": False,
        })
        _write_exclusive(output_root / f"{trace_key}.json", result)
        return result
    finally:
        env.close()


def _build_run_manifest() -> dict[str, Any]:
    source_data, _ = _source_rows()
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "registered_init8_reset_lifecycle_smoke",
        "source_manifest_path": str(SOURCE_MANIFEST),
        "source_manifest_sha256": source_data["manifest_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "code_sha256": _code_hashes(),
        "trace_keys": list(TRACE_KEYS),
        "fresh_worker_per_trace": True,
        "preconstruction_python_numpy_seed": ENVIRONMENT_SEED,
        "gpu_lock": str(SOURCE_GPU_LOCK),
        "output_root": str(OUTPUT_ROOT),
        "gpu_budget": 0,
        "policy_calls": 0,
        "new_rollouts": 0,
        "training_updates": 0,
        "automatic_retries": 0,
        "maximum_walltime_seconds": MAX_TOTAL_SECONDS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "fold02": "LOCKED",
        "eight_task_expansion": "LOCKED",
        "historical_v3_invalidation": "unchanged",
    }


def build_manifest(output: Path) -> dict[str, Any]:
    if output.resolve() != MANIFEST_PATH.resolve() or output.exists() or output.is_symlink():
        raise FileExistsError("Unique manifest path is not absent or differs from registration")
    payload = _build_run_manifest()
    payload["manifest_sha256"] = canonical_sha(payload, "manifest_sha256")
    _write_exclusive(output, payload)
    return payload


def run(manifest_path: Path, output_root: Path, authorize_execution: bool) -> dict[str, Any]:
    if not authorize_execution:
        raise ReplayError("Explicit --authorize-execution is required")
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ReplayError("Output path differs from the unique registration")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing output path: {output_root}")
    registration = _load_run_manifest(manifest_path)
    _, selected = _source_rows()
    started = time.monotonic()
    completed: list[str] = []
    try:
        # The parent only probes lock availability. Each isolated worker acquires
        # and holds the shared lock while it verifies/replays its own trace.
        lock_fd = _check_lock(SOURCE_GPU_LOCK)
        os.close(lock_fd)
        gpu_before = require_gpu_idle()
        # Reject known conflicting trainers/replays even if their GPU is momentarily idle.
        ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, timeout=10)
        conflict_tokens = ("replay_fresh_seed_cpu_stage_v1.py",
                           "teacher_native_spatial_task0_freshseed2001", "teacher_native_spatial_4task_freshseed2001")
        for line in ps.splitlines()[1:]:
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and int(fields[0]) != os.getpid() and any(token in fields[1] for token in conflict_tokens):
                raise ReplayError(f"Conflicting train/replay process detected: PID {fields[0]}")

        output_root.mkdir(parents=False, exist_ok=False)
        run_record = dict(registration, status="running", gpu_before=gpu_before,
                          started_monotonic=started, completed_trace_keys=[])
        _write_exclusive(output_root / "manifest.json", run_record)
        for trace_key in TRACE_KEYS:
            if time.monotonic() - started > MAX_TOTAL_SECONDS:
                raise TimeoutError("Registered total walltime exhausted")
            require_gpu_idle()
            item = selected[trace_key]
            for name, text_path in item["trace"]["paths"].items():
                if sha256_file(Path(text_path)) != item["trace"]["sha256"].get(name):
                    raise ReplayError(f"Pinned source changed before worker start: {trace_key} {name}")
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker-key", trace_key, "--manifest", str(manifest_path),
                       "--results-dir", str(output_root)]
            with open(os.devnull, "w", encoding="utf-8") as sink:
                child = run_with_signal_cleanup(command, seconds=WORKER_TIMEOUT_SECONDS,
                                                grace_seconds=GRACE_SECONDS,
                                                cwd=str(Path(__file__).resolve().parent), stdout=sink)
            if child["timed_out"] or child["returncode"] != 0:
                failure_path = output_root / f"{trace_key}.worker_failure.json"
                failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else None
                raise ReplayError(f"Fresh worker failed; no retry: {trace_key} {child}; detail={failure}")
            result_path = output_root / f"{trace_key}.json"
            if not result_path.is_file():
                raise ReplayError(f"Worker result missing: {trace_key}")
            completed.append(trace_key)
            run_record["completed_trace_keys"] = list(completed)
            # The initial manifest is immutable; progress is in a separate exclusive record.
            _write_exclusive(output_root / f"progress_{len(completed)}.json", {
                "completed_trace_keys": list(completed), "worker": child,
                "last_trace": trace_key,
            })
        gpu_after = require_gpu_idle()
        completion = {
            "status": "completed_init8_method_smoke_not_stage_inference",
            "experiment_id": EXPERIMENT_ID,
            "source_manifest_sha256": registration["source_manifest_sha256"],
            "completed_trace_keys": list(completed),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "gpu_budget": 0,
            "policy_calls": 0,
            "new_rollouts": 0,
            "training_updates": 0,
            "automatic_retries": 0,
        }
        _write_exclusive(output_root / "completion.json", completion)
        _write_exclusive(output_root / "exit_code.json", {"status": "complete", "exit_code": 0})
        return completion
    except BaseException as error:
        if output_root.exists():
            if not (output_root / "failure.json").exists():
                _write_exclusive(output_root / "failure.json", {
                    "status": "stopped_partial_preserved",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_trace_keys": list(completed),
                    "automatic_retry": False,
                })
            if not (output_root / "exit_code.json").exists():
                _write_exclusive(output_root / "exit_code.json", {
                    "status": "error", "exit_code": 1,
                    "error_type": type(error).__name__,
                })
        raise
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--worker-key", choices=TRACE_KEYS)
    args = parser.parse_args()
    if args.worker_key or args.authorize_execution:
        _force_cpu_only()
    if args.build_manifest:
        result = build_manifest(args.manifest)
        print(json.dumps({"status": "hash_pinned", "experiment_id": EXPERIMENT_ID,
                          "manifest_sha256": result["manifest_sha256"],
                          "simulator_started": False, "gpu_used": False}, indent=2))
        return 0
    if args.worker_key:
        if not args.manifest.is_file() or not args.results_dir.is_dir():
            raise ReplayError("Worker requires the pinned manifest and existing output root")
        lock_fd = _check_lock(SOURCE_GPU_LOCK)
        try:
            try:
                worker(args.worker_key, args.results_dir, args.manifest)
            except BaseException as error:
                _write_exclusive(args.results_dir / f"{args.worker_key}.worker_failure.json", {
                    "status": "worker_error",
                    "trace_key": args.worker_key,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                raise
        finally:
            os.close(lock_fd)
        return 0
    if not hasattr(signal, "SIGALRM"):
        raise ReplayError("A total walltime alarm is required")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300-second smoke bound")))
    signal.alarm(MAX_TOTAL_SECONDS)
    try:
        result = run(args.manifest, args.results_dir, args.authorize_execution)
    finally:
        signal.alarm(0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
