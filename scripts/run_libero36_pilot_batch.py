#!/usr/bin/env python3
"""Sequential, resumable LIBERO-36 five-episode pilot batch runner."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import validate_libero_36_envs as reset_validator


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/demos/libero36_expansion_plan_v1.json"
UNIFIED = ROOT / "data/training/libero36_feasible_v1/manifest.json"
LOG_ROOT = ROOT / "logs/libero36_pilot_batch_v1"
COLLECTOR = ROOT / "scripts/collect_libero36_task0_success_pilot.py"
CONVERTER = ROOT / "scripts/convert_libero_hdf5_to_lerobot.py"
SEEDS_PER_TASK = 5
SETTLE_STEPS = 20
MAX_SEED_OFFSETS = 100


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def update_plan(plan: dict) -> None:
    completed = [x["task_id"] for x in plan["tasks"] if x.get("status") == "completed"]
    pending = [x["task_id"] for x in plan["tasks"] if x.get("status") not in {"completed", "failed"}]
    failed = [x["task_id"] for x in plan["tasks"] if x.get("status") == "failed"]
    plan.update({"completed_task_ids": completed, "pending_task_ids": pending,
                 "failed_task_ids": failed, "completed_count": len(completed),
                 "pending_count": len(pending), "failed_count": len(failed),
                 "last_updated_epoch": time.time()})
    atomic_json(PLAN, plan)


def update_unified(plan: dict) -> None:
    if UNIFIED.exists():
        unified = json.loads(UNIFIED.read_text(encoding="utf-8"))
    else:
        unified = {"protocol": "LIBERO-36 unified training manifest v1"}
    base = {0, 16}
    done = sorted(base | {x["task_id"] for x in plan["tasks"] if x.get("status") == "completed"})
    unified["included_task_ids"] = done
    unified["dataset_roots"] = [f"datasets/lerobot/libero36_task{tid}_success_pilot_v1" for tid in done]
    unified["status"] = f"incomplete_until_all_{32-len(done)}_remaining_tasks_are_collected" if len(done) < 32 else "complete_32_feasible_tasks"
    unified["do_not_train_as_full_32_task_set"] = len(done) < 32
    atomic_json(UNIFIED, unified)


def find_replay(task_id: int) -> Path:
    paths = sorted((ROOT / "results/libero36_gate5_official_v6").glob(f"task_{task_id:03d}_*/replay.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"task {task_id}: expected one formal replay, got {paths}")
    return paths[0]


def find_row(task_id: int) -> dict:
    rows = reset_validator.read_layout_spec(ROOT / "data/libero_36/layout_spec.csv")
    matches = [r for r in rows if int(r["task_id"]) == task_id and int(r["layout_id"]) == 1]
    if len(matches) != 1:
        raise RuntimeError(f"task {task_id}: expected one layout-1 row")
    return matches[0]


def search_success_seeds(task_id: int, log) -> list[int]:
    replay = find_replay(task_id)
    row = find_row(task_id)
    trace = np.load(replay, allow_pickle=False)
    actions = np.asarray(trace["actions"], dtype=np.float32)
    base_seed = int(np.asarray(trace["seed"]).item()) if "seed" in trace.files else 361000 + task_id * 10000
    env = reset_validator.make_environment(Path(row["bddl_path"]))
    selected: list[int] = []
    try:
        for offset in range(MAX_SEED_OFFSETS):
            seed = base_seed + offset
            reset_validator.seed_environment(env, seed)
            env.env.reset()
            for _ in range(SETTLE_STEPS):
                env.step(np.zeros(7, dtype=np.float32))
            success = False
            positive = 0
            try:
                success_trace = []
                for action in actions:
                    _, reward, _, _ = env.step(action)
                    positive += int(float(reward) > 0)
                    success_trace.append(bool(env.check_success()))
                success = bool(success_trace[-1] and np.asarray(success_trace[-20:], dtype=bool).all())
            finally:
                # The same environment is reset at the top of the next iteration.
                pass
            log.write(f"seed_search task={task_id} seed={seed} success={success} positive={positive}\n")
            log.flush()
            if success:
                selected.append(seed)
                if len(selected) == SEEDS_PER_TASK:
                    return selected
    finally:
        env.close()
    raise RuntimeError(f"task {task_id}: only found {len(selected)}/{SEEDS_PER_TASK} successful seeds")


def lerobot_qc(task_id: int, expected_frames: int) -> dict:
    root = ROOT / f"datasets/lerobot/libero36_task{task_id}_success_pilot_v1"
    info_path = root / "meta/info.json"
    failures: list[str] = []
    if not info_path.exists():
        return {"passed": False, "failures": ["missing_meta_info"], "episodes": 0, "frames": 0}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rows = 0
    episode_ids: set[int] = set()
    for parquet in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet)
        rows += table.num_rows
        for key, dim in (("action", 7), ("observation.state", 15)):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape[1:] != (dim,):
                failures.append(f"{parquet}:{key}:shape={values.shape}")
            if not np.isfinite(values).all():
                failures.append(f"{parquet}:{key}:nonfinite")
        episode_ids.update(np.asarray(table["episode_index"].to_numpy(zero_copy_only=False)).reshape(-1).tolist())
    if rows != expected_frames or info.get("total_frames") != expected_frames:
        failures.append(f"frame_count={rows}/{expected_frames}")
    if len(episode_ids) != SEEDS_PER_TASK or info.get("total_episodes") != SEEDS_PER_TASK:
        failures.append(f"episode_count={len(episode_ids)}")
    features = info.get("features", {})
    for camera in ("observation.images.agentview", "observation.images.wrist"):
        if camera not in features:
            failures.append(f"missing_feature={camera}")
    return {"passed": not failures, "failures": failures, "episodes": len(episode_ids),
            "frames": rows, "expected_frames": expected_frames, "path": str(root)}


def run_task(plan: dict, task: dict, log) -> None:
    task_id = int(task["task_id"])
    output_root = ROOT / f"data/demos/libero36_task{task_id}_5_success_pilot_v1"
    hdf5 = output_root / f"task_{task_id:03d}_layout_1_5_success.hdf5"
    lerobot = ROOT / f"datasets/lerobot/libero36_task{task_id}_success_pilot_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = search_success_seeds(task_id, log)
    log.write(f"selected_seeds={seeds}\n"); log.flush()
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "DISPLAY": ":100", "MUJOCO_GL": "glx",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    collector_cmd = [sys.executable, str(COLLECTOR), "--task-id", str(task_id),
                     "--seeds", *map(str, seeds), "--output-root", str(output_root), "--overwrite"]
    collected = subprocess.run(collector_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if collected.returncode != 0 or not hdf5.exists():
        raise RuntimeError(f"collector failed rc={collected.returncode}")
    hdf5_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if not hdf5_manifest.get("passed"):
        raise RuntimeError("HDF5 manifest QC failed")
    if lerobot.exists():
        shutil.rmtree(lerobot)
    convert_cmd = [sys.executable, str(CONVERTER), "--input", str(hdf5),
                   "--output", str(lerobot), "--repo-id", f"local/libero36_task{task_id}_success_pilot_v1", "--overwrite"]
    # The legacy converter exits nonzero after writing a valid five-episode set
    # because its final assertions still expect the old 50-episode dataset.
    subprocess.run(convert_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    expected_frames = int(hdf5_manifest["hdf5_qc"]["frames"])
    qc = lerobot_qc(task_id, expected_frames)
    (output_root / "lerobot_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    if not qc["passed"]:
        raise RuntimeError(f"LeRobot QC failed: {qc['failures']}")
    task.update({"status": "completed", "episodes": SEEDS_PER_TASK, "frames": expected_frames,
                 "seeds": seeds, "hdf5": str(hdf5.relative_to(ROOT)),
                 "lerobot": str(lerobot.relative_to(ROOT)),
                 "hdf5_qc": str((output_root / "manifest.json").relative_to(ROOT)),
                 "lerobot_qc": str((output_root / "lerobot_qc.json").relative_to(ROOT)),
                 "completed_at_epoch": time.time()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOG_ROOT / "batch.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another LIBERO-36 pilot batch is already running")
        master = LOG_ROOT / "batch.log"
        with master.open("a", encoding="utf-8") as log:
            plan = read_plan()
            update_plan(plan); update_unified(plan)
            candidates = [x for x in plan["tasks"] if x.get("status") != "completed" and (args.retry_failed or x.get("status") != "failed")]
            if args.max_tasks is not None:
                candidates = candidates[:args.max_tasks]
            log.write(f"batch_start candidates={[x['task_id'] for x in candidates]}\n"); log.flush()
            for task in candidates:
                task_id = int(task["task_id"])
                task_log_path = LOG_ROOT / f"task_{task_id:03d}.log"
                with task_log_path.open("a", encoding="utf-8") as task_log:
                    task_log.write(f"task_start epoch={time.time()}\n"); task_log.flush()
                    try:
                        run_task(plan, task, task_log)
                        task_log.write("task_status=completed\n")
                        log.write(f"task={task_id} status=completed\n")
                    except Exception as exc:
                        task["status"] = "failed"
                        task["failure"] = str(exc)
                        task["failed_at_epoch"] = time.time()
                        task_log.write(f"task_status=failed error={exc!r}\n")
                        log.write(f"task={task_id} status=failed error={exc!r}\n")
                    task_log.flush(); log.flush()
                    update_plan(plan); update_unified(plan)
            log.write("batch_end\n"); log.flush()


if __name__ == "__main__":
    main()
