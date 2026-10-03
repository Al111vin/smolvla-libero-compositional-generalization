#!/usr/bin/env python3
"""Build and read-only validate a task-1-only config from the frozen 0-3 run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK1_TEXT = "pick up the black bowl next to the ramekin and place it on the plate"
EXPECTED_EVALUATOR_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.source_config, args.conversion_manifest, args.evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.config_out.exists() or args.preflight_out.exists():
        raise FileExistsError("Refusing to overwrite a config or preflight record")

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(source)
    task_id = 1
    selected_episodes = list(range(50, 100))
    run_name = "teacher_native_spatial_task1_single_current_recipe_40k_batch2_v1"
    expected_training_output = (
        "/root/smolvla-training-prep/results/training/" + run_name
    )
    expected_eval_output = (
        "/root/smolvla-eval-prep/results/" + run_name
    )

    if cfg["job_name"] != "teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2":
        raise ValueError("Unexpected source training config")
    if cfg["dataset"]["episodes"] != list(range(200)):
        raise ValueError("Source config no longer describes the frozen 200-episode set")
    if cfg["dataset"]["repo_id"] != "local/libero_spatial_tasks0_3_native_v1":
        raise ValueError("Unexpected dataset repo_id")
    if cfg["steps"] != 40000 or cfg["batch_size"] != 8:
        raise ValueError("Unexpected source update/batch settings")
    if cfg["policy"]["pretrained_path"] != "/root/smolvla-training-prep/models/smolvla_base_libero":
        raise ValueError("Unexpected model initialization")
    if cfg["optimizer"]["lr"] != 1e-4:
        raise ValueError("Unexpected optimizer learning rate")

    task_rows = manifest["tasks"]
    task_row = next(row for row in task_rows if row["task_id"] == task_id)
    if task_row["language"] != TASK1_TEXT or task_row["episodes"] != 50 or task_row["frames"] != 6707:
        raise ValueError("Task-1 manifest identity/counts differ from the registered values")

    episodes_path = (
        Path(cfg["dataset"]["root"])
        / "meta/episodes/chunk-000/file-000.parquet"
    )
    episode_rows = pq.read_table(episodes_path).to_pylist()
    chosen_rows = episode_rows[50:100]
    if len(chosen_rows) != 50:
        raise ValueError("Expected 50 task-1 episode rows")
    for expected_index, row in zip(selected_episodes, chosen_rows, strict=True):
        if row["episode_index"] != expected_index or row["tasks"] != [TASK1_TEXT]:
            raise ValueError(f"Episode/task mapping mismatch at episode {expected_index}")
    selected_frames = sum(row["length"] for row in chosen_rows)
    if selected_frames != 6707:
        raise ValueError(f"Unexpected task-1 frame count: {selected_frames}")

    cfg["dataset"]["episodes"] = selected_episodes
    cfg["job_name"] = run_name
    cfg["output_dir"] = expected_training_output
    cfg["steps"] = 40000
    cfg["batch_size"] = 2
    cfg["seed"] = 1000
    cfg["resume"] = False
    cfg["save_freq"] = 10000
    if cfg["policy"]["optimizer_lr"] != 1e-4:
        raise ValueError("Policy optimizer lr does not match the current 4-task run")
    if cfg["policy"]["chunk_size"] != 50 or cfg["policy"]["n_action_steps"] != 25:
        raise ValueError("Policy action horizon differs from the current 4-task run")

    for output in (Path(expected_training_output), Path(expected_eval_output)):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")

    model_root = Path(cfg["policy"]["pretrained_path"])
    if not model_root.is_dir():
        raise FileNotFoundError(f"Pretrained model directory missing: {model_root}")
    stats_path = Path(cfg["dataset"]["root"]) / "meta/stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Frozen normalization statistics missing: {stats_path}")

    process_lines = subprocess.run(
        ["bash", "-lc", "pgrep -af 'python.*(lerobot_train|task_balanced_train_wrapper|eval_v3_task0_state_capture)' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    processes = "\n".join(
        line for line in process_lines
        if line.split(maxsplit=1)[0].isdigit() and int(line.split(maxsplit=1)[0]) != os.getpid()
    )
    if processes:
        raise RuntimeError(f"Conflicting workload present: {processes}")
    gpu_processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if gpu_processes:
        raise RuntimeError(f"GPU compute process present: {gpu_processes}")
    free_gib = shutil.disk_usage("/root").free / (1024**3)
    if free_gib < 35:
        raise RuntimeError(f"Insufficient free disk space: {free_gib:.1f} GiB")

    evaluator_sha = sha256(args.evaluator)
    if evaluator_sha != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(f"Unexpected evaluator SHA256: {evaluator_sha}")

    dataset = LeRobotDataset(
        repo_id=cfg["dataset"]["repo_id"],
        root=cfg["dataset"]["root"],
        episodes=selected_episodes,
        video_backend=cfg["dataset"]["video_backend"],
    )
    if len(dataset) != 6707:
        raise ValueError(f"Filtered task-1 dataset length mismatch: {len(dataset)}")
    samples = [dataset[0], dataset[len(dataset) - 1]]
    for sample in samples:
        if sample["task"] != TASK1_TEXT or int(sample["task_index"]) != task_id:
            raise ValueError("Filtered loader returned a non-task-1 sample")
        if tuple(sample["observation.state"].shape) != (15,):
            raise ValueError("Unexpected state shape")
        if tuple(sample["action"].shape) != (7,):
            raise ValueError("Unexpected action shape")
        for key in ("observation.images.agentview", "observation.images.wrist"):
            if tuple(sample[key].shape) != (3, 128, 128):
                raise ValueError(f"Unexpected image shape for {key}")
        for key in ("observation.state", "action", "observation.images.agentview", "observation.images.wrist"):
            value = sample[key]
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite value in {key}")
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError(f"Non-finite value in {key}")

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = (json.dumps(cfg, indent=2, ensure_ascii=False) + "\n").encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    report = {
        "status": "passed",
        "training_started": False,
        "dataset_modified": False,
        "task_id": task_id,
        "task_text": TASK1_TEXT,
        "episode_indices": [50, 99],
        "episodes": 50,
        "frames": selected_frames,
        "filtered_dataset_len": len(dataset),
        "schema": {"image_shape": [3, 128, 128], "state_dim": 15, "action_dim": 7},
        "normalization_stats_source": "unchanged frozen four-task dataset meta/stats.json",
        "training": {
            "steps": cfg["steps"],
            "batch_size": cfg["batch_size"],
            "sample_draws": cfg["steps"] * cfg["batch_size"],
            "seed": cfg["seed"],
            "learning_rate": cfg["optimizer"]["lr"],
            "pretrained_path": cfg["policy"]["pretrained_path"],
        },
        "training_output": expected_training_output,
        "evaluation_output": expected_eval_output,
        "evaluator_sha256": evaluator_sha,
        "source_config_sha256": sha256(args.source_config),
        "conversion_manifest_sha256": sha256(args.conversion_manifest),
        "global_dataset_stats_sha256": sha256(Path(cfg["dataset"]["root"]) / "meta/stats.json"),
        "pretrained_model_present": True,
        "gpu_idle": True,
        "conflicting_workload_absent": True,
        "disk_free_gib": round(free_gib, 2),
        "new_config_sha256": config_hash,
    }
    write_new_json(args.config_out, cfg)
    write_new_json(args.preflight_out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
