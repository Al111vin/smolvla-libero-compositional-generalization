#!/usr/bin/env python3
"""Read-only preflight and isolated config builder for native Spatial task0."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK0_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
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
    parser.add_argument("--slice-audit", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.source_config, args.conversion_manifest, args.slice_audit, args.evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.config_out.exists() or args.preflight_out.exists():
        raise FileExistsError("Refusing to overwrite config or preflight report")

    audit = json.loads(args.slice_audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("frames_checked") != 5068 or audit.get("images_checked") != 10136:
        raise ValueError("The full task0 slice parity audit has not passed")
    if audit.get("state_max_abs_diff_float32") != 0 or audit.get("action_max_abs_diff_float32") != 0:
        raise ValueError("Task0 state/action conversion was not exact")

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(source)
    task_id = 0
    episodes = list(range(50))
    run_name = "teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1"
    training_output = f"/root/smolvla-training-prep/results/training/{run_name}"
    evaluation_output = f"/root/smolvla-eval-prep/results/{run_name}"

    if cfg["job_name"] != "teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2":
        raise ValueError("Unexpected source training config")
    if cfg["dataset"]["episodes"] != list(range(200)):
        raise ValueError("Source config does not describe the frozen 200-episode dataset")
    if cfg["dataset"]["repo_id"] != "local/libero_spatial_tasks0_3_native_v1":
        raise ValueError("Unexpected dataset repo id")
    if cfg["steps"] != 40000 or cfg["batch_size"] != 8 or cfg["seed"] != 1000:
        raise ValueError("Unexpected source step/batch/seed settings")
    if cfg["policy"]["pretrained_path"] != "/root/smolvla-training-prep/models/smolvla_base_libero":
        raise ValueError("Unexpected model initialization")
    if cfg["optimizer"]["lr"] != 1e-4 or cfg["policy"]["optimizer_lr"] != 1e-4:
        raise ValueError("Unexpected current recipe learning rate")
    if cfg["policy"]["chunk_size"] != 50 or cfg["policy"]["n_action_steps"] != 25:
        raise ValueError("Unexpected action chunk configuration")

    task_row = next(row for row in manifest["tasks"] if row["task_id"] == task_id)
    if task_row["language"] != TASK0_TEXT or task_row["episodes"] != 50 or task_row["frames"] != 5068:
        raise ValueError("Task0 source manifest identity/counts mismatch")
    if task_row["sha256"] != audit["source_sha256"]:
        raise ValueError("Task0 HDF5 manifest hash and full slice audit disagree")

    dataset_root = Path(cfg["dataset"]["root"])
    episode_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    selected_rows = episode_table[:50]
    if len(selected_rows) != 50:
        raise ValueError("Expected 50 task0 episode metadata rows")
    for expected_index, row in zip(episodes, selected_rows, strict=True):
        if row["episode_index"] != expected_index or row["tasks"] != [TASK0_TEXT]:
            raise ValueError(f"Episode/task mapping mismatch at {expected_index}")
    selected_frames = sum(int(row["length"]) for row in selected_rows)
    if selected_frames != 5068:
        raise ValueError(f"Unexpected task0 frame count: {selected_frames}")

    cfg["dataset"]["episodes"] = episodes
    cfg["job_name"] = run_name
    cfg["output_dir"] = training_output
    cfg["steps"] = 40000
    cfg["batch_size"] = 2
    cfg["seed"] = 1000
    cfg["resume"] = False
    cfg["save_freq"] = 10000

    for output in (Path(training_output), Path(evaluation_output)):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    model_root = Path(cfg["policy"]["pretrained_path"])
    stats_path = dataset_root / "meta/stats.json"
    if not model_root.is_dir() or not stats_path.is_file():
        raise FileNotFoundError("Base model or frozen pooled normalization stats are missing")
    evaluator_sha = sha256(args.evaluator)
    if evaluator_sha != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(f"Unexpected evaluator SHA256: {evaluator_sha}")

    import_spec = importlib.util.find_spec("scripts.lerobot_train_loco_compat")
    if import_spec is None:
        raise ImportError("Current training module scripts.lerobot_train_loco_compat is unavailable")

    process_lines = subprocess.run(
        ["bash", "-lc", "pgrep -af 'python.*(lerobot_train|task_balanced_train_wrapper|eval_v3_task0_state_capture)' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    process_lines = [
        line for line in process_lines
        if line.split(maxsplit=1)[0].isdigit() and int(line.split(maxsplit=1)[0]) != os.getpid()
    ]
    if process_lines:
        raise RuntimeError("Conflicting workload present: " + "\n".join(process_lines))
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
        raise RuntimeError(f"Insufficient disk space: {free_gib:.1f} GiB")

    dataset = LeRobotDataset(
        repo_id=cfg["dataset"]["repo_id"],
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / 20 for i in range(50)]},
        video_backend=cfg["dataset"]["video_backend"],
    )
    if len(dataset) != 5068 or dataset.num_episodes != 50:
        raise ValueError(f"Filtered dataset size mismatch: {len(dataset)} frames, {dataset.num_episodes} episodes")
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        if sample["task"] != TASK0_TEXT or int(sample["task_index"]) != 0:
            raise ValueError(f"Filtered loader returned wrong task at sample {index}")
        expected_shapes = {
            "action": (50, 7),
            "observation.state": (15,),
            "observation.images.agentview": (3, 128, 128),
            "observation.images.wrist": (3, 128, 128),
        }
        for key, shape in expected_shapes.items():
            value = sample[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"Unexpected {key} shape: {tuple(value.shape)}")
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite sample in {key}")
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError(f"Non-finite sample in {key}")

    config_payload = (json.dumps(cfg, indent=2, ensure_ascii=False) + "\n").encode()
    config_hash = hashlib.sha256(config_payload).hexdigest()
    report = {
        "status": "passed",
        "training_started": False,
        "dataset_modified": False,
        "task_id": 0,
        "task_text": TASK0_TEXT,
        "episode_indices": [0, 49],
        "episodes": 50,
        "frames": selected_frames,
        "filtered_dataset_len": len(dataset),
        "schema": {"image_shape": [3, 128, 128], "chunked_action_shape": [50, 7], "state_dim": 15},
        "source_manifest_sha256": sha256(args.conversion_manifest),
        "task0_slice_audit_sha256": sha256(args.slice_audit),
        "global_dataset_stats_sha256": sha256(stats_path),
        "normalization_stats_source": "frozen pooled four-task dataset meta/stats.json, unchanged",
        "training": {"steps": 40000, "batch_size": 2, "sample_draws": 80000, "seed": 1000, "lr": 1e-4},
        "training_output": training_output,
        "evaluation_output": evaluation_output,
        "evaluator_sha256": evaluator_sha,
        "training_module": str(import_spec.origin),
        "gpu_idle": True,
        "conflicting_workload_absent": True,
        "disk_free_gib": round(free_gib, 2),
        "new_config_sha256": config_hash,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.config_out, cfg)
    write_new_json(args.preflight_out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
