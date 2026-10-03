"""Final read-only launch gate for the matched native Spatial 4-task run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from libero.libero import benchmark
from lerobot.datasets.lerobot_dataset import LeRobotDataset


EXPECTED_EVAL_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"
EXPECTED_DATASET = "local/libero_spatial_tasks0_3_native_v1"
EXPECTED_DATA_ROOT = Path("/root/smolvla-training-prep/datasets/lerobot/libero_spatial_tasks0_3_native_20261003_v1")
TRAINING_ROOT = Path("/root/smolvla-training-prep/results/training")
EVALUATION_ROOT = Path("/root/smolvla-eval-prep/results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--evaluator", type=Path, default=Path("/root/smolvla-eval-prep/scripts/eval_v3_task0_state_capture_v1.py"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_id = config["job_name"]
    expected_train_root = TRAINING_ROOT / run_id
    expected_eval_root = EVALUATION_ROOT / run_id
    manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert audit["status"] == "passed" and audit["all_state_action_image_checks_exact"] is True
    assert config["dataset"]["repo_id"] == EXPECTED_DATASET
    assert Path(config["dataset"]["root"]) == EXPECTED_DATA_ROOT
    assert Path(config["output_dir"]) == expected_train_root
    assert args.eval_root == expected_eval_root
    assert config["steps"] == 40000 and config["save_freq"] == 10000
    assert config["batch_size"] == 8 and config["seed"] == 1000
    assert config["dataset"]["episodes"] == list(range(200))
    assert config["policy"]["pretrained_path"] == "/root/smolvla-training-prep/models/smolvla_base_libero"
    assert config["policy"]["chunk_size"] == 50 and config["policy"]["n_action_steps"] == 25
    assert config["optimizer"]["lr"] == 1e-4
    assert config["scheduler"]["num_warmup_steps"] == 3000
    assert config["scheduler"]["num_decay_steps"] == 90000
    assert config["steps"] * 2 == 80000
    assert config["steps"] * config["batch_size"] == 320000

    dataset = LeRobotDataset(config["dataset"]["repo_id"], root=config["dataset"]["root"], download_videos=False)
    assert len(dataset) == manifest["frame_count"] == 22709
    assert dataset.num_episodes == manifest["episode_count"] == 200
    assert dataset.meta.total_tasks == 4
    expected_mapping = {"0": 0, "1": 1, "2": 2, "3": 3}
    mapping_by_instruction = dataset.meta.tasks["task_index"].to_dict()
    instruction_to_id = {
        row["language"]: int(row["task_id"])
        for row in manifest["tasks"]
    }
    observed_id_to_index = {
        str(instruction_to_id[instruction]): int(task_index)
        for instruction, task_index in mapping_by_instruction.items()
    }
    assert observed_id_to_index == expected_mapping == manifest["task_index_by_libero_id"]

    eval_hash = sha256(args.evaluator)
    assert eval_hash == EXPECTED_EVAL_SHA256
    model_root = Path(config["policy"]["pretrained_path"])
    assert model_root.is_dir()
    for expected_path in (expected_train_root, expected_eval_root, expected_eval_root / "summary.json"):
        if expected_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {expected_path}")

    disk = shutil.disk_usage("/root")
    free_gib = disk.free / (1024**3)
    assert free_gib >= 20.0, f"Insufficient free disk: {free_gib:.1f} GiB"
    suite = benchmark.get_benchmark("libero_spatial")()
    init_counts = {}
    for task_id in range(4):
        states = suite.get_task_init_states(task_id)
        count = 0 if states is None else len(states)
        assert count >= 20, f"Task {task_id} has only {count} benchmark initial states"
        init_counts[str(task_id)] = count

    processes = subprocess.run(
        ["bash", "-lc", "pgrep -af 'python.*(lerobot_train|task_balanced_train_wrapper|eval_v3_task0_state_capture)' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert not processes, f"Conflicting Python workload(s) found: {processes}"
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert not gpu, f"GPU compute process present: {gpu}"

    result = {
        "status": "passed",
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "dataset": EXPECTED_DATASET,
        "dataset_root": str(EXPECTED_DATA_ROOT),
        "dataset_episodes": dataset.num_episodes,
        "dataset_frames": len(dataset),
        "task_index_by_libero_id": observed_id_to_index,
        "benchmark_init_state_counts": init_counts,
        "optimizer_steps": 40000,
        "samples_per_task_per_batch": 2,
        "planned_sample_draws_per_task": 80000,
        "total_batch_sample_draws": 320000,
        "evaluator_sha256": eval_hash,
        "disk_free_gib": round(free_gib, 2),
        "training_output_and_eval_output_paths_absent": True,
        "run_id": run_id,
        "expected_training_output": str(expected_train_root),
        "expected_evaluation_output": str(expected_eval_root),
        "gpu_idle": True,
        "fold02": "LOCKED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
