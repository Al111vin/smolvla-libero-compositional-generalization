#!/usr/bin/env python3
"""Create an isolated exposure-matched four-task config from the passed run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/root/smolvla-training-prep/recovery/native_task0_20261002/"
    "teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1.json"
)
DEFAULT_RUN_NAME = "teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v1"
DATASET_REPO = "local/libero_spatial_tasks0_3_native_v1"
DATASET_ROOT = Path(
    "/root/smolvla-training-prep/datasets/lerobot/"
    "libero_spatial_tasks0_3_native_20261003_v1"
)
TRAINING_ROOT = Path("/root/smolvla-training-prep/results/training")
RECOVERY_ROOT = Path("/root/smolvla-training-prep/recovery/task_scaling_native_spatial_20261003")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-id", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_name = args.run_id
    output_root = TRAINING_ROOT / run_name
    output_path = args.output or RECOVERY_ROOT / f"{run_name}.json"
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if output_path.exists() or output_root.exists():
        raise FileExistsError("Refusing to overwrite config or training output")

    source = json.loads(args.source.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    config["job_name"] = run_name
    config["output_dir"] = str(output_root)
    config["dataset"]["repo_id"] = DATASET_REPO
    config["dataset"]["root"] = str(DATASET_ROOT)
    config["dataset"]["episodes"] = list(range(200))
    config["steps"] = 40000
    config["save_freq"] = 10000

    # Guard every control variable other than task count, task-balanced sampling,
    # and the 4x update count required for equal per-task sample exposure.
    assert config["policy"]["pretrained_path"] == "/root/smolvla-training-prep/models/smolvla_base_libero"
    assert config["batch_size"] == 8 and config["seed"] == 1000
    assert config["policy"]["chunk_size"] == 50
    assert config["policy"]["n_action_steps"] == 25
    assert config["policy"]["optimizer_lr"] == 1e-4
    assert config["optimizer"]["lr"] == 1e-4
    assert config["scheduler"]["peak_lr"] == 1e-4
    assert config["scheduler"]["decay_lr"] == 2.5e-6
    assert config["scheduler"]["num_warmup_steps"] == 3000
    assert config["scheduler"]["num_decay_steps"] == 90000
    assert config["dataset"]["repo_id"] == DATASET_REPO
    assert config["dataset"]["episodes"] == list(range(200))
    assert config["steps"] * 2 == 80000

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_name,
        "config_path": str(output_path),
        "output_dir": str(output_root),
        "dataset_repo_id": DATASET_REPO,
        "episodes": 200,
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "samples_per_task_per_batch": 2,
        "per_task_sample_draws": 80000,
        "lr": config["optimizer"]["lr"],
        "warmup_steps": config["scheduler"]["num_warmup_steps"],
        "decay_steps": config["scheduler"]["num_decay_steps"],
        "save_freq": config["save_freq"],
    }, indent=2))


if __name__ == "__main__":
    main()
