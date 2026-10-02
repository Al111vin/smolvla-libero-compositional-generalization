#!/usr/bin/env python3
"""Create an isolated 10x-learning-rate task-0 config from the verified run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


RUN_NAME = "teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1"
SOURCE_NAME = "teacher_native_task0_base_init_current_recipe_10k_20261002_v1"
BASE = Path("/root/smolvla-training-prep")
REC = BASE / "recovery/native_task0_20261002"
LR = 1e-4
DECAY_LR = 2.5e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REC / f"{SOURCE_NAME}.json")
    parser.add_argument("--output", type=Path, default=REC / f"{RUN_NAME}.json")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output}")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)

    # One conceptual factor changes: scale the full cosine LR schedule by 10x.
    config["policy"]["optimizer_lr"] = LR
    config["policy"]["scheduler_decay_lr"] = DECAY_LR
    config["optimizer"]["lr"] = LR
    config["scheduler"]["peak_lr"] = LR
    config["scheduler"]["decay_lr"] = DECAY_LR
    config["output_dir"] = str(BASE / "results/training" / RUN_NAME)
    config["job_name"] = RUN_NAME

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_name": RUN_NAME,
        "source_run": SOURCE_NAME,
        "only_training_factor_changed": "cosine learning-rate scale x10",
        "peak_lr": LR,
        "decay_lr": DECAY_LR,
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "dataset": config["dataset"]["repo_id"],
        "pretrained_path": config["policy"]["pretrained_path"],
        "output_dir": config["output_dir"],
        "config_path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
