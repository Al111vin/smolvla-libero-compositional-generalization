#!/usr/bin/env python3
"""Prepare a non-overwriting single-task config on the frozen LIBERO-36 source."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output}")
    config = copy.deepcopy(json.loads(args.source.read_text(encoding="utf-8")))
    config["dataset"]["repo_id"] = args.repo_id
    config["dataset"]["root"] = str(args.dataset_root)
    # The source single-task config limits episodes to 0..49. Remove that
    # inherited cap; the opt-in sampler alone restricts training draws to task 0.
    config["dataset"]["episodes"] = None
    config["steps"] = args.updates
    config["batch_size"] = 8
    config["output_dir"] = str(args.training_output)
    config["job_name"] = args.run_name

    if args.training_output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.training_output}")
    if config["batch_size"] != 8 or args.updates * 8 // len(args.task_ids) != 80000:
        raise AssertionError("each requested task must receive exactly 80000 sample draws")
    if config["policy"]["chunk_size"] != 50 or config["policy"]["n_action_steps"] != 25:
        raise AssertionError("Stage E1 action-chunk settings changed unexpectedly")
    if config["policy"]["optimizer_lr"] != 1e-4 or config["optimizer"]["lr"] != 1e-4:
        raise AssertionError("Stage E1 must reuse the previously tested 10x LR scale")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_name": args.run_name,
        "source_config": str(args.source),
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "episodes_filter": None,
        "selected_task_ids": args.task_ids,
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "expected_samples_per_task": args.updates * 8 // len(args.task_ids),
        "peak_lr": config["policy"]["optimizer_lr"],
        "output_dir": config["output_dir"],
        "config_path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
