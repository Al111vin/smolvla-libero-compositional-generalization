#!/usr/bin/env python3
"""No-gradient audit of a task-filtered balanced sampler on a LeRobot dataset."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import draccus
import torch
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401 - registers the policy choice for draccus
from scripts.lerobot_loco_episode_compat import install_episode_filter_compat
from task_subset_balanced_sampler_v1 import install_task_subset_balanced_dataloader_patch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--probe-batches", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output}")
    cfg = draccus.parse(TrainPipelineConfig, config_path=str(args.config), args=[])
    cfg.dataset.repo_id = args.repo_id
    cfg.dataset.root = args.dataset_root
    # The source positive-control config was built for a 50-episode dataset.
    # Clear that inherited episode subset so the task sampler sees the full
    # frozen 32-task source and alone controls which task rows are drawn.
    cfg.dataset.episodes = None
    cfg.steps = args.updates
    cfg.batch_size = args.batch_size
    cfg.output_dir = Path("/tmp/task_subset_sampler_preflight_no_training")
    cfg.validate()
    install_episode_filter_compat()
    dataset = make_dataset(cfg)
    if getattr(dataset, "repo_id", None) != args.repo_id:
        raise AssertionError(f"dataset repo_id mismatch: {getattr(dataset, 'repo_id', None)!r}")

    hf_data = dataset.hf_dataset.data
    task_values = [int(x) for x in hf_data.column("task_index").to_pylist()]
    episode_values = [int(x) for x in hf_data.column("episode_index").to_pylist()]
    frame_counts: collections.Counter[int] = collections.Counter(task_values)
    episodes_by_task: dict[int, set[int]] = collections.defaultdict(set)
    for task_id, episode_id in zip(task_values, episode_values):
        episodes_by_task[task_id].add(episode_id)
    missing = [task_id for task_id in args.task_ids if frame_counts[task_id] == 0]
    if missing:
        raise AssertionError(f"requested task ids missing from data: {missing}")

    cls, original_init = install_task_subset_balanced_dataloader_patch(
        target_repo_id=args.repo_id,
        task_ids=args.task_ids,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        # Exercise the same Accelerate DataLoader re-wrap used in training. The
        # sampler patch must preserve Accelerate's sampler wrapper rather than
        # treating it as an attempt to replace the task-balanced sampler.
        loader = Accelerator(cpu=True).prepare(loader)
        observed_batches = []
        observed_shapes = None
        for batch_index, batch in enumerate(loader):
            task_batch = [int(x) for x in batch["task_index"].reshape(-1).tolist()]
            counts = {task_id: task_batch.count(task_id) for task_id in args.task_ids}
            if len(task_batch) != args.batch_size:
                raise AssertionError(f"bad batch length at {batch_index}: {len(task_batch)}")
            expected = args.batch_size // len(args.task_ids)
            if any(count != expected for count in counts.values()):
                raise AssertionError(f"unbalanced batch {batch_index}: {counts}")
            for key in ("action", "observation.state"):
                if key not in batch or not torch.isfinite(batch[key]).all():
                    raise AssertionError(f"missing or non-finite {key}")
            if batch_index == 0:
                expected_shapes = {
                    "action": [args.batch_size, 50, 7],
                    "observation.state": [args.batch_size, 1, 15],
                    # LeRobot retains the observation-time axis for images;
                    # SmolVLA.prepare_images explicitly selects batch[:, -1]
                    # when the camera tensor is rank five.
                    "observation.images.agentview": [args.batch_size, 1, 3, 128, 128],
                    "observation.images.wrist": [args.batch_size, 1, 3, 128, 128],
                }
                observed_shapes = {key: list(batch[key].shape) for key in expected_shapes if key in batch}
                if observed_shapes != expected_shapes:
                    raise AssertionError(f"sample batch shape contract failed: {observed_shapes}")
            observed_batches.append(counts)
            if len(observed_batches) >= args.probe_batches:
                break
    finally:
        cls.__init__ = original_init

    if len(observed_batches) != args.probe_batches:
        raise AssertionError(f"only received {len(observed_batches)} batches")
    info_path = args.dataset_root / "meta/info.json"
    tasks_path = args.dataset_root / "meta/tasks.parquet"
    result = {
        "schema_version": 1,
        "passed": True,
        "training_started": False,
        "gradient_steps": 0,
        "dataset_repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "dataset_frames": int(dataset.num_frames),
        "dataset_episodes": int(dataset.num_episodes),
        "dataset_episode_filter": None,
        "requested_task_ids": args.task_ids,
        "frames_per_requested_task": {str(t): frame_counts[t] for t in args.task_ids},
        "episodes_per_requested_task": {str(t): len(episodes_by_task[t]) for t in args.task_ids},
        "optimizer_updates_planned": args.updates,
        "batch_size": args.batch_size,
        "expected_samples_per_task": args.updates * args.batch_size // len(args.task_ids),
        "per_batch_samples_per_task": args.batch_size // len(args.task_ids),
        "probe_batches": len(observed_batches),
        "all_probe_batches_balanced": True,
        "accelerate_dataloader_wrap_tested": True,
        "first_batch_shapes": observed_shapes,
        "metadata_sha256": {"info.json": sha256(info_path), "tasks.parquet": sha256(tasks_path)},
        "base_config": str(args.config),
        "config_overridden_only_for_no_write_dataset_and_sampler_probe": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
