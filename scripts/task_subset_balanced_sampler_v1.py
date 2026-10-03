"""Deterministic task-subset sampler for exposure-matched LIBERO scaling runs.

The sampler emits indices in complete optimizer-batch groups. With batch size
8, a one-task run receives 8 examples from that task per update; a four-task
run receives exactly 2 examples from each task per update. It does not modify
the dataset or its preprocessing and is opt-in through the loader patch.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def _task_index_values(dataset) -> list[int]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and "task_index" in getattr(hf_dataset, "column_names", ()):
        return [int(value) for value in hf_dataset.data.column("task_index").to_pylist()]
    values = []
    for index in range(len(dataset)):
        value = dataset[index]["task_index"]
        if isinstance(value, torch.Tensor):
            value = value.item()
        values.append(int(value))
    return values


class TaskSubsetBalancedSampler(Sampler[int]):
    """Emit a fixed number of deterministic, task-balanced optimizer batches."""

    def __init__(
        self,
        dataset,
        *,
        task_ids: Sequence[int],
        updates: int,
        batch_size: int = 8,
        seed: int = 1000,
    ) -> None:
        self.task_ids = tuple(int(task_id) for task_id in task_ids)
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be a non-empty sequence without duplicates")
        if updates <= 0 or batch_size <= 0 or batch_size % len(self.task_ids):
            raise ValueError("updates/batch_size must be positive and batch_size divisible by task count")
        self.updates = int(updates)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.per_task_per_batch = self.batch_size // len(self.task_ids)

        by_task: dict[int, list[int]] = defaultdict(list)
        values = _task_index_values(dataset)
        if len(values) != len(dataset):
            raise ValueError("task_index column length does not match dataset length")
        for index, task_id in enumerate(values):
            if task_id in self.task_ids:
                by_task[task_id].append(index)
        missing = [task_id for task_id in self.task_ids if not by_task[task_id]]
        if missing:
            raise ValueError(f"requested task ids have no frames: {missing}")
        self.indices_by_task = {task_id: by_task[task_id] for task_id in self.task_ids}

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        orders: dict[int, list[int]] = {}
        cursors = {task_id: 0 for task_id in self.task_ids}

        def reshuffle(task_id: int) -> None:
            indices = self.indices_by_task[task_id]
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            orders[task_id] = [indices[position] for position in permutation]
            cursors[task_id] = 0

        for task_id in self.task_ids:
            reshuffle(task_id)

        for _ in range(self.updates):
            for task_id in self.task_ids:
                for _ in range(self.per_task_per_batch):
                    if cursors[task_id] >= len(orders[task_id]):
                        reshuffle(task_id)
                    yield orders[task_id][cursors[task_id]]
                    cursors[task_id] += 1

    def __len__(self) -> int:
        return self.updates * self.batch_size


def install_task_subset_balanced_dataloader_patch(
    *,
    target_repo_id: str,
    task_ids: Sequence[int],
    updates: int,
    batch_size: int = 8,
    seed: int = 1000,
):
    """Patch only the requested LeRobot repo id; return the original loader init."""
    from torch.utils import data as torch_data

    loader_cls = torch_data.DataLoader
    original_init = loader_cls.__init__

    def patched_init(self, dataset, *args, **kwargs):
        if getattr(dataset, "repo_id", None) != target_repo_id:
            return original_init(self, dataset, *args, **kwargs)
        # Accelerate re-wraps an already-constructed DataLoader by creating a
        # DataLoaderShard with a sampler/batch_sampler that wraps the original
        # loader's batch sampler. Preserve that wrapper unchanged; only install
        # our task sampler on the first, ordinary DataLoader construction.
        if kwargs.get("sampler") is not None or kwargs.get("batch_sampler") is not None:
            return original_init(self, dataset, *args, **kwargs)
        if len(args) > 1:
            raise RuntimeError("target DataLoader uses unsupported positional options; inspect loader call before training")
        configured_batch_size = kwargs.get("batch_size", args[0] if args else 1)
        if int(configured_batch_size) != int(batch_size):
            raise RuntimeError(
                f"configured batch_size={configured_batch_size} does not match sampler batch_size={batch_size}"
            )
        sampler = TaskSubsetBalancedSampler(
            dataset,
            task_ids=task_ids,
            updates=updates,
            batch_size=batch_size,
            seed=seed,
        )
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
        return original_init(self, dataset, *args, **kwargs)

    loader_cls.__init__ = patched_init
    return loader_cls, original_init
