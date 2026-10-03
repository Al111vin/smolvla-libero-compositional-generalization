"""Deterministic task-balanced sampling with exact per-batch task quotas.

Unlike the v1 sampler, which emitted contiguous task blocks and therefore
could form homogeneous DataLoader batches, this sampler emits each batch as
an independent quota-balanced block. It remains opt-in and does not patch
LeRobot unless ``install_task_balanced_dataloader_patch`` is called.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class TaskBalancedBatchOrderSampler(Sampler[int]):
    """Yield indices whose consecutive DataLoader batches meet exact quotas."""

    def __init__(
        self,
        dataset,
        *,
        seed: int,
        samples_per_task: int,
        samples_per_task_per_batch: int,
        expected_tasks: Sequence[int] | None = None,
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.samples_per_task = int(samples_per_task)
        self.samples_per_task_per_batch = int(samples_per_task_per_batch)
        self.batch_size = int(batch_size)
        if min(
            self.samples_per_task,
            self.samples_per_task_per_batch,
            self.batch_size,
        ) <= 0:
            raise ValueError("sampling sizes and batch_size must be positive")
        if self.samples_per_task % self.samples_per_task_per_batch:
            raise ValueError(
                "samples_per_task must be divisible by "
                "samples_per_task_per_batch"
            )

        by_task: dict[int, list[int]] = defaultdict(list)
        if (
            hasattr(dataset, "hf_dataset")
            and "task_index" in dataset.hf_dataset.column_names
        ):
            task_values = (
                dataset.hf_dataset.data.column("task_index").to_pylist()
            )
            for index, value in enumerate(task_values):
                by_task[int(value)].append(index)
        else:
            for index in range(len(dataset)):
                value = dataset[index]["task_index"]
                if hasattr(value, "item"):
                    value = value.item()
                by_task[int(value)].append(index)

        actual_tasks = tuple(sorted(by_task))
        if not actual_tasks:
            raise ValueError("dataset has no task_index values")
        if expected_tasks is not None:
            expected = tuple(sorted(int(task) for task in expected_tasks))
            if actual_tasks != expected:
                raise ValueError(
                    f"dataset task ids {actual_tasks} do not match expected "
                    f"{expected}"
                )
        quota_total = len(actual_tasks) * self.samples_per_task_per_batch
        if self.batch_size != quota_total:
            raise ValueError(
                f"batch_size={self.batch_size}, but exact task quotas require "
                f"{quota_total} ({len(actual_tasks)} tasks x "
                f"{self.samples_per_task_per_batch})"
            )

        self.tasks = actual_tasks
        self.indices_by_task = {
            task: by_task[task] for task in self.tasks
        }
        self.num_batches = (
            self.samples_per_task // self.samples_per_task_per_batch
        )

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        orders: dict[int, list[int]] = {}
        cursors = {task: 0 for task in self.tasks}
        for task in self.tasks:
            indices = self.indices_by_task[task]
            orders[task] = [
                indices[i]
                for i in torch.randperm(len(indices), generator=generator)
                .tolist()
            ]

        def next_index(task: int) -> int:
            if cursors[task] >= len(orders[task]):
                indices = self.indices_by_task[task]
                orders[task] = [
                    indices[i]
                    for i in torch.randperm(len(indices), generator=generator)
                    .tolist()
                ]
                cursors[task] = 0
            index = orders[task][cursors[task]]
            cursors[task] += 1
            return index

        for _ in range(self.num_batches):
            task_order = [
                self.tasks[i]
                for i in torch.randperm(len(self.tasks), generator=generator)
                .tolist()
            ]
            for task in task_order:
                for _ in range(self.samples_per_task_per_batch):
                    yield next_index(task)

    def __len__(self) -> int:
        return len(self.tasks) * self.samples_per_task


def install_task_balanced_dataloader_patch(
    *,
    target_repo_id: str,
    seed: int,
    samples_per_task: int,
    samples_per_task_per_batch: int,
    expected_tasks: Sequence[int],
    expected_batch_size: int,
):
    """Patch only DataLoaders for one repo and verify batch settings exactly."""
    from torch.utils import data as torch_data

    loader_cls = torch_data.DataLoader
    original_init = loader_cls.__init__

    def patched_init(self, dataset, *args, **kwargs):
        if getattr(dataset, "repo_id", None) == target_repo_id:
            # Accelerate wraps an already-created DataLoader in a
            # DataLoaderShard. Its constructor rebuilds an inner DataLoader
            # with batch_size=1 plus the existing batch_sampler. That is not
            # a new trainer loader: preserve the sampler and let PyTorch build
            # the wrapper without applying the task quota twice.
            if kwargs.get("batch_sampler") is not None:
                original_init(self, dataset, *args, **kwargs)
                return
            if args:
                raise ValueError(
                    "balanced loader requires DataLoader optional arguments "
                    "as keywords so its batch_size can be verified"
                )
            configured_batch_size = kwargs.get("batch_size", 1)
            if configured_batch_size != expected_batch_size:
                raise ValueError(
                    f"target dataset DataLoader batch_size is "
                    f"{configured_batch_size}; expected "
                    f"{expected_batch_size}"
                )
            if kwargs.get("sampler") is not None:
                raise ValueError("target DataLoader already has a sampler")
            if kwargs.get("batch_sampler") is not None:
                raise ValueError("target DataLoader already has a batch_sampler")
            sampler = TaskBalancedBatchOrderSampler(
                dataset,
                seed=seed,
                samples_per_task=samples_per_task,
                samples_per_task_per_batch=samples_per_task_per_batch,
                expected_tasks=expected_tasks,
                batch_size=configured_batch_size,
            )
            kwargs["sampler"] = sampler
            kwargs["shuffle"] = False
        original_init(self, dataset, *args, **kwargs)

    loader_cls.__init__ = patched_init
    return loader_cls, original_init


def task_counts_in_batch(batch) -> Counter[int]:
    """Helper used by smoke tests and runtime assertions."""
    values = batch["task_index"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    return Counter(int(value) for value in values)
