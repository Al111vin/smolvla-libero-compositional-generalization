from __future__ import annotations

from collections import Counter

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from task_balanced_batch_sampler_v2 import (
    TaskBalancedBatchOrderSampler,
    install_task_balanced_dataloader_patch,
    task_counts_in_batch,
)


class FakeDataset:
    repo_id = "target"

    def __init__(self):
        self.items = [
            {"task_index": torch.tensor(task)}
            for task, size in enumerate([3, 5, 2, 7])
            for _ in range(size)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_sampler_emits_exact_two_per_task_per_batch_and_repeats_deterministically():
    dataset = FakeDataset()
    args = dict(
        seed=17,
        samples_per_task=8,
        samples_per_task_per_batch=2,
        expected_tasks=[0, 1, 2, 3],
        batch_size=8,
    )
    first = list(TaskBalancedBatchOrderSampler(dataset, **args))
    second = list(TaskBalancedBatchOrderSampler(dataset, **args))
    assert first == second
    assert len(first) == 32
    for start in range(0, len(first), 8):
        counts = Counter(
            int(dataset[i]["task_index"].item())
            for i in first[start : start + 8]
        )
        assert counts == {0: 2, 1: 2, 2: 2, 3: 2}


def test_dataloader_patch_preserves_exact_batch_composition():
    loader_cls, original_init = install_task_balanced_dataloader_patch(
        target_repo_id="target",
        seed=3,
        samples_per_task=8,
        samples_per_task_per_batch=2,
        expected_tasks=[0, 1, 2, 3],
        expected_batch_size=8,
    )
    try:
        loader = DataLoader(FakeDataset(), batch_size=8, shuffle=True)
        batches = list(loader)
        assert len(batches) == 4
        for batch in batches:
            assert task_counts_in_batch(batch) == {0: 2, 1: 2, 2: 2, 3: 2}
    finally:
        loader_cls.__init__ = original_init


def test_accelerate_rewrap_preserves_existing_batch_sampler():
    loader_cls, original_init = install_task_balanced_dataloader_patch(
        target_repo_id="target",
        seed=3,
        samples_per_task=8,
        samples_per_task_per_batch=2,
        expected_tasks=[0, 1, 2, 3],
        expected_batch_size=8,
    )
    try:
        trainer_loader = DataLoader(FakeDataset(), batch_size=8, shuffle=True)
        # Exercise the actual Accelerate wrapper, which reconstructs an inner
        # DataLoader using the existing batch_sampler and batch_size=1.
        prepared_loader = Accelerator(cpu=True).prepare(trainer_loader)
        batches = list(prepared_loader)
        assert len(batches) == 4
        for batch in batches:
            assert task_counts_in_batch(batch) == {0: 2, 1: 2, 2: 2, 3: 2}
    finally:
        loader_cls.__init__ = original_init


def test_sampler_rejects_wrong_task_ids_and_wrong_batch_size():
    dataset = FakeDataset()
    common = dict(
        seed=0,
        samples_per_task=8,
        samples_per_task_per_batch=2,
        batch_size=8,
    )
    with pytest.raises(ValueError, match="task ids"):
        TaskBalancedBatchOrderSampler(dataset, expected_tasks=[0, 1, 2], **common)
    with pytest.raises(ValueError, match="require 8"):
        TaskBalancedBatchOrderSampler(
            dataset,
            expected_tasks=[0, 1, 2, 3],
            batch_size=16,
            seed=0,
            samples_per_task=8,
            samples_per_task_per_batch=2,
        )
