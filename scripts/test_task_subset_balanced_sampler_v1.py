from __future__ import annotations

import unittest

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from task_subset_balanced_sampler_v1 import (
    TaskSubsetBalancedSampler,
    install_task_subset_balanced_dataloader_patch,
)


class _FakeArrowColumn:
    def __init__(self, values):
        self.values = values

    def to_pylist(self):
        return list(self.values)


class _FakeArrowTable:
    def __init__(self, values):
        self.values = values

    def column(self, name):
        assert name == "task_index"
        return _FakeArrowColumn(self.values)


class _FakeHFTable:
    column_names = ["task_index"]

    def __init__(self, values):
        self.data = _FakeArrowTable(values)


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, counts, repo_id="local/libero36_feasible_32_frozen_v1"):
        self.repo_id = repo_id
        self.tasks = [task for task, count in enumerate(counts) for _ in range(count)]
        self.hf_dataset = _FakeHFTable(self.tasks)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        return {"task_index": torch.tensor(self.tasks[index]), "index": index}


class TaskSubsetBalancedSamplerTests(unittest.TestCase):
    def test_four_task_batches_have_two_samples_per_task_and_exact_exposure(self):
        dataset = _FakeDataset([3, 5, 7, 9, 11])
        sampler = TaskSubsetBalancedSampler(
            dataset, task_ids=[0, 1, 2, 3], updates=10, batch_size=8, seed=42
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 80)
        for start in range(0, len(indices), 8):
            batch = [dataset.tasks[index] for index in indices[start : start + 8]]
            self.assertEqual({task: batch.count(task) for task in range(4)}, {0: 2, 1: 2, 2: 2, 3: 2})
        self.assertNotIn(4, [dataset.tasks[index] for index in indices])

    def test_single_task_batches_have_eight_samples_and_exposure_matches_four_task(self):
        dataset = _FakeDataset([3, 5, 7, 9, 11])
        single = list(TaskSubsetBalancedSampler(dataset, task_ids=[0], updates=10, batch_size=8, seed=5))
        four = list(TaskSubsetBalancedSampler(dataset, task_ids=[0, 1, 2, 3], updates=40, batch_size=8, seed=5))
        self.assertEqual(len(single), 80)
        self.assertEqual(sum(dataset.tasks[index] == 0 for index in four), 80)
        self.assertTrue(all(dataset.tasks[index] == 0 for index in single))
        for start in range(0, len(single), 8):
            self.assertEqual(len(single[start : start + 8]), 8)

    def test_sampler_is_deterministic_and_rejects_invalid_requests(self):
        dataset = _FakeDataset([3, 5, 7, 9])
        kwargs = dict(task_ids=[0, 1, 2, 3], updates=5, batch_size=8, seed=99)
        self.assertEqual(list(TaskSubsetBalancedSampler(dataset, **kwargs)), list(TaskSubsetBalancedSampler(dataset, **kwargs)))
        with self.assertRaises(ValueError):
            TaskSubsetBalancedSampler(dataset, task_ids=[0, 0], updates=5, batch_size=8)
        with self.assertRaises(ValueError):
            TaskSubsetBalancedSampler(dataset, task_ids=[0, 1, 2], updates=5, batch_size=8)

    def test_loader_patch_only_changes_target_repo_and_preserves_batch_contract(self):
        target = "local/libero36_feasible_32_frozen_v1"
        cls, original = install_task_subset_balanced_dataloader_patch(
            target_repo_id=target, task_ids=[0, 1, 2, 3], updates=3, batch_size=8, seed=11
        )
        try:
            target_loader = DataLoader(_FakeDataset([3, 5, 7, 9]), batch_size=8, shuffle=True)
            batches = list(target_loader)
            self.assertEqual(len(batches), 3)
            for batch in batches:
                counts = torch.bincount(batch["task_index"], minlength=4).tolist()
                self.assertEqual(counts, [2, 2, 2, 2])
            other_loader = DataLoader(_FakeDataset([3, 5, 7, 9], repo_id="other/repo"), batch_size=8)
            self.assertEqual(len(list(other_loader)), 3)
        finally:
            cls.__init__ = original

    def test_accelerate_rewrap_preserves_the_task_balanced_batch_sampler(self):
        target = "local/libero36_feasible_32_frozen_v1"
        cls, original = install_task_subset_balanced_dataloader_patch(
            target_repo_id=target, task_ids=[0, 1, 2, 3], updates=3, batch_size=8, seed=13
        )
        try:
            loader = DataLoader(_FakeDataset([3, 5, 7, 9]), batch_size=8, shuffle=True)
            prepared = Accelerator(cpu=True).prepare(loader)
            batches = list(prepared)
            self.assertEqual(len(batches), 3)
            for batch in batches:
                counts = torch.bincount(batch["task_index"], minlength=4).tolist()
                self.assertEqual(counts, [2, 2, 2, 2])
        finally:
            cls.__init__ = original


if __name__ == "__main__":
    unittest.main()
