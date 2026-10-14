from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_teacher_native_spatial_4task_mixbatch8_freshseed2001_config_v1.py"
SAMPLER_PATH = ROOT / "scripts/task_balanced_batch_sampler_v2.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticDataset:
    repo_id = "local/libero_spatial_tasks0_3_native_v1"

    def __init__(self):
        self.rows = [task for task, n in enumerate((7, 9, 11, 13)) for _ in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"task_index": torch.tensor(self.rows[index])}


def test_config_changes_only_registered_update_composition_fields():
    builder = load_module("registered_config_builder", BUILDER_PATH)
    single = json.loads(builder.SINGLE.read_text())
    old = json.loads(builder.JOINT.read_text())
    new = builder.build_config()
    allowed = {"steps", "batch_size", "job_name", "output_dir"}
    changed = {key for key in old if old[key] != new[key]}
    assert changed == allowed
    assert new["scheduler"] == single["scheduler"]
    assert new["optimizer"] == single["optimizer"]
    assert new["steps"] == 40000 and new["batch_size"] == 8 and new["seed"] == 2001
    assert len(new["dataset"]["episodes"]) == 200


def test_exact_two_samples_per_task_each_batch_and_80k_draw_budget():
    sampler = load_module("pinned_task_balanced_sampler", SAMPLER_PATH)
    dataset = SyntheticDataset()
    args = dict(seed=2001, samples_per_task=80000,
                samples_per_task_per_batch=2, expected_tasks=(0, 1, 2, 3),
                batch_size=8)
    first = list(sampler.TaskBalancedBatchOrderSampler(dataset, **args))
    second = list(sampler.TaskBalancedBatchOrderSampler(dataset, **args))
    assert first == second
    assert len(first) == 320000
    totals = Counter(dataset.rows[index] for index in first)
    assert totals == {0: 80000, 1: 80000, 2: 80000, 3: 80000}
    for start in range(0, len(first), 8):
        counts = Counter(dataset.rows[index] for index in first[start:start + 8])
        assert counts == {0: 2, 1: 2, 2: 2, 3: 2}
