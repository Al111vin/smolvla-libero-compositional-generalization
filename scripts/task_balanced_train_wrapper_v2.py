"""Opt-in launcher for native LIBERO Spatial tasks 0-3 balanced training."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_balanced_batch_sampler_v2 import install_task_balanced_dataloader_patch

install_task_balanced_dataloader_patch(
    target_repo_id="local/libero_spatial_tasks0_3_native_v1",
    seed=1000,
    samples_per_task=80000,
    samples_per_task_per_batch=2,
    expected_tasks=(0, 1, 2, 3),
    expected_batch_size=8,
)

runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
