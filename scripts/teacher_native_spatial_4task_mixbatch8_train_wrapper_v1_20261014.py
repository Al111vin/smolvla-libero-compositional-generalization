"""One-shot, exact-quota mixed-batch wrapper for the registered 4-task run."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from task_balanced_batch_sampler_v2 import install_task_balanced_dataloader_patch

CONFIG_PATH = Path(sys.argv[sys.argv.index("--config_path") + 1]) if "--config_path" in sys.argv else None
if CONFIG_PATH is None and any(arg.startswith("--config_path=") for arg in sys.argv):
    CONFIG_PATH = Path(next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--config_path=")))
if CONFIG_PATH is None or not CONFIG_PATH.is_file():
    raise SystemExit("missing --config_path; refusing to run trainer")

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
if (CONFIG["job_name"] != "teacher_native_spatial_4task_mixbatch8_freshseed2001_40k_v1_20261014"
        or CONFIG["seed"] != 2001 or CONFIG["steps"] != 40000
        or CONFIG["batch_size"] != 8 or CONFIG["dataset"]["repo_id"] != "local/libero_spatial_tasks0_3_native_v1"):
    raise SystemExit("config does not match the registered 4-task mixed-batch experiment")

install_task_balanced_dataloader_patch(
    target_repo_id="local/libero_spatial_tasks0_3_native_v1",
    seed=2001,
    samples_per_task=80000,
    samples_per_task_per_batch=2,
    expected_tasks=(0, 1, 2, 3),
    expected_batch_size=8,
)
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
