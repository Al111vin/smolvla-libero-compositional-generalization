"""Training entry wrapper that activates the Stage E task-index sampler."""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

from task_subset_balanced_sampler_v1 import install_task_subset_balanced_dataloader_patch


def main() -> None:
    target_repo_id = os.environ.get("STAGEE_DATASET_REPO_ID", "local/libero36_feasible_32_frozen_v1")
    task_ids = [int(value) for value in os.environ["STAGEE_TASK_IDS"].split(",") if value]
    updates = int(os.environ["STAGEE_UPDATES"])
    batch_size = int(os.environ.get("STAGEE_BATCH_SIZE", "8"))
    seed = int(os.environ.get("STAGEE_SAMPLER_SEED", "1000"))

    config_path = None
    for index, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--config_path="):
            config_path = Path(arg.split("=", 1)[1])
        elif arg == "--config_path" and index + 2 <= len(sys.argv[1:]):
            config_path = Path(sys.argv[index + 2])
    if config_path is None or not config_path.is_file():
        raise SystemExit("missing --config_path; refusing to launch trainer")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"]["repo_id"] != target_repo_id:
        raise SystemExit("target repo id does not match training config")
    if int(config["steps"]) != updates or int(config["batch_size"]) != batch_size:
        raise SystemExit("sampler updates/batch size do not match training config")
    if config["dataset"].get("episodes") is not None:
        raise SystemExit("Stage E requires the full frozen dataset; episodes filter must be null")

    install_task_subset_balanced_dataloader_patch(
        target_repo_id=target_repo_id,
        task_ids=task_ids,
        updates=updates,
        batch_size=batch_size,
        seed=seed,
    )
    runpy.run_module("scripts.lerobot_train_loco_compat", run_name="__main__")


if __name__ == "__main__":
    main()
