"""Hash-pinned entrypoint for the single authorized seed-2001 task1 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUN_ID = "teacher_native_spatial_task1_freshseed2001_40k_batch2_v1_20261014"
CONFIG_SHA256 = "150269259fbc2711a6aacb3beb8f5a20abc24f744ffac08dc988bf16d62446de"


def validate_registered_config(cfg) -> None:
    import draccus
    from lerobot.configs.train import TrainPipelineConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{RUN_ID}.json"
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("registered task1 seed-2001 config hash mismatch")
    expected = draccus.decode(TrainPipelineConfig, json.loads(raw))
    expected.validate()
    if cfg.to_dict() != expected.to_dict():
        raise ValueError("runtime config differs from registered task1 config")
    if (cfg.job_name != RUN_ID or cfg.steps != 40000 or cfg.seed != 2001
            or cfg.batch_size != 2 or cfg.resume or cfg.env is not None
            or cfg.dataset.episodes != list(range(50, 100))):
        raise ValueError("runtime config violates task1 run registration")


def install(trainer):
    original = trainer.make_optimizer_and_scheduler

    def checked_factory(cfg, policy):
        validate_registered_config(cfg)
        return original(cfg, policy)

    trainer.make_optimizer_and_scheduler = checked_factory
    return original


def main() -> None:
    import lerobot.scripts.lerobot_train as trainer

    install(trainer)
    trainer.train()


if __name__ == "__main__":
    main()
