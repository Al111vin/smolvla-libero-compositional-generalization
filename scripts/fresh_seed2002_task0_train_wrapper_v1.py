"""Hash-pinned fresh task0 single-task seed-2002 training wrapper."""
import hashlib
import json
from pathlib import Path

CONFIG_NAME = "teacher_native_spatial_task0_freshseed2002_40k_batch2_v1_20261014.json"
CONFIG_SHA256 = "ae600eb68936841ffeb8fa24defe0fcbcd217dc7d8c6a6a508ece31f8d46133d"


def validate_registered_config(cfg):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    path = Path(__file__).resolve().parents[1] / "configs" / CONFIG_NAME
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("registered seed-2002 single config changed")
    expected = draccus.decode(TrainPipelineConfig, json.loads(raw))
    expected.validate()
    if cfg.to_dict() != expected.to_dict():
        raise ValueError("runtime config differs from complete registered single recipe")


def install(trainer):
    original = trainer.make_optimizer_and_scheduler

    def checked_factory(cfg, policy):
        validate_registered_config(cfg)
        if cfg.steps != 40000 or cfg.seed != 2002 or cfg.resume or cfg.env is not None:
            raise ValueError("unregistered seed-2002 single training")
        return original(cfg, policy)

    trainer.make_optimizer_and_scheduler = checked_factory
    return original


def main():
    import lerobot.scripts.lerobot_train as trainer
    install(trainer)
    trainer.train()


if __name__ == "__main__":
    main()
