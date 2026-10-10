"""CPU optimizer restore v2: scheduler constructed first; no parameter updates."""
import argparse
import json
import sys
import draccus
import torch
from pathlib import Path
import lerobot.scripts.lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.optim.optimizers import load_optimizer_state
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    pretrained = args.checkpoint / "pretrained_model"
    state_dir = args.checkpoint / "training_state"
    cfg = draccus.decode(TrainPipelineConfig, json.loads((pretrained / "train_config.json").read_text()))
    config = PreTrainedConfig.from_pretrained(pretrained)
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=config,
                                         local_files_only=True, strict=True)
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    if cfg.steps == 40000:
        scheduler = cfg.scheduler.build(optimizer, 40000)
    elif cfg.steps == 160000:
        sys.path.insert(0, "/root/smolvla-training-prep/recovery/exposure_matched_4task_v1_20261009/scripts")
        from exposure_matched_blocks_v1 import build_block_lambda_scheduler
        scheduler = build_block_lambda_scheduler(optimizer, cfg.scheduler, blocks=40000, task_count=4)
    else:
        raise ValueError("unregistered schedule")
    load_optimizer_state(optimizer, state_dir)
    saved_schedule = json.loads((state_dir / "scheduler_state.json").read_text())
    scheduler.load_state_dict(saved_schedule)
    restored = optimizer.state_dict()
    groups = json.loads((state_dir / "optimizer_param_groups.json").read_text())
    assert len(restored["param_groups"]) == len(groups)
    for actual, saved, lr in zip(restored["param_groups"], groups, scheduler.get_last_lr()):
        assert actual["params"] == saved["params"]
        assert actual["lr"] == saved["lr"] == lr
    checked = 0
    with safe_open(state_dir / "optimizer_state.safetensors", framework="pt", device="cpu") as source:
        expected_ids = {int(key.split("/")[1]) for key in source.keys()}
        assert set(restored["state"]) == expected_ids
        for key in source.keys():
            _, identifier, field = key.split("/")
            actual = restored["state"][int(identifier)][field]
            assert torch.equal(actual, source.get_tensor(key)), key
            assert torch.isfinite(actual).all().item(), key
            checked += 1
    print(json.dumps({"status": "actual_cpu_optimizer_restore_passed",
                      "state_tensors_equal_and_finite": checked, "active_states": len(expected_ids),
                      "scheduler_last_epoch": scheduler.last_epoch,
                      "optimizer_step_called": False, "policy_forward_called": False}))


if __name__ == "__main__":
    main()
