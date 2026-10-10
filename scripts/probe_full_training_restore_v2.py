"""Bounded CPU full restore and half-LR audit; no inference or updates."""
import argparse
import json
import sys
from pathlib import Path
import draccus
import torch
import lerobot.scripts.lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.train_utils import load_training_state
from safetensors import safe_open
from resume_lr_guard_v1 import apply_post_restore_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    pretrained = args.checkpoint / "pretrained_model"
    state_dir = args.checkpoint / "training_state"
    cfg = draccus.decode(TrainPipelineConfig, json.loads((pretrained / "train_config.json").read_text()))
    expected = {40000: 20000, 160000: 80000}[cfg.steps]
    assert args.checkpoint.name == f"{expected:06d}"
    config = PreTrainedConfig.from_pretrained(pretrained)
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=config,
                                         local_files_only=True, strict=True)
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    if cfg.steps == 40000:
        scheduler = cfg.scheduler.build(optimizer, 40000)
    else:
        sys.path.insert(0, "/root/smolvla-training-prep/recovery/exposure_matched_4task_v1_20261009/scripts")
        from exposure_matched_blocks_v1 import build_block_lambda_scheduler
        scheduler = build_block_lambda_scheduler(optimizer, cfg.scheduler, blocks=40000, task_count=4)
    step, optimizer, scheduler = load_training_state(args.checkpoint, optimizer, scheduler)
    assert step == expected == scheduler.last_epoch
    saved_groups = json.loads((state_dir / "optimizer_param_groups.json").read_text())
    lrs = [g["lr"] for g in saved_groups]
    assert scheduler.get_last_lr() == lrs
    restored = optimizer.state_dict()
    assert [g["params"] for g in restored["param_groups"]] == [g["params"] for g in saved_groups]
    bases = list(scheduler.base_lrs)
    apply_post_restore_factor(optimizer, scheduler, restored_step=step,
                             expected_step=expected, expected_lrs=lrs, factor=0.5)
    assert scheduler.base_lrs == bases
    assert scheduler.get_last_lr() == [lr * 0.5 for lr in lrs]
    assert [g["lr"] for g in optimizer.param_groups] == scheduler.get_last_lr()
    checked = 0
    restored = optimizer.state_dict()
    with safe_open(state_dir / "optimizer_state.safetensors", framework="pt", device="cpu") as source:
        assert set(restored["state"]) == {int(k.split("/")[1]) for k in source.keys()}
        for key in source.keys():
            _, identifier, field = key.split("/")
            actual = restored["state"][int(identifier)][field]
            assert torch.equal(actual, source.get_tensor(key)), key
            assert torch.isfinite(actual).all().item(), key
            checked += 1
    print(json.dumps({"status": "full_restore_half_lr_cpu_passed", "step": step,
                      "state_tensors_equal_and_finite_after_intervention": checked,
                      "boundary_lrs": scheduler.get_last_lr(), "base_lrs_unchanged": True,
                      "rng_restore_called": True, "optimizer_step_called": False,
                      "policy_forward_called": False}))


if __name__ == "__main__":
    main()
