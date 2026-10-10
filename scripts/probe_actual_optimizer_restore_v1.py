"""Read-only CPU optimizer restore audit; no forward/backward/optimizer.step."""
import argparse
import json
from pathlib import Path
import draccus
import torch
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
    load_optimizer_state(optimizer, state_dir)
    saved_groups = json.loads((state_dir / "optimizer_param_groups.json").read_text())
    restored = optimizer.state_dict()
    assert len(restored["param_groups"]) == len(saved_groups)
    for actual, saved in zip(restored["param_groups"], saved_groups):
        assert actual["params"] == saved["params"] and actual["lr"] == saved["lr"]
    checked = 0
    with safe_open(state_dir / "optimizer_state.safetensors", framework="pt", device="cpu") as source:
        expected_ids = {int(key.split("/")[1]) for key in source.keys()}
        assert set(restored["state"]) == expected_ids
        for key in source.keys():
            _, identifier, field = key.split("/")
            saved = source.get_tensor(key)
            actual = restored["state"][int(identifier)][field]
            assert torch.equal(actual, saved), key
            assert torch.isfinite(actual).all().item(), key
            checked += 1
    print(json.dumps({"status": "actual_cpu_optimizer_restore_passed",
                      "state_tensors_equal_and_finite": checked,
                      "active_states": len(expected_ids),
                      "optimizer_step_called": False, "policy_forward_called": False}))


if __name__ == "__main__":
    main()
