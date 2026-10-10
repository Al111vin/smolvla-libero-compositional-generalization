"""CPU-only checkpoint parameter mapping audit; never forward/backward/step."""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = SmolVLAConfig.from_pretrained(args.checkpoint / "pretrained_model")
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint / "pretrained_model",
                                         config=config, local_files_only=True, strict=True)
    parameters = list(policy.parameters())
    selected = list(policy.get_optim_params())
    assert len(parameters) == len(selected)
    assert all(a is b for a, b in zip(parameters, selected))
    groups = json.loads((args.checkpoint / "training_state/optimizer_param_groups.json").read_text())
    ids = [i for group in groups for i in group["params"]]
    assert ids == list(range(len(parameters))), "saved parameter order mismatch"
    with safe_open(args.checkpoint / "training_state/optimizer_state.safetensors",
                   framework="pt", device="cpu") as source:
        active = {int(key.split("/")[1]) for key in source.keys()}
        trainable = {i for i, param in enumerate(parameters) if param.requires_grad}
        assert active == trainable, "saved active IDs differ from current freeze mask"
        for i in active:
            assert list(parameters[i].shape) == source.get_slice(f"state/{i}/exp_avg").get_shape()
    print(json.dumps({"status": "cpu_parameter_mapping_passed", "parameters": len(parameters),
                      "active_states": len(active), "trainable_ids_exactly_match": True,
                      "optimizer_restored": False, "policy_forward_called": False}))


if __name__ == "__main__":
    main()
