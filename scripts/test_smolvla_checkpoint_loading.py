from pathlib import Path
import json

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


DEMO_PATH = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

CHECKPOINT_PATH = Path("outputs/smolvla_v1/checkpoint_final.pt")


class LiberoEvalBatchDataset(Dataset):
    def __init__(self, hdf5_path: Path, max_samples: int = 16):
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        self.index = []

        with h5py.File(self.hdf5_path, "r") as f:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            self.language = "".join(problem_info["language_instruction"]).strip('"')

            demo_keys = sorted(
                list(f["data"].keys()),
                key=lambda x: int(x.replace("demo_", "")),
            )

            for demo_key in demo_keys:
                num_steps = f[f"data/{demo_key}/actions"].shape[0]
                for step_idx in range(num_steps):
                    self.index.append((demo_key, step_idx))

        self.index = self.index[:max_samples]

        print("Loaded eval dataset:")
        print("  file:", self.hdf5_path)
        print("  language:", self.language)
        print("  used transitions:", len(self.index))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_key, step_idx = self.index[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            demo = f[f"data/{demo_key}"]

            agentview_rgb = demo["obs/agentview_rgb"][step_idx]
            wrist_rgb = demo["obs/eye_in_hand_rgb"][step_idx]

            joint_states = demo["obs/joint_states"][step_idx]
            ee_pos = demo["obs/ee_pos"][step_idx]
            ee_ori = demo["obs/ee_ori"][step_idx]
            gripper_states = demo["obs/gripper_states"][step_idx]

        agentview_rgb = torch.from_numpy(agentview_rgb).permute(2, 0, 1).float() / 255.0
        wrist_rgb = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0

        state = torch.cat(
            [
                torch.from_numpy(joint_states).float(),
                torch.from_numpy(ee_pos).float(),
                torch.from_numpy(ee_ori).float(),
                torch.from_numpy(gripper_states).float(),
            ],
            dim=0,
        )

        return {
            "observation.images.agentview": agentview_rgb,
            "observation.images.wrist": wrist_rgb,
            "observation.state": state,
            "task": self.language,
        }


def build_policy(device):
    input_features = {
        "observation.images.agentview": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 128, 128),
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 128, 128),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(15,),
        ),
    }

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        chunk_size=1,
        n_action_steps=1,
        max_state_dim=32,
        max_action_dim=32,
        tokenizer_max_length=48,
        load_vlm_weights=False,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        num_vlm_layers=1,
        num_expert_layers=1,
        resize_imgs_with_padding=(512, 512),
        push_to_hub=False,
    )

    policy = SmolVLAPolicy(config)
    policy.to(device)

    return policy


def add_language_tokens(batch, policy):
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    tokenized = tokenizer(
        batch["task"],
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )

    batch["observation.language.tokens"] = tokenized["input_ids"]
    batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool()

    del batch["task"]

    return batch


def move_batch_to_device(batch, device):
    out = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value

    return out


def main():
    print("=" * 80)
    print("SmolVLA checkpoint loading test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    dataset = LiberoEvalBatchDataset(DEMO_PATH)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print("=" * 80)
    print("Building policy...")
    policy = build_policy(device)

    print("=" * 80)
    print("Loading checkpoint:", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    batch = next(iter(dataloader))
    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, device)

    print("=" * 80)
    print("Prepared eval batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"{key}: {type(value)}")

    print("=" * 80)
    print("Running predict_action_chunk...")

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)

    print("predicted actions shape:", tuple(actions.shape))
    print("predicted actions dtype:", actions.dtype)
    print("predicted actions device:", actions.device)
    print("first action chunk:", actions[0])

    assert actions.shape == (2, 1, 7)

    print("=" * 80)
    print("SmolVLA checkpoint loading test passed.")


if __name__ == "__main__":
    main()