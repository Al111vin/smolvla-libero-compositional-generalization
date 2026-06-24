from pathlib import Path
import json

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


DEMO_PATH = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

OUTPUT_DIR = Path("outputs/smolvla_smoke")


class LiberoSmolVLADataset(Dataset):
    def __init__(self, hdf5_path: Path, max_samples: int = 256):
        self.hdf5_path = Path(hdf5_path)
        self.max_samples = max_samples

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        self.index = []

        with h5py.File(self.hdf5_path, "r") as f:
            self.demo_keys = sorted(
                list(f["data"].keys()),
                key=lambda x: int(x.replace("demo_", "")),
            )

            problem_info = json.loads(f["data"].attrs["problem_info"])
            self.language = "".join(problem_info["language_instruction"]).strip('"')

            for demo_key in self.demo_keys:
                num_steps = f[f"data/{demo_key}/actions"].shape[0]
                for step_idx in range(num_steps):
                    self.index.append((demo_key, step_idx))

        self.index = self.index[: self.max_samples]

        print("Loaded LIBERO HDF5:")
        print("  file:", self.hdf5_path)
        print("  language:", self.language)
        print("  num demos:", len(self.demo_keys))
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

            action = demo["actions"][step_idx]

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

        action = torch.from_numpy(action).float().unsqueeze(0)

        return {
            "observation.images.agentview": agentview_rgb,
            "observation.images.wrist": wrist_rgb,
            "observation.state": state,
            "action": action,
            "task": self.language,
        }


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


def main():
    print("=" * 80)
    print("SmolVLA mini training smoke test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = LiberoSmolVLADataset(DEMO_PATH, max_samples=256)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    policy = build_policy(device)
    policy.train()

    optimizer = AdamW(
        policy.get_optim_params(),
        lr=1e-5,
        weight_decay=1e-6,
    )

    max_steps = 20
    step = 0

    for batch in dataloader:
        batch = add_language_tokens(batch, policy)
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        loss, loss_dict = policy.forward(batch)

        if not torch.isfinite(loss):
            raise ValueError(f"Loss is not finite: {loss}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        print(
            f"step {step:03d} | "
            f"loss={loss.item():.6f} | "
            f"grad_norm={float(grad_norm):.6f} | "
            f"loss_dict={loss_dict}"
        )

        step += 1

        if step >= max_steps:
            break

    checkpoint_path = OUTPUT_DIR / "smolvla_smoke_checkpoint.pt"

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )

    print("=" * 80)
    print("Saved checkpoint:", checkpoint_path)
    print("SmolVLA mini training smoke test passed.")


if __name__ == "__main__":
    main()