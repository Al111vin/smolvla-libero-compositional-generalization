from pathlib import Path
import csv
import json
import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


DATASET_DIR = Path("datasets/libero/datasets/libero_spatial")
OUTPUT_DIR = Path("outputs/smolvla_v1_2")


class MultiLiberoSmolVLADataset(Dataset):
    def __init__(self, dataset_dir: Path, max_samples_per_file: int | None = 1000):
        self.dataset_dir = Path(dataset_dir)
        self.max_samples_per_file = max_samples_per_file

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.hdf5_paths = sorted(self.dataset_dir.glob("*.hdf5"))

        if len(self.hdf5_paths) == 0:
            raise FileNotFoundError(f"No hdf5 files found in {self.dataset_dir}")

        self.index = []
        self.languages = {}

        for hdf5_path in self.hdf5_paths:
            local_index = []

            with h5py.File(hdf5_path, "r") as f:
                problem_info = json.loads(f["data"].attrs["problem_info"])
                language = "".join(problem_info["language_instruction"]).strip('"')
                self.languages[str(hdf5_path)] = language

                demo_keys = sorted(
                    list(f["data"].keys()),
                    key=lambda x: int(x.replace("demo_", "")),
                )

                for demo_key in demo_keys:
                    num_steps = f[f"data/{demo_key}/actions"].shape[0]
                    for step_idx in range(num_steps):
                        local_index.append((str(hdf5_path), demo_key, step_idx))

            if self.max_samples_per_file is not None:
                random.shuffle(local_index)
                local_index = local_index[: self.max_samples_per_file]

            self.index.extend(local_index)

        random.shuffle(self.index)

        print("Loaded multi-task LIBERO dataset:")
        print("  dataset_dir:", self.dataset_dir)
        print("  num hdf5 files:", len(self.hdf5_paths))
        print("  total used transitions:", len(self.index))
        print("  max_samples_per_file:", self.max_samples_per_file)

        print("=" * 80)
        print("Tasks:")
        for path in self.hdf5_paths:
            print(" ", path.name)
            print("   language:", self.languages[str(path)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        hdf5_path, demo_key, step_idx = self.index[idx]

        with h5py.File(hdf5_path, "r") as f:
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
            "task": self.languages[hdf5_path],
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


@torch.no_grad()
def run_validation(policy, val_loader, device, max_val_batches=5):
    policy.eval()

    val_losses = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_val_batches:
            break

        batch = add_language_tokens(batch, policy)
        batch = move_batch_to_device(batch, device)

        loss, _ = policy.forward(batch)
        val_losses.append(loss.item())

    policy.train()

    if len(val_losses) == 0:
        return None

    return sum(val_losses) / len(val_losses)


def save_checkpoint(policy, optimizer, step, path):
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def main():
    print("=" * 80)
    print("SmolVLA full fine-tuning v1")

    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = MultiLiberoSmolVLADataset(
        DATASET_DIR,
        max_samples_per_file=1000,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print("=" * 80)
    print("Split:")
    print("  train transitions:", len(train_dataset))
    print("  val transitions:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    policy = build_policy(device)
    policy.train()

    optimizer = AdamW(
        policy.get_optim_params(),
        lr=1e-5,
        weight_decay=1e-6,
    )

    max_steps = 10000
    val_every = 50
    save_every = 100

    log_path = OUTPUT_DIR / "train_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "train_loss", "grad_norm", "val_loss"],
        )
        writer.writeheader()

    step = 0

    while step < max_steps:
        for batch in train_loader:
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

            val_loss = None

            if step % val_every == 0:
                val_loss = run_validation(policy, val_loader, device)
                print(
                    f"step {step:04d} | "
                    f"train_loss={loss.item():.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"grad_norm={float(grad_norm):.6f}"
                )
            else:
                print(
                    f"step {step:04d} | "
                    f"train_loss={loss.item():.6f} | "
                    f"grad_norm={float(grad_norm):.6f}"
                )

            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["step", "train_loss", "grad_norm", "val_loss"],
                )
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "grad_norm": float(grad_norm),
                        "val_loss": "" if val_loss is None else val_loss,
                    }
                )

            if step > 0 and step % save_every == 0:
                checkpoint_path = OUTPUT_DIR / f"checkpoint_step_{step}.pt"
                save_checkpoint(policy, optimizer, step, checkpoint_path)
                print("Saved checkpoint:", checkpoint_path)

            step += 1

            if step >= max_steps:
                break

    final_checkpoint_path = OUTPUT_DIR / "checkpoint_final.pt"
    save_checkpoint(policy, optimizer, step, final_checkpoint_path)

    print("=" * 80)
    print("Saved final checkpoint:", final_checkpoint_path)
    print("Saved log:", log_path)
    print("SmolVLA full fine-tuning v1 passed.")


if __name__ == "__main__":
    main()