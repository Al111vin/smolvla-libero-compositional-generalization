from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


DEMO_PATH = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)


class LiberoHDF5Dataset(Dataset):
    def __init__(self, hdf5_path: Path):
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        self.index = []

        with h5py.File(self.hdf5_path, "r") as f:
            self.demo_keys = sorted(
                list(f["data"].keys()),
                key=lambda x: int(x.replace("demo_", "")),
            )

            self.language = f["data"].attrs["problem_info"]

            for demo_key in self.demo_keys:
                num_steps = f[f"data/{demo_key}/actions"].shape[0]
                for step_idx in range(num_steps):
                    self.index.append((demo_key, step_idx))

        print("Loaded HDF5 dataset:")
        print("  file:", self.hdf5_path)
        print("  num demos:", len(self.demo_keys))
        print("  num transitions:", len(self.index))

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

            problem_info = f["data"].attrs["problem_info"]

        # image: uint8 HWC -> float32 CHW, range 0 to 1
        agentview_rgb = torch.from_numpy(agentview_rgb).permute(2, 0, 1).float() / 255.0
        wrist_rgb = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0

        # state: concat robot-related low-dimensional states
        state = torch.cat(
            [
                torch.from_numpy(joint_states).float(),      # 7
                torch.from_numpy(ee_pos).float(),            # 3
                torch.from_numpy(ee_ori).float(),            # 3
                torch.from_numpy(gripper_states).float(),    # 2
            ],
            dim=0,
        )

        action = torch.from_numpy(action).float()

        sample = {
            "observation.images.agentview": agentview_rgb,
            "observation.images.wrist": wrist_rgb,
            "observation.state": state,
            "action": action,
            "language": str(problem_info),
            "demo_key": demo_key,
            "step_idx": step_idx,
        }

        return sample


def main():
    print("=" * 80)
    print("LIBERO PyTorch Dataset loading test")

    dataset = LiberoHDF5Dataset(DEMO_PATH)

    print("=" * 80)
    print("Single sample test")

    sample = dataset[0]

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")

    print("=" * 80)
    print("DataLoader batch test")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: type={type(value)}")

    assert batch["observation.images.agentview"].shape == (4, 3, 128, 128)
    assert batch["observation.images.wrist"].shape == (4, 3, 128, 128)
    assert batch["action"].shape == (4, 7)

    print("=" * 80)
    print("LIBERO PyTorch Dataset loading test passed.")


if __name__ == "__main__":
    main()