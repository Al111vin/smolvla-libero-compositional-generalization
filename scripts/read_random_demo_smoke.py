from pathlib import Path
import pickle
import json

import numpy as np


DEMO_DIR = Path("data/demos/random_smoke")
PKL_PATH = DEMO_DIR / "libero_spatial_task_0_random_episode.pkl"
META_PATH = DEMO_DIR / "libero_spatial_task_0_random_episode_meta.json"


def main():
    print("=" * 80)
    print("Read random demo smoke test")

    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Demo pkl not found: {PKL_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata json not found: {META_PATH}")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    print("Metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    with open(PKL_PATH, "rb") as f:
        episode = pickle.load(f)

    print("=" * 80)
    print("Episode basic info:")
    print("suite:", episode["suite"])
    print("task_id:", episode["task_id"])
    print("language:", episode["language"])
    print("bddl_file:", episode["bddl_file"])

    observations = episode["observations"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    dones = episode["dones"]

    print("=" * 80)
    print("Episode length:")
    print("num observations:", len(observations))
    print("num actions:", len(actions))
    print("num rewards:", len(rewards))
    print("num dones:", len(dones))

    if len(observations) == 0:
        raise ValueError("No observations found in episode.")

    if len(actions) == 0:
        raise ValueError("No actions found in episode.")

    first_obs = observations[0]
    first_action = actions[0]

    print("=" * 80)
    print("First observation keys and shapes:")

    for key, value in first_obs.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    print("=" * 80)
    print("First action:")
    print("  shape:", first_action.shape)
    print("  dtype:", first_action.dtype)
    print("  value:", first_action)

    print("=" * 80)
    print("Reward summary:")
    print("  rewards:", rewards)
    print("  total reward:", float(np.sum(rewards)))
    print("  any done:", any(dones))

    required_obs_keys = [
        "robot0_joint_pos",
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "robot0_proprio-state",
        "object-state",
        "agentview_image",
        "robot0_eye_in_hand_image",
    ]

    for key in required_obs_keys:
        if key not in first_obs:
            raise KeyError(f"Missing required observation key: {key}")

    if first_obs["agentview_image"].shape != (128, 128, 3):
        raise ValueError(
            f"Unexpected agentview_image shape: {first_obs['agentview_image'].shape}"
        )

    if first_obs["robot0_eye_in_hand_image"].shape != (128, 128, 3):
        raise ValueError(
            "Unexpected robot0_eye_in_hand_image shape: "
            f"{first_obs['robot0_eye_in_hand_image'].shape}"
        )

    if first_action.shape != (7,):
        raise ValueError(f"Unexpected action shape: {first_action.shape}")

    print("=" * 80)
    print("Read random demo smoke test passed.")


if __name__ == "__main__":
    main()