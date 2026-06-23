import os
from pathlib import Path
import json
import pickle

os.environ["MUJOCO_GL"] = "glx"

import numpy as np
import pandas as pd

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


FINAL_TASK_PATH = Path("data/final_task_set.csv")
OUTPUT_DIR = Path("data/demos/random_smoke")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FINAL_TASK_PATH)
    row = df.iloc[0]

    suite = row["suite"]
    task_id = int(row["libero_task_id"])

    print("=" * 80)
    print("Random demo smoke collection")
    print("suite:", suite)
    print("task_id:", task_id)

    bench_cls = benchmark.get_benchmark(suite)
    bench = bench_cls()

    task = bench.get_task(task_id)

    print("language:", task.language)
    print("bddl_file:", task.bddl_file)
    print("problem_folder:", task.problem_folder)

    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    print("full bddl path:", bddl_path)

    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
    )

    env.seed(0)
    env.reset()

    init_states = bench.get_task_init_states(task_id)
    print("number of init states:", len(init_states))

    obs = env.set_init_state(init_states[0])

    if obs is None:
        obs = env.reset()

    action_low, action_high = env.env.action_spec

    print("action_low shape:", action_low.shape)
    print("action_high shape:", action_high.shape)
    print("action_low:", action_low)
    print("action_high:", action_high)

    episode = {
        "suite": suite,
        "task_id": task_id,
        "language": task.language,
        "bddl_file": task.bddl_file,
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
    }

    max_steps = 20

    for step in range(max_steps):
        action = np.random.uniform(low=action_low, high=action_high).astype(np.float32)

        next_obs, reward, done, info = env.step(action)

        episode["observations"].append(
            {
                "robot0_joint_pos": obs["robot0_joint_pos"],
                "robot0_eef_pos": obs["robot0_eef_pos"],
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                "robot0_proprio-state": obs["robot0_proprio-state"],
                "object-state": obs["object-state"],
                "agentview_image": obs["agentview_image"],
                "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            }
        )

        episode["actions"].append(action)
        episode["rewards"].append(reward)
        episode["dones"].append(done)
        episode["infos"].append(info)

        print(f"step {step:03d} | reward={reward} | done={done}")

        obs = next_obs

        if done:
            break

    env.close()

    output_path = OUTPUT_DIR / f"{suite}_task_{task_id}_random_episode.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(episode, f)

    meta_path = OUTPUT_DIR / f"{suite}_task_{task_id}_random_episode_meta.json"

    meta = {
        "suite": suite,
        "task_id": task_id,
        "language": task.language,
        "bddl_file": task.bddl_file,
        "num_steps": len(episode["actions"]),
        "output_path": str(output_path),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print("Saved random demo smoke episode to:")
    print(output_path)
    print("Saved metadata to:")
    print(meta_path)
    print("Random demo smoke collection passed.")


if __name__ == "__main__":
    main()