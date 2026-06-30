import os
os.environ["MUJOCO_GL"] = "glx"

from pathlib import Path
import csv
import torch
import numpy as np

from train_smolvla_v1 import build_policy, add_language_tokens, move_batch_to_device
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


device = "cuda"
suite_name = "libero_spatial"
task_id = 0
ckpt_path = "outputs/smolvla_v1/checkpoint_final.pt"
results_path = Path("results/debug_v1_actions_task0.csv")
max_steps = 300


ckpt = torch.load(ckpt_path, map_location=device)

policy = build_policy(device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.to(device)
policy.eval()

suite_cls = benchmark.get_benchmark(suite_name)
suite = suite_cls()
task = suite.get_task(task_id)

bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_heights=128,
    camera_widths=128,
)

init_states = suite.get_task_init_states(task_id)

obs = env.reset()
if init_states is not None and len(init_states) > 0:
    tmp = env.set_init_state(init_states[0])
    if tmp is not None:
        obs = tmp


def image_to_tensor(img):
    x = torch.as_tensor(img).float()
    if x.max() > 1:
        x = x / 255.0
    if x.ndim == 3:
        x = x.permute(2, 0, 1)
    return x.unsqueeze(0)


def make_batch(obs, language):
    joint_pos = torch.as_tensor(obs["robot0_joint_pos"]).float()
    eef_pos = torch.as_tensor(obs["robot0_eef_pos"]).float()
    eef_quat = torch.as_tensor(obs["robot0_eef_quat"]).float()[:3]
    gripper = torch.as_tensor(obs["robot0_gripper_qpos"]).float()

    state = torch.cat([joint_pos, eef_pos, eef_quat, gripper], dim=0).unsqueeze(0)

    batch = {
        "observation.images.agentview": image_to_tensor(obs["agentview_image"]),
        "observation.images.wrist": image_to_tensor(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": [language],
    }

    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, device)
    return batch


results_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
total_reward = 0.0
info = {}

for step in range(max_steps):
    batch = make_batch(obs, task.language)

    with torch.no_grad():
        action = policy.select_action(batch)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]

    action = np.asarray(action).reshape(-1)
    action = action[:7]

    row = {
        "step": step,
        "action_0": action[0],
        "action_1": action[1],
        "action_2": action[2],
        "action_3": action[3],
        "action_4": action[4],
        "action_5": action[5],
        "action_6": action[6],
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "action_mean": float(action.mean()),
        "action_std": float(action.std()),
    }

    rows.append(row)

    obs, reward, done, info = env.step(action)
    total_reward += float(reward)

    if done:
        break


env.close()

with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "step",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "action_4",
            "action_5",
            "action_6",
            "action_min",
            "action_max",
            "action_mean",
            "action_std",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

success = bool(info.get("success", info.get("is_success", False)))

print("TASK:", task.language)
print("SUCCESS:", success)
print("TOTAL REWARD:", total_reward)
print("STEPS:", len(rows))
print("SAVED TO:", results_path)

all_actions = np.array(
    [[r[f"action_{i}"] for i in range(7)] for r in rows],
    dtype=np.float32,
)

print("ACTION GLOBAL MIN:", float(all_actions.min()))
print("ACTION GLOBAL MAX:", float(all_actions.max()))
print("ACTION GLOBAL MEAN:", float(all_actions.mean()))
print("ACTION GLOBAL STD:", float(all_actions.std()))
print("ACTION ABS MEAN:", float(np.abs(all_actions).mean()))