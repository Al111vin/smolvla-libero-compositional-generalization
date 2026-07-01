import os
os.environ["MUJOCO_GL"] = "glx"

from pathlib import Path
import csv
import torch
import numpy as np

from train_smolvla_v1_5_task0_gripper_weighted import build_policy, add_language_tokens, move_batch_to_device
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


device = "cuda"
suite_name = "libero_spatial"
task_ids = [0]
ckpt_path = "outputs/smolvla_v1_5_task0_gripper_weighted/checkpoint_step_6000.pt"
results_path = Path("results/eval_v1_5_task0_step6000_postprocess.csv")
max_steps = 300


ckpt = torch.load(ckpt_path, map_location=device)

policy = build_policy(device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.to(device)
policy.eval()

suite_cls = benchmark.get_benchmark(suite_name)
suite = suite_cls()


def image_to_tensor(img):
    x = torch.as_tensor(img).float()
    if x.max() > 1:
        x = x / 255.0
    if x.ndim == 3:
        x = x.permute(2, 0, 1)
    return x.unsqueeze(0)


def quat_xyzw_to_axis_angle(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q

    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    q = q / norm
    x, y, z, w = q

    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))

    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / s

    return (axis * angle).astype(np.float32)


def make_batch(obs, language):
    joint_pos = torch.as_tensor(obs["robot0_joint_pos"]).float()
    eef_pos = torch.as_tensor(obs["robot0_eef_pos"]).float()
    eef_ori = torch.as_tensor(quat_xyzw_to_axis_angle(obs["robot0_eef_quat"])).float()
    gripper = torch.as_tensor(obs["robot0_gripper_qpos"]).float()

    state = torch.cat([joint_pos, eef_pos, eef_ori, gripper], dim=0).unsqueeze(0)

    batch = {
        "observation.images.agentview": image_to_tensor(obs["agentview_image"]),
        "observation.images.wrist": image_to_tensor(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": [language],
    }

    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, device)
    return batch


def run_one_task(task_id):
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

    total_reward = 0.0
    info = {}
    prev_action = None

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

        # V1.5 post-processing:
        # 1. clip actions to expert range
        # 2. smooth motion actions to reduce closed-loop jitter
        # 3. binarize gripper action
        action = action.clip(-1.0, 1.0)

        if prev_action is not None:
            action[:6] = 0.7 * prev_action[:6] + 0.3 * action[:6]

        action[6] = -1.0 if action[6] < 0 else 1.0
        prev_action = action.copy()

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        if done:
            break

    success = bool(info.get("success", info.get("is_success", False))) or total_reward > 0
    steps = step + 1
    env.close()

    print("=" * 80)
    print("TASK ID:", task_id)
    print("TASK:", task.language)
    print("SUCCESS:", success)
    print("TOTAL REWARD:", total_reward)
    print("STEPS:", steps)

    return {
        "suite": suite_name,
        "task_id": task_id,
        "language": task.language,
        "success": success,
        "total_reward": total_reward,
        "steps": steps,
        "checkpoint": ckpt_path,
    }


results_path.parent.mkdir(parents=True, exist_ok=True)

all_results = []

with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "suite",
            "task_id",
            "language",
            "success",
            "total_reward",
            "steps",
            "checkpoint",
        ],
    )

    writer.writeheader()

    for task_id in task_ids:
        result = run_one_task(task_id)
        writer.writerow(result)
        f.flush()
        all_results.append(result)


success_count = sum(1 for r in all_results if r["success"])
success_rate = success_count / len(all_results)

print("=" * 80)
print("EVALUATION FINISHED")
print("SUITE:", suite_name)
print("NUM TASKS:", len(all_results))
print("SUCCESS COUNT:", success_count)
print("SUCCESS RATE:", success_rate)
print("SAVED TO:", results_path)
