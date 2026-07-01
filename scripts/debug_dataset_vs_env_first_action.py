from pathlib import Path
import h5py
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from train_smolvla_v1 import build_policy, add_language_tokens, move_batch_to_device


CHECKPOINT = Path("outputs/smolvla_v1_2_full/checkpoint_final.pt")
DATASET_PATH = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"


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


def image_to_tensor(img):
    img = torch.as_tensor(img).float()

    # HWC -> CHW
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)

    # 0-255 -> 0-1
    if img.max() > 2:
        img = img / 255.0

    return img.unsqueeze(0)


def make_batch_from_dataset(demo):
    agent = image_to_tensor(demo["obs/agentview_rgb"][0])
    wrist = image_to_tensor(demo["obs/eye_in_hand_rgb"][0])

    joint = torch.as_tensor(demo["obs/joint_states"][0]).float()
    ee_pos = torch.as_tensor(demo["obs/ee_pos"][0]).float()
    ee_ori = torch.as_tensor(demo["obs/ee_ori"][0]).float()
    grip = torch.as_tensor(demo["obs/gripper_states"][0]).float()

    state = torch.cat([joint, ee_pos, ee_ori, grip], dim=0).unsqueeze(0)

    batch = {
        "observation.images.agentview": agent,
        "observation.images.wrist": wrist,
        "observation.state": state,
        "task": [LANGUAGE],
    }

    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, "cuda")
    return batch


def make_batch_from_env(obs):
    agent = image_to_tensor(obs["agentview_image"])
    wrist = image_to_tensor(obs["robot0_eye_in_hand_image"])

    joint = torch.as_tensor(obs["robot0_joint_pos"]).float()
    ee_pos = torch.as_tensor(obs["robot0_eef_pos"]).float()
    ee_ori = torch.as_tensor(quat_xyzw_to_axis_angle(obs["robot0_eef_quat"])).float()
    grip = torch.as_tensor(obs["robot0_gripper_qpos"]).float()

    state = torch.cat([joint, ee_pos, ee_ori, grip], dim=0).unsqueeze(0)

    batch = {
        "observation.images.agentview": agent,
        "observation.images.wrist": wrist,
        "observation.state": state,
        "task": [LANGUAGE],
    }

    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, "cuda")
    return batch


def select_action(policy, batch):
    with torch.no_grad():
        action = policy.select_action(batch)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]

    return action


device = torch.device("cuda")
policy = build_policy(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()

print("=" * 80)
print("LOAD DATASET FIRST FRAME")
print("=" * 80)

with h5py.File(DATASET_PATH, "r") as f:
    demo = f["data/demo_0"]
    dataset_batch = make_batch_from_dataset(demo)
    expert_action = np.asarray(demo["actions"][0])

dataset_pred = select_action(policy, dataset_batch)

print("expert action:")
print(expert_action)
print("model pred on dataset first frame:")
print(dataset_pred)
print("abs diff dataset_pred - expert:")
print(np.abs(dataset_pred - expert_action))
print("MAE dataset first action:", np.mean(np.abs(dataset_pred - expert_action)))

print("=" * 80)
print("LOAD ENV FIRST FRAME")
print("=" * 80)

suite_cls = benchmark.get_benchmark("libero_spatial")
suite = suite_cls()
task = suite.get_task(0)
bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_heights=128,
    camera_widths=128,
)

init_states = suite.get_task_init_states(0)
obs = env.reset()
tmp = env.set_init_state(init_states[0])
if tmp is not None:
    obs = tmp

env_batch = make_batch_from_env(obs)
env_pred = select_action(policy, env_batch)

env.close()

print("model pred on env first frame:")
print(env_pred)
print("abs diff env_pred - dataset_pred:")
print(np.abs(env_pred - dataset_pred))
print("MAE env_pred vs dataset_pred:", np.mean(np.abs(env_pred - dataset_pred)))

print("=" * 80)
print("STATE COMPARE")
print("=" * 80)

print("dataset state:")
print(dataset_batch["observation.state"].detach().cpu().numpy()[0])
print("env state:")
print(env_batch["observation.state"].detach().cpu().numpy()[0])
print("state abs diff:")
print(np.abs(
    env_batch["observation.state"].detach().cpu().numpy()[0]
    - dataset_batch["observation.state"].detach().cpu().numpy()[0]
))
