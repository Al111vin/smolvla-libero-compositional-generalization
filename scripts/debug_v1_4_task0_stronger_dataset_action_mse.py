from pathlib import Path
import h5py
import torch
import numpy as np

from train_smolvla_v1_4_task0_stronger import build_policy, add_language_tokens, move_batch_to_device


device = "cuda"
ckpt_path = "outputs/smolvla_v1_4_task0_stronger/checkpoint_final.pt"
data_path = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
num_samples = 64


ckpt = torch.load(ckpt_path, map_location=device)

policy = build_policy(device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.to(device)
policy.eval()


def image_to_tensor(imgs):
    x = torch.as_tensor(imgs).float()
    if x.max() > 1:
        x = x / 255.0
    x = x.permute(0, 3, 1, 2)
    return x


with h5py.File(data_path, "r") as f:
    demo_key = sorted(f["data"].keys())[0]
    demo = f["data"][demo_key]

    obs = demo["obs"]

    agentview = obs["agentview_rgb"][:num_samples]
    wrist = obs["eye_in_hand_rgb"][:num_samples]
    joint = obs["joint_states"][:num_samples]
    ee_pos = obs["ee_pos"][:num_samples]
    ee_ori = obs["ee_ori"][:num_samples]
    gripper = obs["gripper_states"][:num_samples]
    true_actions = demo["actions"][:num_samples]


state = np.concatenate([joint, ee_pos, ee_ori, gripper], axis=1).astype(np.float32)

batch = {
    "observation.images.agentview": image_to_tensor(agentview),
    "observation.images.wrist": image_to_tensor(wrist),
    "observation.state": torch.as_tensor(state).float(),
    "task": [language] * num_samples,
}

batch = add_language_tokens(batch, policy)
batch = move_batch_to_device(batch, device)

with torch.no_grad():
    pred = policy.predict_action_chunk(batch)

if isinstance(pred, torch.Tensor):
    pred = pred.detach().cpu().numpy()

if pred.ndim == 3:
    pred = pred[:, 0, :]

true_actions = true_actions.astype(np.float32)

mse = ((pred - true_actions) ** 2).mean()
mae = np.abs(pred - true_actions).mean()

print("DATASET ACTION MSE CHECK")
print("========================")
print("data file:", data_path)
print("demo key:", demo_key)
print("num samples:", num_samples)
print("pred shape:", pred.shape)
print("true shape:", true_actions.shape)
print("MSE:", float(mse))
print("MAE:", float(mae))

print()
print("Pred action stats:")
print("min:", float(pred.min()))
print("max:", float(pred.max()))
print("mean:", float(pred.mean()))
print("std:", float(pred.std()))
print("abs mean:", float(np.abs(pred).mean()))

print()
print("True action stats:")
print("min:", float(true_actions.min()))
print("max:", float(true_actions.max()))
print("mean:", float(true_actions.mean()))
print("std:", float(true_actions.std()))
print("abs mean:", float(np.abs(true_actions).mean()))

print()
for i in range(7):
    p = pred[:, i]
    t = true_actions[:, i]
    dim_mse = ((p - t) ** 2).mean()
    dim_mae = np.abs(p - t).mean()
    print(
        f"action_{i}: "
        f"mse={dim_mse:.6f}, "
        f"mae={dim_mae:.6f}, "
        f"pred_abs_mean={np.abs(p).mean():.6f}, "
        f"true_abs_mean={np.abs(t).mean():.6f}"
    )