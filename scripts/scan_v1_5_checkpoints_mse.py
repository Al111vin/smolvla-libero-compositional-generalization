from pathlib import Path
import re
import h5py
import numpy as np
import torch
import pandas as pd

from train_smolvla_v1_5_task0_gripper_weighted import (
    build_policy,
    add_language_tokens,
    move_batch_to_device,
)


CKPT_DIR = Path("outputs/smolvla_v1_5_task0_gripper_weighted")
DATASET_PATH = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
OUT_CSV = Path("results/scan_v1_5_checkpoints_mse.csv")

LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def image_to_tensor(img):
    x = torch.as_tensor(img).float()

    if x.max() > 1:
        x = x / 255.0

    if x.ndim == 3:
        x = x.permute(2, 0, 1)

    return x.unsqueeze(0)


def make_batch(demo, i, device, policy):
    agent = image_to_tensor(demo["obs/agentview_rgb"][i])
    wrist = image_to_tensor(demo["obs/eye_in_hand_rgb"][i])

    joint = torch.as_tensor(demo["obs/joint_states"][i]).float()
    ee_pos = torch.as_tensor(demo["obs/ee_pos"][i]).float()
    ee_ori = torch.as_tensor(demo["obs/ee_ori"][i]).float()
    gripper = torch.as_tensor(demo["obs/gripper_states"][i]).float()

    state = torch.cat([joint, ee_pos, ee_ori, gripper], dim=0).unsqueeze(0)

    batch = {
        "observation.images.agentview": agent,
        "observation.images.wrist": wrist,
        "observation.state": state,
        "task": [LANGUAGE],
    }

    batch = add_language_tokens(batch, policy)
    batch = move_batch_to_device(batch, device)

    return batch


def select_action(policy, batch):
    with torch.no_grad():
        pred = policy.select_action(batch)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if pred.ndim == 3:
        pred = pred[0, 0]
    elif pred.ndim == 2:
        pred = pred[0]

    return pred


def ckpt_sort_key(p):
    if p.name == "checkpoint_final.pt":
        return 10**18
    m = re.search(r"step_(\d+)", p.name)
    return int(m.group(1)) if m else 0


def main():
    device = torch.device("cuda")

    ckpts = sorted(CKPT_DIR.glob("checkpoint_step_*.pt"), key=ckpt_sort_key)
    ckpts.append(CKPT_DIR / "checkpoint_final.pt")

    rows = []

    with h5py.File(DATASET_PATH, "r") as f:
        demo = f["data/demo_0"]
        num_samples = min(64, demo["actions"].shape[0])

        for ckpt_path in ckpts:
            print("=" * 80)
            print("checkpoint:", ckpt_path)

            policy = build_policy(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(ckpt["model_state_dict"])
            policy.eval()

            preds = []
            trues = []

            for i in range(num_samples):
                batch = make_batch(demo, i, device, policy)
                pred = select_action(policy, batch)
                true = np.asarray(demo["actions"][i])

                preds.append(pred)
                trues.append(true)

            preds = np.asarray(preds)
            trues = np.asarray(trues)

            abs_err = np.abs(preds - trues)
            sq_err = (preds - trues) ** 2

            row = {
                "checkpoint": ckpt_path.name,
                "mse": float(sq_err.mean()),
                "mae": float(abs_err.mean()),
            }

            for j in range(7):
                row[f"action_{j}_mae"] = float(abs_err[:, j].mean())
                row[f"action_{j}_mse"] = float(sq_err[:, j].mean())

            rows.append(row)

            print("MAE:", row["mae"])
            print("action_0 MAE:", row["action_0_mae"])
            print("action_2 MAE:", row["action_2_mae"])
            print("action_6 MAE:", row["action_6_mae"])

            del policy
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=" * 80)
    print("SAVED TO:", OUT_CSV)
    print("=" * 80)

    cols = [
        "checkpoint",
        "mae",
        "mse",
        "action_0_mae",
        "action_1_mae",
        "action_2_mae",
        "action_6_mae",
    ]
    print(df[cols].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
