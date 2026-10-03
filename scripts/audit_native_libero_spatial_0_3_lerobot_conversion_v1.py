"""Read-only sampled parity audit for native Spatial tasks 0-3 conversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_native_libero_spatial_0_3_to_lerobot_v1 import TASKS, sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Conversion manifest is not complete")
    for source in manifest["tasks"]:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    expected_episodes = sum(task["episodes"] for task in manifest["tasks"])
    expected_frames = sum(task["frames"] for task in manifest["tasks"])
    if dataset.num_episodes != expected_episodes or len(dataset) != expected_frames:
        raise ValueError("LeRobot episode/frame counts differ from conversion manifest")
    if dataset.meta.total_tasks != len(TASKS):
        raise ValueError(f"Expected four tasks, got {dataset.meta.total_tasks}")

    global_offsets = {}
    cursor = 0
    for task_id in TASKS:
        global_offsets[task_id] = cursor
        task = manifest["tasks"][task_id]
        cursor += task["frames"]

    checks = []
    for task_id in TASKS:
        task = manifest["tasks"][task_id]
        episode_details = task["episodes_detail"]
        selected_episode_indices = sorted({0, len(episode_details) // 2, len(episode_details) - 1})
        with h5py.File(args.source_dir / TASKS[task_id]["filename"], "r") as source_file:
            data = source_file["data"]
            for episode_index in selected_episode_indices:
                episode = episode_details[episode_index]
                demo = data[episode["demo_key"]]
                episode_offset = sum(item["frames"] for item in episode_details[:episode_index])
                frame_indices = sorted({0, episode["frames"] // 2, episode["frames"] - 1})
                expected_task_index = int(manifest["task_index_by_libero_id"][str(task_id)])
                for frame_index in frame_indices:
                    row_index = global_offsets[task_id] + episode_offset + frame_index
                    item = dataset[row_index]
                    state_expected = np.concatenate([
                        np.asarray(demo["obs/joint_states"][frame_index], dtype=np.float32),
                        np.asarray(demo["obs/ee_pos"][frame_index], dtype=np.float32),
                        np.asarray(demo["obs/ee_ori"][frame_index], dtype=np.float32),
                        np.asarray(demo["obs/gripper_states"][frame_index], dtype=np.float32),
                    ])
                    state_actual = np.asarray(item["observation.state"], dtype=np.float32)
                    action_expected = np.asarray(demo["actions"][frame_index], dtype=np.float32)
                    action_actual = np.asarray(item["action"], dtype=np.float32)
                    state_error = float(np.max(np.abs(state_actual - state_expected)))
                    action_error = float(np.max(np.abs(action_actual - action_expected)))
                    if state_error != 0.0 or action_error != 0.0:
                        raise ValueError(f"State/action parity error task={task_id} {episode['demo_key']} frame={frame_index}: {state_error}/{action_error}")

                    image_errors = {}
                    for feature, source_key in (
                        ("observation.images.agentview", "obs/agentview_rgb"),
                        ("observation.images.wrist", "obs/eye_in_hand_rgb"),
                    ):
                        actual = item[feature]
                        if isinstance(actual, torch.Tensor):
                            actual = actual.detach().cpu()
                            if actual.dtype.is_floating_point:
                                actual = (actual * 255.0).round().to(torch.uint8)
                            actual = actual.permute(1, 2, 0).numpy()
                        else:
                            actual = np.asarray(actual)
                            if actual.shape == (3, 128, 128):
                                actual = actual.transpose(1, 2, 0)
                        expected = np.asarray(demo[source_key][frame_index], dtype=np.uint8)
                        if actual.shape != expected.shape:
                            raise ValueError(f"Image shape mismatch {feature}: {actual.shape} vs {expected.shape}")
                        error = int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))
                        if error != 0:
                            raise ValueError(f"Image pixel mismatch task={task_id} {episode['demo_key']} frame={frame_index} key={source_key}: max={error}")
                        image_errors[feature] = error

                    task_index = int(item["task_index"].item() if hasattr(item["task_index"], "item") else item["task_index"])
                    if task_index != expected_task_index:
                        raise ValueError(f"Task-index mismatch task={task_id}: {task_index} != {expected_task_index}")
                    checks.append({
                        "libero_task_id": task_id,
                        "demo_key": episode["demo_key"],
                        "frame_index": frame_index,
                        "state_max_abs_diff": state_error,
                        "action_max_abs_diff": action_error,
                        "image_max_abs_diff": image_errors,
                        "task_index": task_index,
                    })

    result = {
        "status": "passed",
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "task_count": dataset.meta.total_tasks,
        "sampled_frames": len(checks),
        "sample_design": "tasks 0-3 x episodes 0/25/49 x first/middle/last frame",
        "all_state_action_image_checks_exact": True,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "episodes", "frames", "task_count", "sampled_frames", "all_state_action_image_checks_exact")}, indent=2))


if __name__ == "__main__":
    main()
