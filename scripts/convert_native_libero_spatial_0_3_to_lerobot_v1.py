"""Convert official LIBERO Spatial tasks 0-3 into one new LeRobot dataset.

This converter is deliberately separate from the LOCO split converter: native
Spatial task ids 0-3 are valid here, and all existing outputs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASKS = {
    0: {
        "filename": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "language": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    1: {
        "filename": "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "language": "pick up the black bowl next to the ramekin and place it on the plate",
    },
    2: {
        "filename": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        "language": "pick up the black bowl from table center and place it on the plate",
    },
    3: {
        "filename": "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        "language": "pick up the black bowl on the cookie box and place it on the plate",
    },
}

STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "joint_6", "ee_x", "ee_y", "ee_z", "ee_ori_0", "ee_ori_1",
    "ee_ori_2", "gripper_0", "gripper_1",
]
ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z", "delta_rot_x", "delta_rot_y",
    "delta_rot_z", "gripper",
]
REQUIRED = (
    "actions", "obs/agentview_rgb", "obs/eye_in_hand_rgb",
    "obs/joint_states", "obs/ee_pos", "obs/ee_ori",
    "obs/gripper_states",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--expected-episodes-per-task", type=int, default=50)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value) -> str:
    return " ".join(str(value).strip().split())


def read_language(data_group) -> str:
    raw = data_group.attrs.get("problem_info")
    if raw is None:
        raise KeyError("Missing data/problem_info")
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    info = json.loads(str(raw))
    language = info["language_instruction"]
    if isinstance(language, list):
        language = " ".join(language)
    return normalize_text(language)


def sorted_demo_keys(group) -> list[str]:
    return sorted(group.keys(), key=lambda key: int(key.removeprefix("demo_")))


def validate_source(task_id: int, path: Path, expected_episodes: int, fps: int):
    expected_language = TASKS[task_id]["language"]
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise KeyError(f"Task {task_id}: missing data group")
        group = file["data"]
        language = read_language(group)
        if language != expected_language:
            raise ValueError(f"Task {task_id}: unexpected language {language!r}")
        env_args_raw = group.attrs.get("env_args")
        if env_args_raw is None:
            raise KeyError(f"Task {task_id}: missing env_args")
        if isinstance(env_args_raw, (bytes, np.bytes_)):
            env_args_raw = env_args_raw.decode("utf-8")
        env_args = json.loads(str(env_args_raw))
        control_freq = int(env_args["env_kwargs"]["control_freq"])
        if control_freq != fps:
            raise ValueError(f"Task {task_id}: control_freq {control_freq} != fps {fps}")
        demo_keys = sorted_demo_keys(group)
        if len(demo_keys) != expected_episodes:
            raise ValueError(f"Task {task_id}: {len(demo_keys)} episodes, expected {expected_episodes}")
        frames = 0
        per_episode = []
        for key in demo_keys:
            demo = group[key]
            for field in REQUIRED:
                if field not in demo:
                    raise KeyError(f"Task {task_id}/{key}: missing {field}")
            n = int(demo["actions"].shape[0])
            expected_shapes = {
                "actions": (n, 7),
                "obs/agentview_rgb": (n, 128, 128, 3),
                "obs/eye_in_hand_rgb": (n, 128, 128, 3),
                "obs/joint_states": (n, 7),
                "obs/ee_pos": (n, 3),
                "obs/ee_ori": (n, 3),
                "obs/gripper_states": (n, 2),
            }
            for field, shape in expected_shapes.items():
                if tuple(demo[field].shape) != shape:
                    raise ValueError(f"Task {task_id}/{key}: {field} shape {demo[field].shape}, expected {shape}")
                if field.endswith("_rgb"):
                    if demo[field].dtype != np.uint8:
                        raise ValueError(f"Task {task_id}/{key}: {field} must be uint8")
                elif not np.isfinite(np.asarray(demo[field])).all():
                    raise ValueError(f"Task {task_id}/{key}: non-finite values in {field}")
            frames += n
            per_episode.append({"demo_key": key, "frames": n})
        recorded_total = group.attrs.get("total")
        if recorded_total is not None and int(recorded_total) != frames:
            raise ValueError(f"Task {task_id}: total metadata mismatch")
    return {
        "task_id": task_id,
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "language": expected_language,
        "episodes": len(demo_keys),
        "frames": frames,
        "episodes_detail": per_episode,
    }


def make_frame(demo, i: int, language: str):
    state = np.concatenate([
        np.asarray(demo["obs/joint_states"][i], dtype=np.float32),
        np.asarray(demo["obs/ee_pos"][i], dtype=np.float32),
        np.asarray(demo["obs/ee_ori"][i], dtype=np.float32),
        np.asarray(demo["obs/gripper_states"][i], dtype=np.float32),
    ])
    action = np.asarray(demo["actions"][i], dtype=np.float32)
    if state.shape != (15,) or action.shape != (7,):
        raise ValueError(f"Unexpected frame state/action shapes: {state.shape}/{action.shape}")
    return {
        "observation.images.agentview": np.asarray(demo["obs/agentview_rgb"][i], dtype=np.uint8),
        "observation.images.wrist": np.asarray(demo["obs/eye_in_hand_rgb"][i], dtype=np.uint8),
        "observation.state": state,
        "action": action,
        "task": language,
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    if output.resolve() in {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"Refusing unsafe output path: {output}")
    if manifest_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Manifest must be outside output dataset directory")

    # Complete read-only preflight before creating any output.
    sources = []
    for task_id, entry in TASKS.items():
        path = source_dir / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(validate_source(task_id, path, args.expected_episodes_per_task, args.fps))
    expected_frames = sum(source["frames"] for source in sources)
    if args.preflight_only:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({
                "status": "preflight_passed",
                "repo_id": args.repo_id,
                "fps": args.fps,
                "tasks": sources,
                "episode_count": sum(source["episodes"] for source in sources),
                "frame_count": expected_frames,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "preflight_passed", "episode_count": sum(source["episodes"] for source in sources), "frame_count": expected_frames}, indent=2))
        return
    task_frame_offsets = {}
    running_offset = 0
    for source in sources:
        task_frame_offsets[source["task_id"]] = running_offset
        running_offset += source["frames"]
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features={
            "observation.images.agentview": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (15,), "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        },
        root=output,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )

    episodes_manifest = []
    total_frames = 0
    try:
        for source in sources:
            with h5py.File(source["path"], "r") as file:
                group = file["data"]
                for detail in source["episodes_detail"]:
                    demo = group[detail["demo_key"]]
                    for i in range(detail["frames"]):
                        dataset.add_frame(make_frame(demo, i, source["language"]))
                    dataset.save_episode(parallel_encoding=False)
                    task_index = int(dataset.meta.get_task_index(source["language"]))
                    episodes_manifest.append({
                        "episode_index": len(episodes_manifest),
                        "libero_spatial_task_id": source["task_id"],
                        "lerobot_task_index": task_index,
                        "demo_key": detail["demo_key"],
                        "frames": detail["frames"],
                        "language": source["language"],
                    })
                    total_frames += detail["frames"]
                    print(f"task={source['task_id']} {detail['demo_key']} frames={detail['frames']}", flush=True)
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    except Exception:
        if hasattr(dataset, "has_pending_frames") and dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        raise

    if total_frames != expected_frames or len(episodes_manifest) != 4 * args.expected_episodes_per_task:
        raise RuntimeError("Final frame/episode count does not match preflight")
    reloaded = LeRobotDataset(repo_id=args.repo_id, root=output)
    if reloaded.num_episodes != len(episodes_manifest) or len(reloaded) != expected_frames:
        raise RuntimeError("Reloaded LeRobot counts do not match conversion manifest")
    if reloaded.meta.total_tasks != 4:
        raise RuntimeError(f"Expected 4 task strings, found {reloaded.meta.total_tasks}")
    task_ids = {int(row["lerobot_task_index"]) for row in episodes_manifest}
    if len(task_ids) != 4:
        raise RuntimeError(f"Expected 4 LeRobot task ids, found {sorted(task_ids)}")

    # Check action round-trip against a few points from each source episode.
    parity_checks = []
    parity_rows = []
    for task_id in TASKS:
        task_rows = [row for row in episodes_manifest if row["libero_spatial_task_id"] == task_id]
        parity_rows.extend(task_rows[index] for index in sorted({0, len(task_rows) // 2, len(task_rows) - 1}))
    for row in parity_rows:
        source = next(item for item in sources if item["task_id"] == row["libero_spatial_task_id"])
        demo_index = int(row["demo_key"].removeprefix("demo_"))
        start = task_frame_offsets[row["libero_spatial_task_id"]] + sum(
            ep["frames"] for ep in source["episodes_detail"][:demo_index]
        )
        with h5py.File(source["path"], "r") as file:
            demo = file["data"][row["demo_key"]]
            for offset in sorted({0, row["frames"] // 2, row["frames"] - 1}):
                converted = np.asarray(reloaded[start + offset]["action"], dtype=np.float32)
                original = np.asarray(demo["actions"][offset], dtype=np.float32)
                error = float(np.max(np.abs(converted - original)))
                if error != 0.0:
                    raise RuntimeError(f"Action round-trip mismatch at task {row['libero_spatial_task_id']} {row['demo_key']} frame {offset}: {error}")
                parity_checks.append({"task_id": row["libero_spatial_task_id"], "demo_key": row["demo_key"], "frame": offset, "max_abs_diff": error})

    result = {
        "status": "complete",
        "repo_id": args.repo_id,
        "output": str(output),
        "fps": args.fps,
        "tasks": sources,
        "episodes": episodes_manifest,
        "episode_count": len(episodes_manifest),
        "frame_count": expected_frames,
        "task_index_by_libero_id": {
            str(task): int(next(row["lerobot_task_index"] for row in episodes_manifest if row["libero_spatial_task_id"] == task))
            for task in TASKS
        },
        "sampled_action_round_trip_checks": parity_checks,
        "max_abs_action_diff": 0.0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "episode_count", "frame_count", "task_index_by_libero_id", "max_abs_action_diff")}, indent=2))


if __name__ == "__main__":
    main()
