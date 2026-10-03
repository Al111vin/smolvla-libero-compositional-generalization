#!/usr/bin/env python3
"""Exhaustively compare task-0 source HDF5 images with the frozen LeRobot rows.

The episode mapping is derived from exact action and state sequence equality,
not file order. This is a read-only data audit; it does not load a policy,
create an environment, train, or modify the dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


CAMERAS = {
    "agentview": ("observation.images.agentview", "agentview_rgb"),
    "wrist": ("observation.images.wrist", "eye_in_hand_rgb"),
}
OBS_KEYS = ("joint_states", "ee_pos", "ee_ori", "gripper_states")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_state(obs: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[key], dtype=np.float32) for key in OBS_KEYS], axis=1
    )


def read_task_zero_episode_index(dataset_root: Path) -> dict[int, list[Path]]:
    files_by_episode: dict[int, list[Path]] = {}
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path, columns=["episode_index", "task_index"])
        episode_ids = table.column("episode_index").to_pylist()
        task_ids = table.column("task_index").to_pylist()
        for episode_id, task_id in zip(episode_ids, task_ids, strict=True):
            if int(task_id) == 0:
                files_by_episode.setdefault(int(episode_id), []).append(parquet_path)
    return files_by_episode


def source_episodes(hdf5_path: Path) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(hdf5_path, "r") as source:
        for name in sorted(source["data"].keys()):
            demo = source[f"data/{name}"]
            obs = demo["obs"]
            result[name] = {
                "actions": np.asarray(demo["actions"], dtype=np.float32),
                "state": source_state(obs),
            }
    return result


def read_lerobot_episode(
    parquet_paths: list[Path], episode_id: int
) -> tuple[np.ndarray, np.ndarray, dict[str, list[dict[str, object]]]]:
    columns = [
        "episode_index",
        "task_index",
        "frame_index",
        "action",
        "observation.state",
        *[feature for feature, _ in CAMERAS.values()],
    ]
    tables = [
        pq.read_table(
            path,
            columns=columns,
            filters=[("episode_index", "=", episode_id), ("task_index", "=", 0)],
        )
        for path in sorted(set(parquet_paths))
    ]
    tables = [table for table in tables if table.num_rows]
    if not tables:
        raise AssertionError(f"no rows read for task_index=0 episode_index={episode_id}")
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    order = np.argsort(frame_indices, kind="stable")
    if not np.array_equal(frame_indices[order], np.arange(len(order), dtype=np.int64)):
        raise AssertionError(f"episode {episode_id}: frame indices are not contiguous from zero")
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[order]
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[order]
    images: dict[str, list[dict[str, object]]] = {}
    for camera, (feature, _) in CAMERAS.items():
        values = table.column(feature).to_pylist()
        images[camera] = [values[int(index)] for index in order]
    return actions, states, images


def decode_image(value: dict[str, object], camera: str, frame: int) -> np.ndarray:
    blob = value.get("bytes")
    if not blob:
        raise ValueError(
            f"{camera} frame {frame}: inline image bytes absent; path={value.get('path')!r}"
        )
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def compare_camera(
    hdf5_path: Path,
    demo_name: str,
    camera: str,
    image_records: list[dict[str, object]],
    limit_frames: int | None,
) -> dict[str, object]:
    feature, h5_key = CAMERAS[camera]
    del feature
    with h5py.File(hdf5_path, "r") as source:
        source_images = source[f"data/{demo_name}/obs/{h5_key}"]
        count = len(image_records) if limit_frames is None else min(limit_frames, len(image_records))
        if len(source_images) != len(image_records):
            raise AssertionError(
                f"{demo_name}/{camera}: source has {len(source_images)} frames, LeRobot has {len(image_records)}"
            )
        exact_frames = 0
        different_frames = 0
        different_channel_values = 0
        total_channel_values = 0
        max_abs_diff = 0
        absolute_error_sum = 0
        first_difference: dict[str, object] | None = None
        for frame_index in range(count):
            source_image = np.asarray(source_images[frame_index], dtype=np.uint8)
            decoded = decode_image(image_records[frame_index], camera, frame_index)
            if decoded.shape != source_image.shape:
                raise AssertionError(
                    f"{demo_name}/{camera}/{frame_index}: shape mismatch {decoded.shape} vs {source_image.shape}"
                )
            difference = np.abs(decoded.astype(np.int16) - source_image.astype(np.int16))
            frame_different_values = int(np.count_nonzero(difference))
            total_channel_values += int(difference.size)
            different_channel_values += frame_different_values
            absolute_error_sum += int(difference.sum())
            frame_max = int(difference.max(initial=0))
            max_abs_diff = max(max_abs_diff, frame_max)
            if frame_different_values == 0:
                exact_frames += 1
            else:
                different_frames += 1
                if first_difference is None:
                    first_difference = {
                        "frame_index": frame_index,
                        "different_channel_values": frame_different_values,
                        "max_abs_diff": frame_max,
                        "mean_abs_diff": float(difference.mean()),
                    }
        return {
            "frames_compared": count,
            "frames_exact_pixel_match": exact_frames,
            "frames_with_any_pixel_difference": different_frames,
            "different_channel_values": different_channel_values,
            "total_channel_values": total_channel_values,
            "different_channel_value_fraction": (
                float(different_channel_values / total_channel_values)
                if total_channel_values
                else 0.0
            ),
            "mean_absolute_pixel_difference": (
                float(absolute_error_sum / total_channel_values)
                if total_channel_values
                else 0.0
            ),
            "max_abs_pixel_difference": max_abs_diff,
            "first_difference": first_difference,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--limit-frames-per-episode", type=int)
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_json}")
    if args.limit_frames_per_episode is not None and args.limit_frames_per_episode < 1:
        raise ValueError("limit-frames-per-episode must be >= 1")
    hdf5_hash = sha256(args.hdf5)
    if hdf5_hash != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 sha256: {hdf5_hash}")
    info_path = args.dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "task_index" not in info.get("features", {}):
        raise AssertionError("frozen dataset has no task_index feature")

    source = source_episodes(args.hdf5)
    files_by_episode = read_task_zero_episode_index(args.dataset_root)
    if len(source) != 5 or len(files_by_episode) != 5:
        raise AssertionError(
            f"expected exactly five HDF5 and task-index-0 episodes; got {len(source)} and {len(files_by_episode)}"
        )

    matched_source_demos: set[str] = set()
    episode_results: list[dict[str, object]] = []
    for episode_id in sorted(files_by_episode):
        actions, states, images = read_lerobot_episode(files_by_episode[episode_id], episode_id)
        matches = [
            name
            for name, demo in source.items()
            if name not in matched_source_demos
            and np.array_equal(actions, demo["actions"])
            and np.array_equal(states, demo["state"])
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"task_index=0 episode_index={episode_id}: expected unique exact action/state source match, got {matches}"
            )
        demo_name = matches[0]
        matched_source_demos.add(demo_name)
        if len(actions) != 658:
            raise AssertionError(f"{demo_name}: expected 658 frames, got {len(actions)}")
        camera_results = {
            camera: compare_camera(
                args.hdf5,
                demo_name,
                camera,
                image_records,
                args.limit_frames_per_episode,
            )
            for camera, image_records in images.items()
        }
        episode_results.append(
            {
                "task_index": 0,
                "episode_index": episode_id,
                "matched_hdf5_demo": demo_name,
                "frames": len(actions),
                "action_exact": True,
                "state_exact": True,
                "camera_results": camera_results,
            }
        )

    all_complete = args.limit_frames_per_episode is None
    total_frames_compared = sum(
        int(camera["frames_compared"])
        for episode in episode_results
        for camera in episode["camera_results"].values()
    )
    total_frame_camera_pairs = 5 * 658 * len(CAMERAS)
    all_exact = all(
        camera["frames_exact_pixel_match"] == camera["frames_compared"]
        for episode in episode_results
        for camera in episode["camera_results"].values()
    )
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "formal_benchmark_evidence": False,
        "training_or_dataset_modified": False,
        "audit_complete": all_complete and total_frames_compared == total_frame_camera_pairs,
        "task_index": 0,
        "hdf5_sha256": hdf5_hash,
        "dataset_format": info.get("codebase_version"),
        "dataset_total_episodes": info.get("total_episodes"),
        "dataset_total_frames": info.get("total_frames"),
        "source_episode_count": len(source),
        "task_zero_episode_count": len(files_by_episode),
        "source_frames_per_episode": {name: int(item["actions"].shape[0]) for name, item in source.items()},
        "mapping_method": "for each task_index=0 episode, require exact full-sequence equality of action and concatenated 15-D state against exactly one unused HDF5 demo; no file-order assumption",
        "pixel_parity_all_compared_frames": all_exact,
        "total_camera_frame_pairs_compared": total_frames_compared,
        "expected_full_camera_frame_pairs": total_frame_camera_pairs,
        "limit_frames_per_episode": args.limit_frames_per_episode,
        "episodes": episode_results,
        "interpretation_limit": "This tests source-HDF5 to frozen-LeRobot decoded image parity. It does not by itself establish the exact image tensor seen after the training/evaluation preprocessing pipeline or explain closed-loop failure.",
        "fold02": "LOCKED",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
