#!/usr/bin/env python3
"""Full read-only parity audit for task-0 episodes embedded in the frozen 0-3 dataset."""
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
SOURCE_SHA256 = "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256(args.source_hdf5) != SOURCE_SHA256:
        raise ValueError("Official task0 HDF5 SHA256 mismatch")

    parquet_files = sorted((args.dataset_root / "data").rglob("*.parquet"))
    all_rows = []
    for path in parquet_files:
        all_rows.extend(pq.read_table(path).to_pylist())
    rows = [row for row in all_rows if 0 <= int(row["episode_index"]) < 50]
    rows.sort(key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))

    state_max = action_max = 0.0
    images_checked = 0
    errors: list[str] = []
    with h5py.File(args.source_hdf5, "r") as source:
        demos = source["data"]
        keys = sorted(demos.keys(), key=lambda key: int(key.removeprefix("demo_")))
        expected_frames = sum(len(demos[key]["actions"]) for key in keys)
        if len(keys) != 50 or expected_frames != 5068 or len(rows) != expected_frames:
            errors.append(f"count_mismatch demos={len(keys)} expected_frames={expected_frames} selected_rows={len(rows)}")
        raw_lang = json.loads(demos.attrs["problem_info"])["language_instruction"]
        language = " ".join(raw_lang) if isinstance(raw_lang, list) else str(raw_lang)
        language = language.strip().strip('"')
        if language != TASK_TEXT:
            errors.append("source_language_mismatch")

        for row in rows:
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            demo = demos[keys[episode]]
            state = np.concatenate(
                [demo["obs/" + field][frame] for field in ("joint_states", "ee_pos", "ee_ori", "gripper_states")]
            ).astype(np.float32)
            action = demo["actions"][frame].astype(np.float32)
            state_diff = float(np.max(np.abs(np.asarray(row["observation.state"], dtype=np.float32) - state)))
            action_diff = float(np.max(np.abs(np.asarray(row["action"], dtype=np.float32) - action)))
            state_max = max(state_max, state_diff)
            action_max = max(action_max, action_diff)
            if state_diff or action_diff:
                errors.append(f"numeric_mismatch:{episode}:{frame}")
            if int(row["task_index"]) != 0:
                errors.append(f"task_mapping_mismatch:{episode}:{frame}")
            for converted, raw in (("agentview", "agentview_rgb"), ("wrist", "eye_in_hand_rgb")):
                image = row["observation.images." + converted]
                if image.get("bytes") is not None:
                    decoded = Image.open(io.BytesIO(image["bytes"]))
                else:
                    decoded = Image.open(args.dataset_root / image["path"])
                if not np.array_equal(np.asarray(decoded), demo["obs/" + raw][frame]):
                    errors.append(f"image_mismatch:{episode}:{frame}:{converted}")
                images_checked += 1

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=list(range(50)),
        delta_timestamps={"action": [i / 20 for i in range(50)]},
    )
    if len(dataset) != 5068 or dataset.num_episodes != 50:
        errors.append(f"filtered_loader_count_mismatch:{len(dataset)}:{dataset.num_episodes}")
    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    sample_checks = []
    for index in sample_indices:
        sample = dataset[index]
        shapes = {key: list(sample[key].shape) for key in ("action", "observation.state", "observation.images.agentview", "observation.images.wrist")}
        finite = all(bool(sample[key].isfinite().all()) for key in shapes)
        if sample["task"] != TASK_TEXT or int(sample["task_index"]) != 0:
            errors.append(f"filtered_loader_wrong_task:{index}")
        if shapes != {
            "action": [50, 7],
            "observation.state": [15],
            "observation.images.agentview": [3, 128, 128],
            "observation.images.wrist": [3, 128, 128],
        } or not finite:
            errors.append(f"filtered_loader_schema_or_finite:{index}")
        sample_checks.append({"index": index, "shapes": shapes, "finite": finite})

    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "source_hdf5": str(args.source_hdf5),
        "source_sha256": SOURCE_SHA256,
        "dataset_repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "dataset_stats_sha256": sha256(args.dataset_root / "meta/stats.json"),
        "task_id": 0,
        "task_text": TASK_TEXT,
        "episodes": 50,
        "frames": len(rows),
        "frames_checked": len(rows),
        "images_checked": images_checked,
        "state_max_abs_diff_float32": state_max,
        "action_max_abs_diff_float32": action_max,
        "filtered_loader_length": len(dataset),
        "filtered_loader_episodes": dataset.num_episodes,
        "filtered_loader_samples": sample_checks,
        "errors": errors,
        "training_started": False,
        "fold02": "LOCKED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
