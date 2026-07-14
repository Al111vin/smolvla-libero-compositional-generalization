from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_libero_hdf5_to_lerobot import (
    create_dataset,
    read_language,
    sorted_demo_keys,
    validate_demo,
)


NUMERIC_KEYS = [
    "actions",
    "obs/joint_states",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
]

EXPECTED_TAIL_SHAPES = {
    "actions": (7,),
    "obs/agentview_rgb": (128, 128, 3),
    "obs/eye_in_hand_rgb": (128, 128, 3),
    "obs/joint_states": (7,),
    "obs/ee_pos": (3,),
    "obs/ee_ori": (3,),
    "obs/gripper_states": (2,),
}


@dataclass
class TaskSource:
    task_id: int
    language: str
    hdf5_path: Path
    demo_keys: list[str]
    selected_demo_keys: list[str]
    selected_frames: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LIBERO task split into one independent "
            "LeRobot training dataset."
        )
    )
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--expected-task-ids",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--expected-demos-per-task",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Use only the first N demos per task for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value) -> str:
    return " ".join(str(value).strip().split())


def decode_json_attribute(raw, name: str):
    if raw is None:
        raise KeyError(f"Missing HDF5 attribute: {name}")

    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")

    return json.loads(str(raw))


def is_true(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_split(args) -> pd.DataFrame:
    split_path = Path(args.split_csv)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split CSV does not exist: {split_path}"
        )

    df = pd.read_csv(split_path)

    required_columns = {
        "task_id",
        "language",
        "hdf5_path",
        "eligible_for_primary_loco",
        "exclusion_reason",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Split CSV is missing columns: {sorted(missing)}"
        )

    df["task_id"] = df["task_id"].astype(int)
    df = df.sort_values("task_id").reset_index(drop=True)

    actual_ids = df["task_id"].tolist()
    expected_ids = sorted(args.expected_task_ids)

    if actual_ids != expected_ids:
        raise ValueError(
            f"Task IDs do not match.\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {actual_ids}"
        )

    if df["task_id"].duplicated().any():
        raise ValueError("Duplicate task IDs in split CSV.")

    if df["hdf5_path"].duplicated().any():
        raise ValueError("Duplicate HDF5 paths in split CSV.")

    invalid_eligibility = [
        int(row.task_id)
        for row in df.itertuples()
        if not is_true(row.eligible_for_primary_loco)
    ]

    if invalid_eligibility:
        raise ValueError(
            "Ineligible tasks entered the training split: "
            f"{invalid_eligibility}"
        )

    reasons = df["exclusion_reason"].fillna("").astype(str).str.strip()

    if reasons.ne("").any():
        bad_ids = df.loc[reasons.ne(""), "task_id"].tolist()
        raise ValueError(
            f"Tasks with exclusion reasons entered split: {bad_ids}"
        )

    forbidden = sorted(set(actual_ids) & {1, 5})

    if forbidden:
        raise ValueError(
            f"Primary LOCO datasets must exclude tasks 1 and 5: "
            f"{forbidden}"
        )

    return df


def preflight_task(row, args) -> TaskSource:
    task_id = int(row["task_id"])
    csv_language = normalize_text(row["language"])
    hdf5_path = Path(str(row["hdf5_path"]))

    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"Task {task_id}: missing HDF5 file: {hdf5_path}"
        )

    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise KeyError(
                f"Task {task_id}: HDF5 has no data group."
            )

        data_group = hdf5_file["data"]
        hdf5_language = normalize_text(
            read_language(data_group)
        )

        if hdf5_language != csv_language:
            raise ValueError(
                f"Task {task_id}: language mismatch.\n"
                f"CSV:  {csv_language}\n"
                f"HDF5: {hdf5_language}"
            )

        env_args = decode_json_attribute(
            data_group.attrs.get("env_args"),
            "data/env_args",
        )

        control_freq = int(
            env_args["env_kwargs"]["control_freq"]
        )

        if control_freq != args.fps:
            raise ValueError(
                f"Task {task_id}: HDF5 control_freq is "
                f"{control_freq}, requested fps is {args.fps}"
            )

        demo_keys = sorted_demo_keys(data_group)

        if len(demo_keys) != args.expected_demos_per_task:
            raise ValueError(
                f"Task {task_id}: expected "
                f"{args.expected_demos_per_task} demos, "
                f"found {len(demo_keys)}"
            )

        metadata_num_demos = data_group.attrs.get("num_demos")

        if (
            metadata_num_demos is not None
            and int(metadata_num_demos) != len(demo_keys)
        ):
            raise ValueError(
                f"Task {task_id}: num_demos metadata mismatch."
            )

        frame_counts = {}
        full_frame_total = 0

        for demo_key in demo_keys:
            demo = data_group[demo_key]
            num_frames = validate_demo(demo, demo_key)

            for key, tail_shape in EXPECTED_TAIL_SHAPES.items():
                expected_shape = (num_frames, *tail_shape)
                actual_shape = tuple(demo[key].shape)

                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Task {task_id} {demo_key}: "
                        f"{key} shape {actual_shape}, "
                        f"expected {expected_shape}"
                    )

            for image_key in [
                "obs/agentview_rgb",
                "obs/eye_in_hand_rgb",
            ]:
                if demo[image_key].dtype != np.uint8:
                    raise ValueError(
                        f"Task {task_id} {demo_key}: "
                        f"{image_key} must be uint8, "
                        f"got {demo[image_key].dtype}"
                    )

            for key in NUMERIC_KEYS:
                values = np.asarray(demo[key])

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Task {task_id} {demo_key}: "
                        f"{key} contains non-finite values"
                    )

            frame_counts[demo_key] = num_frames
            full_frame_total += num_frames

        metadata_total = data_group.attrs.get("total")

        if (
            metadata_total is not None
            and int(metadata_total) != full_frame_total
        ):
            raise ValueError(
                f"Task {task_id}: total frame metadata is "
                f"{int(metadata_total)}, calculated "
                f"{full_frame_total}"
            )

    limit = args.max_demos_per_task

    if limit is None:
        selected_demo_keys = demo_keys
    else:
        if not 1 <= limit <= len(demo_keys):
            raise ValueError(
                f"--max-demos-per-task must be in "
                f"1..{len(demo_keys)}"
            )
        selected_demo_keys = demo_keys[:limit]

    selected_frames = sum(
        frame_counts[key] for key in selected_demo_keys
    )

    return TaskSource(
        task_id=task_id,
        language=hdf5_language,
        hdf5_path=hdf5_path,
        demo_keys=demo_keys,
        selected_demo_keys=selected_demo_keys,
        selected_frames=selected_frames,
    )


def make_frame(demo, frame_index: int, language: str):
    joint = np.asarray(
        demo["obs/joint_states"][frame_index],
        dtype=np.float32,
    )
    ee_pos = np.asarray(
        demo["obs/ee_pos"][frame_index],
        dtype=np.float32,
    )
    ee_ori = np.asarray(
        demo["obs/ee_ori"][frame_index],
        dtype=np.float32,
    )
    gripper = np.asarray(
        demo["obs/gripper_states"][frame_index],
        dtype=np.float32,
    )

    state = np.concatenate(
        [joint, ee_pos, ee_ori, gripper],
        axis=0,
    ).astype(np.float32)

    if state.shape != (15,):
        raise ValueError(
            f"State shape is {state.shape}, expected (15,)"
        )

    return {
        "observation.images.agentview": np.asarray(
            demo["obs/agentview_rgb"][frame_index],
            dtype=np.uint8,
        ),
        "observation.images.wrist": np.asarray(
            demo["obs/eye_in_hand_rgb"][frame_index],
            dtype=np.uint8,
        ),
        "observation.state": state,
        "action": np.asarray(
            demo["actions"][frame_index],
            dtype=np.float32,
        ),
        "task": language,
    }


def validate_created_dataset(
    repo_id: str,
    output_path: Path,
    expected_episodes: int,
    expected_frames: int,
    expected_tasks: int,
    expected_languages: set[str],
    fps: int,
):
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_path,
    )

    if dataset.num_episodes != expected_episodes:
        raise ValueError(
            f"Reloaded episode count is {dataset.num_episodes}, "
            f"expected {expected_episodes}"
        )

    if len(dataset) != expected_frames:
        raise ValueError(
            f"Reloaded frame count is {len(dataset)}, "
            f"expected {expected_frames}"
        )

    if dataset.fps != fps:
        raise ValueError(
            f"Reloaded fps is {dataset.fps}, expected {fps}"
        )

    if dataset.meta.total_tasks != expected_tasks:
        raise ValueError(
            f"Reloaded task count is "
            f"{dataset.meta.total_tasks}, "
            f"expected {expected_tasks}"
        )

    actual_languages = {
        normalize_text(value)
        for value in dataset.meta.tasks.index.tolist()
    }

    if actual_languages != expected_languages:
        raise ValueError(
            "Reloaded task-language set does not match split."
        )

    for feature in [
        "observation.state",
        "action",
    ]:
        if feature not in dataset.meta.stats:
            raise KeyError(
                f"Missing statistics for {feature}"
            )

        for statistic, value in dataset.meta.stats[feature].items():
            array = np.asarray(value)

            if (
                array.dtype.kind in "biufc"
                and not np.isfinite(array).all()
            ):
                raise ValueError(
                    f"Non-finite stat: {feature}/{statistic}"
                )

    sample = dataset[0]

    expected_sample_shapes = {
        "observation.images.agentview": (3, 128, 128),
        "observation.images.wrist": (3, 128, 128),
        "observation.state": (15,),
        "action": (7,),
    }

    for key, expected_shape in expected_sample_shapes.items():
        actual_shape = tuple(sample[key].shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Reloaded {key} shape is {actual_shape}, "
                f"expected {expected_shape}"
            )


def main():
    args = parse_args()
    split_path = Path(args.split_csv)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    dangerous_paths = {
        Path("/").resolve(),
        Path(".").resolve(),
        Path.home().resolve(),
    }

    if output_path.resolve() in dangerous_paths:
        raise ValueError(
            f"Refusing dangerous output path: {output_path}"
        )

    if output_path.resolve() in manifest_path.resolve().parents:
        raise ValueError(
            "Manifest must be outside the LeRobot dataset root."
        )

    split_df = load_split(args)

    print("=" * 80)
    print("Running complete preflight before writing output")

    task_sources = []

    for _, row in split_df.iterrows():
        source = preflight_task(row, args)
        task_sources.append(source)

        print(
            f"task {source.task_id}: "
            f"{len(source.selected_demo_keys)} episodes, "
            f"{source.selected_frames} frames"
        )

    expected_episodes = sum(
        len(source.selected_demo_keys)
        for source in task_sources
    )
    expected_frames = sum(
        source.selected_frames
        for source in task_sources
    )

    print("Expected episodes:", expected_episodes)
    print("Expected frames:", expected_frames)
    print("Task IDs:", [source.task_id for source in task_sources])

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Use --overwrite to rebuild it."
            )

        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(
        repo_id=args.repo_id,
        output=output_path,
        fps=args.fps,
    )

    manifest_rows = []
    total_frames = 0

    try:
        for source in task_sources:
            with h5py.File(source.hdf5_path, "r") as hdf5_file:
                data_group = hdf5_file["data"]

                for demo_key in source.selected_demo_keys:
                    demo = data_group[demo_key]
                    num_frames = demo["actions"].shape[0]
                    global_episode_index = len(manifest_rows)

                    for frame_index in range(num_frames):
                        dataset.add_frame(
                            make_frame(
                                demo,
                                frame_index,
                                source.language,
                            )
                        )
                        total_frames += 1

                    dataset.save_episode(
                        parallel_encoding=False
                    )

                    lerobot_task_index = int(
                        dataset.meta.get_task_index(
                            source.language
                        )
                    )

                    demo_index = int(
                        demo_key.replace("demo_", "")
                    )

                    manifest_rows.append(
                        {
                            "global_episode_index":
                                global_episode_index,
                            "lerobot_task_index":
                                lerobot_task_index,
                            "libero_suite":
                                "libero_spatial",
                            "libero_task_id":
                                source.task_id,
                            "demo_index":
                                demo_index,
                            "demo_key":
                                demo_key,
                            "num_frames":
                                num_frames,
                            "language":
                                source.language,
                            "hdf5_path":
                                str(source.hdf5_path),
                            "split_csv":
                                str(split_path),
                            "repo_id":
                                args.repo_id,
                            "fps":
                                args.fps,
                        }
                    )

                    print(
                        f"[{len(manifest_rows):03d}/"
                        f"{expected_episodes:03d}] "
                        f"task={source.task_id} "
                        f"{demo_key} frames={num_frames}"
                    )

        if hasattr(dataset, "finalize"):
            dataset.finalize()

    except Exception:
        if (
            hasattr(dataset, "has_pending_frames")
            and dataset.has_pending_frames()
        ):
            dataset.clear_episode_buffer()

        if hasattr(dataset, "finalize"):
            dataset.finalize()

        print(
            "Conversion failed. Output may be partial: "
            f"{output_path}"
        )
        raise

    if len(manifest_rows) != expected_episodes:
        raise ValueError(
            f"Converted {len(manifest_rows)} episodes, "
            f"expected {expected_episodes}"
        )

    if total_frames != expected_frames:
        raise ValueError(
            f"Converted {total_frames} frames, "
            f"expected {expected_frames}"
        )

    validate_created_dataset(
        repo_id=args.repo_id,
        output_path=output_path,
        expected_episodes=expected_episodes,
        expected_frames=expected_frames,
        expected_tasks=len(task_sources),
        expected_languages={
            source.language for source in task_sources
        },
        fps=args.fps,
    )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    manifest_df.to_csv(
        temporary_manifest,
        index=False,
        lineterminator="\n",
    )
    temporary_manifest.replace(manifest_path)

    print("=" * 80)
    print("Conversion and reload validation passed")
    print("output:", output_path)
    print("manifest:", manifest_path)
    print("tasks:", len(task_sources))
    print("episodes:", expected_episodes)
    print("frames:", expected_frames)


if __name__ == "__main__":
    main()
