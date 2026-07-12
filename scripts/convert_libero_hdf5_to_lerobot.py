from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_INPUT = (
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_OUTPUT = "datasets/lerobot/libero_spatial_task0"
DEFAULT_REPO_ID = "Al111vin/libero_spatial_task0"


STATE_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_ori_0",
    "ee_ori_1",
    "ee_ori_2",
    "gripper_0",
    "gripper_1",
]

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rot_x",
    "delta_rot_y",
    "delta_rot_z",
    "gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one LIBERO HDF5 task to LeRobotDataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory if it already exists.",
    )
    return parser.parse_args()


def sorted_demo_keys(data_group):
    return sorted(
        data_group.keys(),
        key=lambda name: int(name.replace("demo_", "")),
    )


def read_language(data_group) -> str:
    raw = data_group.attrs.get("problem_info")
    if raw is None:
        raise KeyError("Missing data/problem_info attribute.")

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    info = json.loads(raw)
    language = info["language_instruction"]

    if isinstance(language, list):
        language = " ".join(language)

    return str(language).strip().strip('"')


def validate_demo(demo, demo_key: str):
    required = [
        "actions",
        "obs/agentview_rgb",
        "obs/eye_in_hand_rgb",
        "obs/joint_states",
        "obs/ee_pos",
        "obs/ee_ori",
        "obs/gripper_states",
    ]

    for key in required:
        if key not in demo:
            raise KeyError(f"{demo_key}: missing field {key}")

    expected_length = demo["actions"].shape[0]

    for key in required[1:]:
        actual_length = demo[key].shape[0]
        if actual_length != expected_length:
            raise ValueError(
                f"{demo_key}: {key} has {actual_length} frames, "
                f"expected {expected_length}"
            )

    if demo["actions"].shape[1:] != (7,):
        raise ValueError(
            f"{demo_key}: expected action shape (*, 7), "
            f"got {demo['actions'].shape}"
        )

    if demo["obs/agentview_rgb"].shape[1:] != (128, 128, 3):
        raise ValueError(
            f"{demo_key}: unexpected agentview shape "
            f"{demo['obs/agentview_rgb'].shape}"
        )

    if demo["obs/eye_in_hand_rgb"].shape[1:] != (128, 128, 3):
        raise ValueError(
            f"{demo_key}: unexpected wrist image shape "
            f"{demo['obs/eye_in_hand_rgb'].shape}"
        )

    return expected_length


def create_dataset(repo_id: str, output: Path, fps: int):
    features = {
        "observation.images.agentview": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (15,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 does not exist: {input_path}")

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Use --overwrite only if you want to delete and rebuild it."
            )
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(
        repo_id=args.repo_id,
        output=output_path,
        fps=args.fps,
    )

    total_frames = 0

    with h5py.File(input_path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        language = read_language(data_group)
        demo_keys = sorted_demo_keys(data_group)

        print("Input:", input_path)
        print("Output:", output_path)
        print("Repo ID:", args.repo_id)
        print("FPS:", args.fps)
        print("Language:", language)
        print("Episodes:", len(demo_keys))

        for episode_index, demo_key in enumerate(demo_keys):
            demo = data_group[demo_key]
            num_frames = validate_demo(demo, demo_key)

            for frame_index in range(num_frames):
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
                        f"{demo_key} frame {frame_index}: "
                        f"state shape is {state.shape}, expected (15,)"
                    )

                frame = {
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

                dataset.add_frame(frame)
                total_frames += 1

            dataset.save_episode(parallel_encoding=False)

            print(
                f"Saved episode {episode_index + 1:02d}/{len(demo_keys)} "
                f"({demo_key}, {num_frames} frames)"
            )

    # LeRobot v3 requires finalize(); keep compatibility if API changes.
    if hasattr(dataset, "finalize"):
        dataset.finalize()

    print("=" * 80)
    print("Conversion completed")
    print("episodes:", len(demo_keys))
    print("frames:", total_frames)
    print("output:", output_path)

    if len(demo_keys) != 50:
        raise ValueError(f"Expected 50 episodes, got {len(demo_keys)}")

    if total_frames != 5068:
        raise ValueError(f"Expected 5068 frames, got {total_frames}")


if __name__ == "__main__":
    main()