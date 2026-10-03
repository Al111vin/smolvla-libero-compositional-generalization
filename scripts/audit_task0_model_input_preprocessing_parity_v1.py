#!/usr/bin/env python3
"""Compare matched train-loader and evaluation inputs before the VLA model.

Uses the real LeRobot dataset class/config, saved training preprocessor, and
evaluation frame builder semantics on identical HDF5-matched observations.
Diagnostic only: no policy model, environment, optimizer, or dataset writes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy_config, make_pre_post_processors

from eval_v3_task0 import image_to_tensor


HDF5_TO_FEATURE = {
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


def source_state(obs: h5py.Group, frame: int) -> np.ndarray:
    values = [np.asarray(obs[key][frame], dtype=np.float32) for key in OBS_KEYS]
    return np.concatenate(values).astype(np.float32)


def frame_indices(length: int, limit: int | None) -> list[int]:
    standard = [0, 100, 250, 450, length - 1]
    result = sorted(set(index for index in standard if 0 <= index < length))
    return result if limit is None else result[:limit]


def canonical_tensor(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Remove the expected singleton observation-time dimension for comparison."""
    if key.startswith("observation.images.") and tensor.ndim == 5 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if key == "observation.state" and tensor.ndim == 3 and tensor.shape[1] == 1:
        return tensor[:, 0]
    return tensor


def tensor_or_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    return value


def compare_values(key: str, train_value: Any, eval_value: Any) -> dict[str, Any]:
    train_value = tensor_or_value(train_value)
    eval_value = tensor_or_value(eval_value)
    if isinstance(train_value, torch.Tensor) and isinstance(eval_value, torch.Tensor):
        train_shape = list(train_value.shape)
        eval_shape = list(eval_value.shape)
        train_canonical = canonical_tensor(key, train_value)
        eval_canonical = canonical_tensor(key, eval_value)
        if train_canonical.shape != eval_canonical.shape:
            return {
                "equal": False,
                "train_shape": train_shape,
                "eval_shape": eval_shape,
                "train_canonical_shape": list(train_canonical.shape),
                "eval_canonical_shape": list(eval_canonical.shape),
                "reason": "canonical tensor shapes differ",
            }
        difference = (train_canonical.to(torch.float64) - eval_canonical.to(torch.float64)).abs()
        exact = bool(torch.equal(train_canonical, eval_canonical))
        return {
            "equal": exact,
            "train_shape": train_shape,
            "eval_shape": eval_shape,
            "canonical_shape": list(train_canonical.shape),
            "dtype_train": str(train_value.dtype),
            "dtype_eval": str(eval_value.dtype),
            "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
            "mean_abs_diff": float(difference.mean().item()) if difference.numel() else 0.0,
            "train_min": float(train_canonical.min().item()) if train_canonical.numel() else None,
            "train_max": float(train_canonical.max().item()) if train_canonical.numel() else None,
        }
    train_json = json.dumps(train_value, sort_keys=True, default=str)
    eval_json = json.dumps(eval_value, sort_keys=True, default=str)
    return {
        "equal": train_json == eval_json,
        "train_type": type(train_value).__name__,
        "eval_type": type(eval_value).__name__,
        "train_value": train_value if isinstance(train_value, (str, int, float, bool, type(None))) else str(train_value),
        "eval_value": eval_value if isinstance(eval_value, (str, int, float, bool, type(None))) else str(eval_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--limit-frames-per-episode", type=int)
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"REFUSE_OVERWRITE={args.output_json}")
    if args.limit_frames_per_episode is not None and args.limit_frames_per_episode < 1:
        raise ValueError("limit-frames-per-episode must be >= 1")
    checkpoint_file = args.checkpoint / "model.safetensors"
    train_config_file = args.checkpoint / "train_config.json"
    preprocessor_file = args.checkpoint / "policy_preprocessor.json"
    for path in (checkpoint_file, train_config_file, preprocessor_file, args.hdf5):
        if not path.is_file():
            raise FileNotFoundError(path)
    hdf5_hash = sha256(args.hdf5)
    if hdf5_hash != args.expected_hdf5_sha256:
        raise AssertionError(f"unexpected HDF5 sha256: {hdf5_hash}")

    train_config = json.loads(train_config_file.read_text(encoding="utf-8"))
    dataset_cfg = train_config["dataset"]
    instruction = "pick up the akita black bowl and place it on the plate in the left region"
    selected_episodes = [0, 1, 2, 3, 4]
    config_path = args.checkpoint / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = config_payload.pop("type")
    policy_cfg = make_policy_config(policy_type, **config_payload)
    dataset_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, dataset_meta)
    if dataset_cfg.get("image_transforms", {}).get("enable", False):
        raise AssertionError("this audit expects the StageE1 no-image-augmentation pipeline")
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=dataset_cfg.get("revision"),
        video_backend=dataset_cfg.get("video_backend"),
        tolerance_s=float(train_config.get("tolerance_s", 0.0001)),
    )
    expected_task = dataset.meta.tasks.iloc[0].name
    if expected_task != instruction:
        raise AssertionError(f"task text mismatch: {expected_task!r} != {instruction!r}")
    if dataset.num_episodes != 5 or dataset.num_frames != 3290:
        raise AssertionError(
            f"unexpected subset size: episodes={dataset.num_episodes}, frames={dataset.num_frames}"
        )
    preprocess, _postprocess = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    checks: list[dict[str, Any]] = []
    raw_mismatches = 0
    processed_mismatches = 0
    with h5py.File(args.hdf5, "r") as source:
        cursor = 0
        for episode_id, demo_name in enumerate(sorted(source["data"].keys())):
            demo = source[f"data/{demo_name}"]
            obs = demo["obs"]
            length = len(demo["actions"])
            selected_frames = frame_indices(length, args.limit_frames_per_episode)
            for frame in selected_frames:
                dataset_index = cursor + frame
                train_item = dataset[dataset_index]
                observed_episode = int(torch.as_tensor(train_item["episode_index"]).item())
                observed_frame = int(torch.as_tensor(train_item["frame_index"]).item())
                if observed_episode != episode_id or observed_frame != frame:
                    raise AssertionError(
                        f"dataset row mapping mismatch: expected {(episode_id, frame)}, got {(observed_episode, observed_frame)}"
                    )

                eval_frame: dict[str, Any] = {
                    "observation.state": torch.from_numpy(source_state(obs, frame)),
                    "task": instruction,
                }
                train_inputs: dict[str, Any] = {"task": train_item["task"]}
                for camera, (feature, h5_key) in HDF5_TO_FEATURE.items():
                    source_image = np.asarray(obs[h5_key][frame], dtype=np.uint8)
                    eval_image = image_to_tensor(source_image)
                    train_image = train_item[feature]
                    if not isinstance(train_image, torch.Tensor):
                        train_image = torch.as_tensor(train_image)
                    train_image_canonical = train_image[0] if train_image.ndim == 4 and train_image.shape[0] == 1 else train_image
                    eval_image_canonical = eval_image
                    if train_image_canonical.shape != eval_image_canonical.shape:
                        raise AssertionError(
                            f"{demo_name}/{frame}/{camera}: loader/eval image shapes differ: {tuple(train_image_canonical.shape)} vs {tuple(eval_image_canonical.shape)}"
                        )
                    raw_equal = torch.equal(train_image_canonical, eval_image_canonical)
                    raw_mismatches += int(not raw_equal)
                    train_inputs[feature] = train_image
                    eval_frame[feature] = eval_image

                train_state = train_item["observation.state"]
                if not isinstance(train_state, torch.Tensor):
                    train_state = torch.as_tensor(train_state)
                train_state_canonical = train_state[0] if train_state.ndim == 2 and train_state.shape[0] == 1 else train_state
                eval_state = eval_frame["observation.state"]
                raw_state_equal = torch.equal(train_state_canonical, eval_state)
                raw_mismatches += int(not raw_state_equal)
                train_inputs["observation.state"] = train_state

                # Match the actual training collation: one sample becomes B=1.
                train_batch = {
                    key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value])
                    for key, value in train_inputs.items()
                }
                with torch.inference_mode():
                    train_processed = preprocess(copy.deepcopy(train_batch))
                    eval_processed = preprocess(copy.deepcopy(eval_frame))
                common_keys = sorted(set(train_processed).intersection(eval_processed))
                details = {
                    key: compare_values(key, train_processed[key], eval_processed[key])
                    for key in common_keys
                }
                key_sets_equal = set(train_processed) == set(eval_processed)
                unequal_keys = [key for key, detail in details.items() if not detail["equal"]]
                processed_mismatches += int(bool(unequal_keys) or not key_sets_equal)
                checks.append(
                    {
                        "episode_index": episode_id,
                        "hdf5_demo": demo_name,
                        "frame_index": frame,
                        "dataset_task": train_item["task"],
                        "raw_images_and_state_exact": raw_equal and raw_state_equal,
                        "training_processor_keys": sorted(train_processed),
                        "evaluation_processor_keys": sorted(eval_processed),
                        "processor_key_sets_equal": key_sets_equal,
                        "processed_model_input_comparison": details,
                        "unequal_processed_keys": unequal_keys,
                    }
                )
            cursor += length

    all_raw_exact = raw_mismatches == 0
    all_processed_exact = processed_mismatches == 0
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "formal_benchmark_evidence": False,
        "training_or_checkpoint_modified": False,
        "dataset_modified": False,
        "checkpoint_sha256": sha256(checkpoint_file),
        "hdf5_sha256": hdf5_hash,
        "policy_preprocessor_json_sha256": sha256(preprocessor_file),
        "dataset_repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "training_image_transforms_enabled": bool(dataset_cfg.get("image_transforms", {}).get("enable", False)),
        "training_use_imagenet_stats": bool(dataset_cfg.get("use_imagenet_stats", False)),
        "policy_visual_normalization": policy_cfg.normalization_mapping.get("VISUAL"),
        "policy_state_normalization": policy_cfg.normalization_mapping.get("STATE"),
        "delta_timestamps_resolved_by_official_factory": delta_timestamps,
        "samples_per_episode": None if args.limit_frames_per_episode is None else args.limit_frames_per_episode,
        "sampled_frames_per_episode": sorted({item["frame_index"] for item in checks}),
        "matched_observation_count": len(checks),
        "raw_loader_vs_eval_inputs_exact": all_raw_exact,
        "raw_input_mismatch_count": raw_mismatches,
        "processed_model_inputs_exact": all_processed_exact,
        "processed_input_mismatch_count": processed_mismatches,
        "mapping_method": "task0 LeRobot episode ids 0-4 and frame ids are verified from returned row fields; full HDF5-to-dataset action/state/pixel parity was previously completed",
        "checks": checks,
        "interpretation_limit": "This compares saved StageE1 training preprocessor applied to actual LeRobotDataset items against the evaluation frame builder on the same recorded observations. It does not run the model or test simulator-generated observations outside the demonstration states.",
        "fold02": "LOCKED",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
