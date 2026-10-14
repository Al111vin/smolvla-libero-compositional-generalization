"""Build, but do not write or launch, the registered seed-2001 task1 config."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/teacher_native_spatial_task0_freshseed2001_40k_batch2_v1_20261011.json"
EXPECTED_SOURCE_SHA256 = "fabb744b04083ee643e3f7db34798e465f069379fcc1d57146d5dc70aaa43f0b"
RUN_ID = "teacher_native_spatial_task1_freshseed2001_40k_batch2_v1_20261014"
REMOTE_OUTPUT = f"/root/smolvla-training-prep/results/training/{RUN_ID}"
EXPECTED_EPISODES = list(range(50, 100))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        changes: set[str] = set()
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                changes.add(path)
            else:
                changes.update(changed_paths(left[key], right[key], path))
        return changes
    return set() if left == right else {prefix}


def build_config(source: dict) -> dict:
    if source.get("seed") != 2001 or source.get("steps") != 40000 or source.get("batch_size") != 2:
        raise ValueError("source is not the registered seed-2001 task0 40k/batch-2 recipe")
    if source.get("resume") is not False or source.get("checkpoint_path") is not None:
        raise ValueError("task1 control must start from the pretrained base without resume")
    if source["dataset"].get("episodes") != list(range(50)):
        raise ValueError("source must be the 50-episode task0 slice")
    if source["dataset"].get("repo_id") != "local/libero_spatial_tasks0_3_native_v1":
        raise ValueError("unexpected native Spatial dataset repository")
    if source["dataset"].get("root") != "/root/smolvla-training-prep/datasets/lerobot/libero_spatial_tasks0_3_native_20261003_v1":
        raise ValueError("unexpected converted dataset root")
    if source["policy"].get("pretrained_path") != "/root/smolvla-training-prep/models/smolvla_base_libero":
        raise ValueError("unexpected pretrained base model")
    if source["policy"].get("optimizer_lr") != 1e-4:
        raise ValueError("unexpected policy learning rate")

    config = copy.deepcopy(source)
    config["dataset"]["episodes"] = EXPECTED_EPISODES.copy()
    config["job_name"] = RUN_ID
    config["output_dir"] = REMOTE_OUTPUT
    allowed = {"dataset.episodes", "job_name", "output_dir"}
    if changed_paths(source, config) != allowed:
        raise ValueError("generated config changed fields outside the registered scope")
    return config


def load_and_build() -> dict:
    source_bytes = SOURCE.read_bytes()
    if sha256(source_bytes) != EXPECTED_SOURCE_SHA256:
        raise ValueError("frozen task0 seed-2001 source config hash mismatch")
    return build_config(json.loads(source_bytes))


if __name__ == "__main__":
    print(json.dumps(load_and_build(), indent=2, ensure_ascii=False))
