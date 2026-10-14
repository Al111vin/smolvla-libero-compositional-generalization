"""Build one immutable config for the registered 4-task mixed-batch diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "configs/teacher_native_spatial_task0_freshseed2001_40k_batch2_v1_20261011.json"
JOINT = ROOT / "configs/teacher_native_spatial_4task_freshseed2001_160k_batch2_v1_20261011.json"
OUT = ROOT / "configs/teacher_native_spatial_4task_mixbatch8_freshseed2001_40k_v1_20261014.json"

EXPECTED_SINGLE_SHA256 = "fabb744b04083ee643e3f7db34798e465f069379fcc1d57146d5dc70aaa43f0b"
EXPECTED_JOINT_SHA256 = "2ced9539c1ec358b47ce6ae20a2db979088f3e5bfc3e8399405bc2a5d2546ac3"
RUN_ID = "teacher_native_spatial_4task_mixbatch8_freshseed2001_40k_v1_20261014"
REMOTE_OUTPUT = f"/root/smolvla-training-prep/results/training/{RUN_ID}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config() -> dict:
    if sha256(SINGLE) != EXPECTED_SINGLE_SHA256:
        raise ValueError("registered single-task source config hash mismatch")
    if sha256(JOINT) != EXPECTED_JOINT_SHA256:
        raise ValueError("registered joint source config hash mismatch")
    single = json.loads(SINGLE.read_text(encoding="utf-8"))
    config = json.loads(JOINT.read_text(encoding="utf-8"))
    if config["seed"] != single["seed"] or config["seed"] != 2001:
        raise ValueError("training seed is not the registered matched seed")
    if config["scheduler"] != single["scheduler"]:
        raise ValueError("per-task scheduler differs from the frozen single baseline")
    if config["optimizer"] != single["optimizer"]:
        raise ValueError("optimizer differs from the frozen single baseline")
    if config["dataset"]["repo_id"] != "local/libero_spatial_tasks0_3_native_v1":
        raise ValueError("unexpected target dataset repo id")
    if len(config["dataset"]["episodes"]) != 200:
        raise ValueError("expected 50 frozen episodes for each of four tasks")
    if config["resume"] or config["checkpoint_path"] is not None:
        raise ValueError("diagnostic must start from the pretrained base, not resume")

    config["steps"] = 40000
    config["batch_size"] = 8
    config["job_name"] = RUN_ID
    config["output_dir"] = REMOTE_OUTPUT
    return config


def write_config(destination: Path = OUT) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    config = build_config()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return config


if __name__ == "__main__":
    result = write_config()
    print(json.dumps({"path": str(OUT), "sha256": sha256(OUT),
                      "steps": result["steps"], "batch_size": result["batch_size"],
                      "seed": result["seed"]}, sort_keys=True))
