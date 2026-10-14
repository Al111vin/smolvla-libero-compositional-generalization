#!/usr/bin/env python3
"""CPU-only, fail-closed preflight for the registered mixed-batch diagnostic."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import draccus
import torch
import lerobot.scripts.lerobot_train  # register SmolVLA config before decode
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from scripts.lerobot_loco_episode_compat import install_episode_filter_compat


RUN_ID = "teacher_native_spatial_4task_mixbatch8_freshseed2001_40k_v1_20261014"
REPO_ID = "local/libero_spatial_tasks0_3_native_v1"
EXPECTED_MANIFEST_SHA256 = "8bb81e1d1828d2a4be3dd8469e1f2812c35bc4f9a48aea0f7a8da8deb2d11ba1"
EXPECTED_MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
EXPECTED_TRAINER_SHA256 = "e08ba26017674ac8453683e7b54810325fb381f20cb32a161c5d9aa690483449"
EXPECTED_SINGLE_SHA256 = "fabb744b04083ee643e3f7db34798e465f069379fcc1d57146d5dc70aaa43f0b"
EXPECTED_JOINT_SHA256 = "2ced9539c1ec358b47ce6ae20a2db979088f3e5bfc3e8399405bc2a5d2546ac3"
EXPECTED_CONFIG_SHA256 = "7d19527dc45b63c4325da9500bd2eb9964ec4d3cd136d2547bba2864ce1e6e91"
EXPECTED_SAMPLER_SHA256 = "22aeca407667bdf009021a4856ad8262de2365a13e8c970f720c890b3ccdfd29"
EXPECTED_WRAPPER_SHA256 = "a20e97e105807b0ada5c9c8035ae4e1dbc7e7c7c110d8c8d6667d8231d2d2800"
EXPECTED_TEST_SHA256 = "3d3eeabf551fec57e0d33d703af96ce35ef8412476082c6a11f2677d5705379b"
EXPECTED_AUDIT_SHA256 = "e1a79bd46e6f37da1468fd116c356b5554d4c1dce9c9010c5c9cc36b397d1b7d"
EXPECTED_EVALUATOR_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"
EXPECTED_OUTPUT = Path(
    "/root/smolvla-training-prep/results/training/" + RUN_ID
)
EXPECTED_EVAL_OUTPUT = Path(
    "/root/smolvla-eval-prep/results/"
    "teacher_native_spatial_4task_mixbatch8_freshseed2001_40k_eval_v1_20261014"
)
EXPECTED_RECOVERY = Path(
    "/root/smolvla-training-prep/recovery/"
    "teacher_native_spatial_4task_mixbatch8_freshseed2001_v1_20261014"
)
EXPECTED_EVAL_RECOVERY = Path(
    "/root/smolvla-eval-prep/recovery/"
    "teacher_native_spatial_4task_mixbatch8_freshseed2001_eval_v1_20261014"
)
LOCK_PATH = Path("/root/smolvla-training-prep/teacher_control_gpu.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_absent(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to reuse registered path: {path}")


def build_lr_trace(config_path: Path) -> list[list[float]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = draccus.decode(TrainPipelineConfig, raw)
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
    optimizer = cfg.optimizer.build([parameter])
    scheduler = cfg.scheduler.build(optimizer, 40000)
    trace: list[list[float]] = []
    for _ in range(40000):
        trace.append([float(group["lr"]) for group in optimizer.param_groups])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()  # one scalar CPU-only schedule probe; no policy/model
        scheduler.step()
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--single-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--full-parity-audit", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--candidate-wrapper", type=Path, required=True)
    parser.add_argument("--candidate-sampler", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Preflight must run with CUDA_VISIBLE_DEVICES empty")
    assert_absent([args.output])

    expected_hashes = {
        "single_config": EXPECTED_SINGLE_SHA256,
        "joint_config": EXPECTED_JOINT_SHA256,
        "candidate_config": EXPECTED_CONFIG_SHA256,
        "candidate_sampler": EXPECTED_SAMPLER_SHA256,
        "candidate_wrapper": EXPECTED_WRAPPER_SHA256,
        "candidate_test": EXPECTED_TEST_SHA256,
        "full_parity_audit_script": EXPECTED_AUDIT_SHA256,
        "evaluator": EXPECTED_EVALUATOR_SHA256,
    }
    actual_hashes = {
        "single_config": sha256(args.single_config),
        "joint_config": sha256(args.joint_config),
        "candidate_config": sha256(args.config),
        "candidate_sampler": sha256(args.candidate_sampler),
        "candidate_wrapper": sha256(args.candidate_wrapper),
        "candidate_test": sha256(args.candidate_test),
        "full_parity_audit_script": sha256(args.audit_script),
        "evaluator": sha256(args.evaluator),
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"Pinned hash mismatch: {actual_hashes}")
    import lerobot.scripts.lerobot_train as installed_trainer
    trainer_path = Path(inspect.getsourcefile(installed_trainer))
    trainer_hash = sha256(trainer_path)
    if trainer_hash != EXPECTED_TRAINER_SHA256:
        raise ValueError(f"Installed trainer source hash mismatch: {trainer_hash}")

    manifest_hash = sha256(args.conversion_manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Conversion manifest hash mismatch: {manifest_hash}")
    manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.full_parity_audit.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("repo_id") != REPO_ID:
        raise ValueError("Frozen conversion manifest status/repo mismatch")
    if audit.get("status") != "passed" or audit.get("manifest_sha256") != manifest_hash:
        raise ValueError("Previously passed full-frame audit is not bound to this manifest")
    if (audit.get("frames") != 22709 or audit.get("images") != 45418
            or audit.get("episodes") != 200
            or audit.get("all_state_action_image_checks_exact") is not True
            or audit.get("all_episode_frame_task_indices_exact") is not True):
        raise ValueError("Full-frame parity receipt does not satisfy frozen counts/checks")

    single_raw = json.loads(args.single_config.read_text(encoding="utf-8"))
    joint_raw = json.loads(args.joint_config.read_text(encoding="utf-8"))
    candidate_raw = json.loads(args.config.read_text(encoding="utf-8"))
    allowed = {"steps", "batch_size", "job_name", "output_dir"}
    changed = {key for key in joint_raw if joint_raw[key] != candidate_raw[key]}
    if changed != allowed:
        raise ValueError(f"Candidate top-level changes differ from registration: {changed}")
    if (candidate_raw["job_name"] != RUN_ID or candidate_raw["steps"] != 40000
            or candidate_raw["batch_size"] != 8 or candidate_raw["seed"] != 2001
            or candidate_raw["output_dir"] != str(EXPECTED_OUTPUT)
            or candidate_raw["resume"] is not False
            or candidate_raw["checkpoint_path"] is not None):
        raise ValueError("Candidate config does not match registered fresh-start run")
    if (candidate_raw["dataset"]["repo_id"] != REPO_ID
            or candidate_raw["dataset"]["root"] != manifest["output"]
            or candidate_raw["dataset"]["episodes"] != list(range(200))
            or candidate_raw["optimizer"] != single_raw["optimizer"]
            or candidate_raw["scheduler"] != single_raw["scheduler"]):
        raise ValueError("Dataset, optimizer, scheduler, or episode mapping differs")

    data_root = Path(candidate_raw["dataset"]["root"])
    model_root = Path(candidate_raw["policy"]["pretrained_path"])
    model_file = model_root / "model.safetensors"
    if not data_root.is_dir() or not (data_root / "meta/info.json").is_file():
        raise FileNotFoundError("Frozen LeRobot dataset or metadata missing")
    if not model_file.is_file() or sha256(model_file) != EXPECTED_MODEL_SHA256:
        raise ValueError("Frozen base model hash mismatch")

    training_prefix = EXPECTED_RECOVERY / RUN_ID
    training_external = [Path(str(training_prefix) + suffix) for suffix in
                         (".training.log", ".launcher.pid", ".exit_code")]
    evaluation_prefix = EXPECTED_EVAL_RECOVERY / RUN_ID
    evaluation_external = [Path(str(evaluation_prefix) + suffix) for suffix in
                           (".evaluation.log", ".launcher.pid", ".exit_code")]
    assert_absent([
        EXPECTED_OUTPUT, EXPECTED_EVAL_OUTPUT,
        Path(str(EXPECTED_RECOVERY / RUN_ID) + ".launch_manifest.json"),
        EXPECTED_EVAL_RECOVERY / "evaluation_manifest.json", *training_external,
        *evaluation_external,
    ])
    if (not LOCK_PATH.is_file() or LOCK_PATH.is_symlink()):
        raise RuntimeError("Registered production lock is missing or not a regular file")
    with LOCK_PATH.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)

    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        try:
            with open(f"/proc/{parent}/stat", encoding="utf-8") as stream:
                parent = int(stream.read().split()[3])
        except (OSError, ValueError, IndexError):
            break
    process_lines = subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()[1:]
    conflicts = []
    for line in process_lines:
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) in ancestors:
            continue
        if any(token in fields[1] for token in
               ("lerobot_train", "task_balanced", "teacher_native_spatial")):
            conflicts.append(line)
    if conflicts:
        raise RuntimeError(f"Matching process is present: {conflicts[:8]}")
    gpu_pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader"], text=True
    ).strip()
    gpu_fields = [part.strip() for part in gpu_info.split(",")]
    try:
        gpu_used_mib = int(gpu_fields[2].split()[0])
        gpu_idle = (gpu_fields[0] == "NVIDIA GeForce RTX 5090"
                    and gpu_fields[1] == "0 %" and gpu_used_mib <= 128)
    except (IndexError, ValueError):
        gpu_idle = False
    if gpu_pids or not gpu_idle:
        raise RuntimeError(f"GPU is not at the registered idle state: {gpu_info}; pids={gpu_pids}")
    disk = shutil.disk_usage("/root")
    free_gib = disk.free / (1024 ** 3)
    if free_gib < 20:
        raise RuntimeError(f"Insufficient disk headroom: {free_gib:.2f} GiB")

    # Decode actual train configs and compare the pre-optimizer LR vector using
    # the installed optimizer/scheduler classes, on a scalar CPU parameter.
    single_trace = build_lr_trace(args.single_config)
    candidate_trace = build_lr_trace(args.config)
    if len(single_trace) != 40000 or len(candidate_trace) != 40000 or single_trace != candidate_trace:
        raise ValueError("Actual installed 40k scheduler vectors differ")

    install_episode_filter_compat()
    cfg = draccus.decode(TrainPipelineConfig, candidate_raw)
    cfg.validate()
    dataset = make_dataset(cfg)
    if (len(dataset) != 22709 or dataset.num_episodes != 200
            or dataset.meta.total_tasks != 4):
        raise ValueError("Installed trainer dataset factory returned unexpected task subset")
    sys.path.insert(0, str(args.candidate_sampler.resolve().parent))
    from task_balanced_batch_sampler_v2 import install_task_balanced_dataloader_patch
    loader_cls, original_init = install_task_balanced_dataloader_patch(
        target_repo_id=REPO_ID, seed=2001, samples_per_task=80000,
        samples_per_task_per_batch=2, expected_tasks=(0, 1, 2, 3),
        expected_batch_size=8,
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        loader = Accelerator(cpu=True).prepare(loader)
        for i, batch in enumerate(loader):
            values = [int(x) for x in batch["task_index"].reshape(-1).tolist()]
            counts = {task: values.count(task) for task in range(4)}
            if counts != {0: 2, 1: 2, 2: 2, 3: 2}:
                raise ValueError(f"Actual Accelerate-wrapped loader batch {i} quota mismatch: {counts}")
            for key in ("action", "observation.state"):
                if key not in batch or not bool(torch.isfinite(batch[key]).all()):
                    raise ValueError(f"Missing/nonfinite batch field: {key}")
            if i == 31:
                break
        if i != 31:
            raise ValueError("Actual loader yielded fewer than 32 CPU batches")
        sampler = loader.batch_sampler.sampler
        if len(sampler) != 320000 or sampler.num_batches != 40000:
            raise ValueError("Actual task-balanced sampler size/update budget mismatch")
    finally:
        loader_cls.__init__ = original_init

    report = {
        "schema_version": 1,
        "status": "cpu_only_preflight_passed_not_launched",
        "recorded_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "experiment_id": RUN_ID,
        "gpu_visibility_disabled": True,
        "model_loaded": False,
        "policy_forward_called": False,
        "optimizer_updates": 0,
        "training_started": False,
        "evaluation_started": False,
        "pinned_hashes_match": True,
        "conversion_manifest_sha256": manifest_hash,
        "full_frame_audit": {"frames": 22709, "images": 45418, "episodes": 200, "passed": True},
        "base_model_sha256": EXPECTED_MODEL_SHA256,
        "installed_trainer_sha256": trainer_hash,
        "actual_scheduler": {"positions": 40000, "single_candidate_exact_match": True},
        "loader": {"episodes": 200, "frames": 22709, "tasks": 4,
                   "accelerate_cpu_batches_checked": 32, "quota_each_batch": [2, 2, 2, 2],
                   "sampler_indices": len(sampler), "sampler_batches": sampler.num_batches},
        "resources": {"gpu": gpu_info, "gpu_processes": [], "disk_free_gib": round(free_gib, 2),
                      "shared_lock_probe": "acquired_nonblocking"},
        "registered_paths_absent": True,
        "changed_config_keys_from_joint": sorted(changed),
        "ready_for_immediate_final_guard": True,
        "limitations": ["CPU sampler/data-loader checks do not prove CUDA training, checkpoint persistence, or evaluation outcome."],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
