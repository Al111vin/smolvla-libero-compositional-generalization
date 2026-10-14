#!/usr/bin/env python3
"""V3 CPU preflight uses the actual temporal shape emitted by make_dataset."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import draccus
import lerobot.scripts.lerobot_train  # register SmolVLA before config decode
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset

RUN_ID = "teacher_native_spatial_task1_freshseed2001_40k_batch2_v1_20261014"
REPO_ID = "local/libero_spatial_tasks0_3_native_v1"
DATA_ROOT = Path("/root/smolvla-training-prep/datasets/lerobot/libero_spatial_tasks0_3_native_20261003_v1")
MANIFEST = Path("/root/smolvla-training-prep/recovery/task_scaling_native_spatial_20261003/conversion_manifest.json")
MODEL = Path("/root/smolvla-training-prep/models/smolvla_base_libero/model.safetensors")
LOCK = Path("/root/smolvla-training-prep/teacher_control_gpu.lock")
EXPECTED_SOURCE_SHA256 = "fabb744b04083ee643e3f7db34798e465f069379fcc1d57146d5dc70aaa43f0b"
EXPECTED_CONFIG_SHA256 = "150269259fbc2711a6aacb3beb8f5a20abc24f744ffac08dc988bf16d62446de"
EXPECTED_MANIFEST_SHA256 = "8bb81e1d1828d2a4be3dd8469e1f2812c35bc4f9a48aea0f7a8da8deb2d11ba1"
EXPECTED_MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
EXPECTED_TRAINER_SHA256 = "e08ba26017674ac8453683e7b54810325fb381f20cb32a161c5d9aa690483449"
EXPECTED_EPISODE_COMPAT_SHA256 = "5092d6ec1859a0bac91f25990383492a112579055582a5363f0e8b83d43300d8"
EXPECTED_WRAPPER_SHA256 = "ae9918f3945cbafb1813b5bc902c356414f9fcb44de72bc623aa8947e459ee8e"
EXPECTED_SUPERVISOR_SHA256 = "1fce8517d184eb25775aaacad6ac05d24a9e1ad3cc3cfca6c3ac929fbc5c4e69"
EXPECTED_EVALUATOR_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"
EVALUATOR = Path("/root/smolvla-eval-prep/scripts/eval_v3_task0_state_capture_v1.py")
EVAL_CODE_HASHES = {
    "run_scaling_eval_v1.py": "acb7c9349a6e8598856c3a6d9119aa905be05a5c2252d66428bce9f673ae1fea",
    "eval_scaling_environment_v1.py": "d11d463287ad08f828748126f8a666faa5404e3a6a7f2de14d4757de2559da2f",
    "scaling_eval_schedule_v1.py": "094aad5affd0e9c2a29b68a49aab2fd61feb6fa9e4365ec09a3d45da675e9065",
    "diagnostic_process_guard_v1.py": "e5f24725836e4c9178b247332cb33e1eafa9224ed22e4d016fff4778787cd49f",
    "controlled_eval_hooks_v1.py": "88c519c9d3733fdb9c6b74a88e7d4ee15930fb492d360d7d4b083728bfd31ae2",
    "first_decision_provenance_v1.py": "d098b099ebad38c750abc0f0d142109695daaa3823d0901ff6b13db9106d8981",
    "capture_reset_render_provenance_v1.py": "08a28b35654c1e7af9aec78fcca84e03c0db233810fd1f2e1b74931f8bade6fc",
    "task1_freshseed2001_eval_schedule_v1.py": "1f666016800c646e459eed439b6e1e6eeef332fccd593c8608c38e20266d3a96",
    "run_task1_freshseed2001_eval_v1.py": "9a202c00811541030517bf069decefb043f2743d6ced5e84590e7564244f9610",
    "supervise_task1_freshseed2001_eval_v1_20261014.sh": "524c42f9f584ab88dc9baee4a4a69aa9e5d52ad8621ec4cad074cffe710d5a52",
}
TRAIN_OUT = Path("/root/smolvla-training-prep/results/training") / RUN_ID
TRAIN_RECOVERY = Path("/root/smolvla-training-prep/recovery") / RUN_ID
DEPLOY_SCRIPTS = TRAIN_RECOVERY / "scripts"
WRAPPER = DEPLOY_SCRIPTS / "teacher_native_spatial_task1_freshseed2001_train_wrapper_v1_20261014.py"
SUPERVISOR = DEPLOY_SCRIPTS / "supervise_teacher_native_spatial_task1_freshseed2001_40k_v1_20261014.sh"
EVAL_OUT = Path("/root/smolvla-eval-prep/results") / f"{RUN_ID}_eval_v1_20261014"
EVAL_RECOVERY = Path("/root/smolvla-eval-prep/recovery") / f"{RUN_ID}_eval_v1_20261014"
EVAL_SCRIPT_DIR = EVAL_RECOVERY / "scripts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def changed_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for key in set(a) | set(b):
            p = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.add(p)
            else:
                out |= changed_paths(a[key], b[key], p)
        return out
    return set() if a == b else {prefix}


def lr_trace(cfg):
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
    optimizer = cfg.optimizer.build([parameter])
    scheduler = cfg.scheduler.build(optimizer, 40000)
    trace = []
    for _ in range(40000):
        trace.append(tuple(float(g["lr"]) for g in optimizer.param_groups))
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only preflight requires CUDA_VISIBLE_DEVICES empty")

    source_bytes, candidate_bytes = args.source.read_bytes(), args.candidate.read_bytes()
    hashes = {
        "source_config": hashlib.sha256(source_bytes).hexdigest(),
        "candidate_config": hashlib.sha256(candidate_bytes).hexdigest(),
        "conversion_manifest": sha256(MANIFEST),
        "base_model": sha256(MODEL),
        "installed_trainer": sha256(Path(lerobot.scripts.lerobot_train.__file__)),
        "episode_compat_helper": sha256(Path("/root/smolvla-training-prep/scripts/lerobot_loco_episode_compat.py")),
        "task1_wrapper": sha256(WRAPPER),
        "task1_supervisor": sha256(SUPERVISOR),
        "frozen_evaluator": sha256(EVALUATOR),
    }
    expected = {
        "source_config": EXPECTED_SOURCE_SHA256,
        "candidate_config": EXPECTED_CONFIG_SHA256,
        "conversion_manifest": EXPECTED_MANIFEST_SHA256,
        "base_model": EXPECTED_MODEL_SHA256,
        "installed_trainer": EXPECTED_TRAINER_SHA256,
        "episode_compat_helper": EXPECTED_EPISODE_COMPAT_SHA256,
        "task1_wrapper": EXPECTED_WRAPPER_SHA256,
        "task1_supervisor": EXPECTED_SUPERVISOR_SHA256,
        "frozen_evaluator": EXPECTED_EVALUATOR_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"Pinned hash mismatch: {hashes}")

    source_raw, candidate_raw = json.loads(source_bytes), json.loads(candidate_bytes)
    expected_raw = json.loads(source_bytes)
    expected_raw["dataset"]["episodes"] = list(range(50, 100))
    expected_raw["job_name"] = RUN_ID
    expected_raw["output_dir"] = str(TRAIN_OUT)
    if changed_paths(source_raw, candidate_raw) != {"dataset.episodes", "job_name", "output_dir"}:
        raise ValueError("Candidate config differs outside the registered three fields")
    if candidate_raw != expected_raw:
        raise ValueError("Candidate config is not the exact registered task1 transform")
    cfg = draccus.decode(TrainPipelineConfig, candidate_raw)
    cfg.validate()
    if (cfg.steps != 40000 or cfg.seed != 2001 or cfg.batch_size != 2 or cfg.resume
            or cfg.env is not None or cfg.dataset.repo_id != REPO_ID
            or Path(cfg.dataset.root) != DATA_ROOT or cfg.dataset.episodes != list(range(50, 100))
            or Path(cfg.policy.pretrained_path) != Path("/root/smolvla-training-prep/models/smolvla_base_libero")):
        raise ValueError("Decoded runtime config violates registered task1 design")

    baseline = draccus.decode(TrainPipelineConfig, source_raw)
    if lr_trace(baseline) != lr_trace(cfg):
        raise ValueError("Actual optimizer/scheduler LR vectors differ across 40000 updates")

    compat_path = Path("/root/smolvla-training-prep/scripts/lerobot_loco_episode_compat.py")
    if sha256(compat_path) != EXPECTED_EPISODE_COMPAT_SHA256:
        raise ValueError("Installed episode-filter compatibility helper hash mismatch")
    sys.path.insert(0, "/root/smolvla-training-prep")
    from scripts.lerobot_loco_episode_compat import install_episode_filter_compat
    install_episode_filter_compat()
    data = make_dataset(cfg)
    if len(data) != 6707 or data.num_episodes != 50 or data.meta.total_tasks != 4:
        raise ValueError(f"Unexpected filtered dataset: len={len(data)}, episodes={data.num_episodes}")
    task_text = "pick up the black bowl next to the ramekin and place it on the plate"
    samples = []
    for i in (0, len(data) // 2, len(data) - 1):
        sample = data[i]
        shapes = {k: list(sample[k].shape) for k in ("action", "observation.state", "observation.images.agentview", "observation.images.wrist")}
        finite = all(bool(sample[k].isfinite().all()) for k in ("action", "observation.state", "observation.images.agentview", "observation.images.wrist"))
        if int(sample["task_index"]) != 1 or sample["task"] != task_text or not finite:
            raise ValueError(f"Task/text/nonfinite mismatch at filtered row {i}")
        if shapes != {"action": [50, 7], "observation.state": [1, 15], "observation.images.agentview": [1, 3, 128, 128], "observation.images.wrist": [1, 3, 128, 128]}:
            raise ValueError(f"Unexpected task1 sample schema at {i}: {shapes}")
        samples.append({"index": i, "shapes": shapes, "finite": finite, "task_index": int(sample["task_index"])})

    eval_hashes = {name: sha256(EVAL_SCRIPT_DIR / name) for name in EVAL_CODE_HASHES}
    if eval_hashes != EVAL_CODE_HASHES:
        raise ValueError(f"Evaluation code hash mismatch: {eval_hashes}")
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))
    from task1_freshseed2001_eval_schedule_v1 import schedule as task1_schedule
    rows = task1_schedule()
    if len(rows) != 24 or sum(r["kind"] == "paired" for r in rows) != 20 or sum(r["init"] == 3 for r in rows) != 5:
        raise ValueError("Deployed task1 evaluation schedule does not match its 24-rollout registration")
    help_text = subprocess.check_output([sys.executable, str(EVAL_SCRIPT_DIR / "run_task1_freshseed2001_eval_v1.py"), "--help"],
                                        text=True, timeout=60, cwd=EVAL_SCRIPT_DIR)
    if "--manifest" not in help_text:
        raise ValueError("Task1 evaluation runner CLI/import smoke failed")

    recovery_paths = [TRAIN_OUT, EVAL_OUT]
    recovery_paths += [TRAIN_RECOVERY / f"{RUN_ID}{suffix}" for suffix in (
        ".training.log", ".trainer.pid", ".exit_code", ".timed_out",
        ".supervisor.log", ".supervisor.log.pid", ".launch_manifest.json",
    )]
    recovery_paths += [EVAL_RECOVERY / name for name in (
        "evaluation_manifest.json", f"{RUN_ID}.evaluation.log", f"{RUN_ID}.launcher.pid",
        f"{RUN_ID}.exit_code", f"{RUN_ID}.timed_out", f"{RUN_ID}.runner.pid",
        f"{RUN_ID}.supervisor.log",
    )]
    for p in recovery_paths:
        if p.exists() or p.is_symlink():
            raise FileExistsError(f"Registered independent target already exists: {p}")
    if LOCK.is_symlink() or not LOCK.is_file():
        raise RuntimeError("Production lock missing, symlinked, or not a regular file")
    with LOCK.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)

    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        try:
            parent = int(Path(f"/proc/{parent}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            break
    conflicts = []
    for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) not in ancestors and any(
            t in parts[1] for t in ("lerobot_train", "teacher_native_spatial", "eval_v3_task0_state_capture")
        ):
            conflicts.append(line)
    if conflicts:
        raise RuntimeError(f"Conflicting process found: {conflicts[:5]}")

    gpu_apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    gpu_line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip()
    fields = [x.strip() for x in gpu_line.split(",")]
    if gpu_apps or fields[0] != "NVIDIA GeForce RTX 5090" or fields[1] != "0 %" or int(fields[2].split()[0]) > 128:
        raise RuntimeError(f"GPU not idle/unique: {gpu_line}; apps={gpu_apps}")
    free_bytes = shutil.disk_usage("/root").free
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(f"Less than 20 GiB free: {free_bytes}")

    print(json.dumps({"status": "passed_readonly_cpu_preflight_not_launched", "experiment_id": RUN_ID,
                      "hashes": hashes, "runtime_config": {"steps": cfg.steps, "seed": cfg.seed,
                      "batch_size": cfg.batch_size, "task1_episodes": [50, 99]},
                      "lr_trace": {"positions": 40000, "exact_match_to_task0_seed2001": True},
                      "filtered_loader": {"episodes": 50, "frames": 6707, "task_id": 1, "samples": samples},
                      "evaluation": {"evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
                      "code_hashes": eval_hashes, "schedule_rows": len(rows), "runner_import_help": "passed"},
                      "resources": {"gpu": gpu_line, "compute_apps": gpu_apps,
                      "free_disk_bytes": free_bytes, "minimum_free_bytes": 20 * 1024**3,
                      "production_lock_nonblocking_probe": "acquired"},
                      "all_independent_paths_absent": True, "training_started": False,
                      "fold02": "LOCKED", "automatic_retry": False}, indent=2))


if __name__ == "__main__":
    main()
