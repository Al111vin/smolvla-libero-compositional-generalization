"""One-shot, 24-rollout task1 evaluation runner; no retries or progression unlock."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from run_scaling_eval_v1 import (
    CHECKPOINT_FILES, scaling_conflicts, unique_csv, validate_checkpoint,
    validate_code_manifest,
)
from scaling_eval_schedule_v1 import validate_protocol, validate_rollout
from task1_freshseed2001_eval_schedule_v1 import schedule, gate_task1
from capture_reset_render_provenance_v1 import EVALUATOR_SHA

RUN_ID = "teacher_native_spatial_task1_freshseed2001_40k_batch2_v1_20261014"
FROZEN_EVALUATOR_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"
LOCK_PATH = Path("/root/smolvla-training-prep/teacher_control_gpu.lock")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(manifest):
    root = Path(manifest["root"])
    base = Path(__file__).resolve().parent
    checkpoint = Path(manifest["checkpoint"])
    evaluator = Path(manifest["evaluator"])
    if manifest.get("experiment_id") != RUN_ID or manifest.get("rollout_budget") != 24:
        raise ValueError("Manifest exceeds or differs from the registered 24-rollout task1 budget")
    if manifest.get("automatic_retries") is not False:
        raise ValueError("Automatic retries are forbidden")
    if manifest.get("runner_sha256") != sha256(Path(__file__)):
        raise ValueError("Task1 evaluation runner hash mismatch")
    if manifest.get("schedule_sha256") != sha256(base / "task1_freshseed2001_eval_schedule_v1.py"):
        raise ValueError("Task1 evaluation schedule hash mismatch")
    with LOCK_PATH.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        validate_code_manifest(base, manifest["code_hashes"])
        if EVALUATOR_SHA != FROZEN_EVALUATOR_SHA256 or sha256(evaluator) != FROZEN_EVALUATOR_SHA256:
            raise ValueError("Frozen evaluator hash mismatch")
        validate_checkpoint(checkpoint, manifest["model_sha256"])
        expected_checkpoint = "/root/smolvla-training-prep/results/training/" + RUN_ID + "/checkpoints/040000/pretrained_model"
        if str(checkpoint) != expected_checkpoint:
            raise ValueError("Evaluation checkpoint is not the registered final 040000 checkpoint")
        if scaling_conflicts():
            raise RuntimeError("Conflicting process found")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU is occupied")
        if shutil.disk_usage(root.parent).free < 4_000_000_000:
            raise RuntimeError("Insufficient evaluation disk space")
        root.mkdir()
        code = 1
        records, signatures = [], {}
        try:
            for row in schedule():
                target = root / row["key"]
                command = [sys.executable, str(base / "eval_scaling_environment_v1.py"),
                    "--evaluator", str(evaluator), "--checkpoint", str(checkpoint),
                    "--model-sha", manifest["model_sha256"], "--root", str(target), "--key", row["key"]]
                with (root / (row["key"] + ".log")).open("x") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=600, cwd="/root/smolvla-eval-prep")
                if result.returncode != 0:
                    raise RuntimeError(f"Rollout failed: {row['key']}, exit {result.returncode}")
                protocol = json.loads((target / "protocol.json").read_text())
                validate_protocol(row, protocol)
                if protocol.get("checkpoint") != str(checkpoint) or protocol.get("model_sha") != manifest["model_sha256"]:
                    raise ValueError("Checkpoint/model provenance mismatch")
                capture = json.loads((target / "first_input.json").read_text())
                if not capture.get("images") or not capture.get("model_arrays"):
                    raise ValueError("Missing first-input or model-array evidence")
                signature = {k: capture[k] for k in ("model_arrays", "simulator_state", "images", "frame", "input", "processor", "loaded_model", "rng")}
                key = (row["task_id"], row["init"])
                if key in signatures and signatures[key] != signature:
                    raise ValueError(f"Repeated task1 initialization provenance mismatch: {key}")
                signatures[key] = signature
                summaries = unique_csv(target, "*_summary.csv")
                actions = unique_csv(target, "*_actions.csv")
                if len(summaries) != 1:
                    raise ValueError("Expected exactly one summary row")
                success = validate_rollout(row, summaries[0], actions)
                record = dict(row, success=success, steps=int(summaries[0]["steps"]))
                records.append(record)
                with (root / (row["key"] + ".verified.json")).open("x") as f:
                    json.dump(record, f, indent=2)
                print("VERIFIED", row["key"], flush=True)
            result = gate_task1({r["key"]: r["success"] for r in records})
            with (root / "completion.json").open("x") as f:
                json.dump({"records": records, "task1_diagnostic": result,
                           "four_task_gate_unmodified": True,
                           "fold02": "LOCKED", "eight_task_expansion": "LOCKED"}, f, indent=2)
            code = 0
        finally:
            with (root / "exit_code").open("x") as f:
                f.write(str(code) + "\n")
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    raise SystemExit(run(json.loads(Path(args.manifest).read_text())))
