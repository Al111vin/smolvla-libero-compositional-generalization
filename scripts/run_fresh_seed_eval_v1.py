"""One-shot, fail-closed fresh-seed evaluation runner; no retries."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_child_signals_v4 import run_with_signal_cleanup, SupervisorSignal
from fresh_seed_eval_schedule_v1 import schedule
from scaling_eval_schedule_v1 import environment_signature, validate_protocol, validate_rollout


def safe_path(path):
    p = Path(path)
    if not p.is_absolute() or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("absolute non-symlink paths required")
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(manifest):
    root = safe_path(manifest["root"])
    if root.exists():
        raise FileExistsError(root)
    scripts = safe_path(manifest["scripts_root"])
    for name, sha in manifest["code_hashes"].items():
        if digest(safe_path(scripts / name)) != sha:
            raise ValueError(f"Registered code hash mismatch: {name}")
    lock_path = safe_path(manifest["gpu_lock"])
    fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("registered regular lock required")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.exists():
            raise FileExistsError(root)
        for name, sha in manifest["pinned_files"].items():
            if digest(safe_path(name)) != sha:
                raise ValueError(f"Pinned file hash mismatch: {name}")
        for model in manifest["models"].values():
            checkpoint = safe_path(model["checkpoint"])
            if digest(checkpoint / "model.safetensors") != model["sha256"]:
                raise ValueError("Checkpoint hash mismatch")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU compute process conflict")
        if shutil.disk_usage(root.parent).free < manifest["minimum_free_bytes"]:
            raise RuntimeError("insufficient disk")
        root.mkdir()
        (root / "rollout_logs").mkdir()
        rows = schedule()
        signatures = {}
        policy_signatures = {}
        records = []
        terminal_status = "error"
        try:
            for row in rows:
                model = manifest["models"][row["model"]]
                target = root / row["model"] / row["key"]
                if target.exists():
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(scripts / "eval_scaling_environment_v1.py"),
                    "--evaluator", manifest["evaluator"], "--checkpoint", model["checkpoint"],
                    "--model-sha", model["sha256"], "--root", str(target), "--key", row["key"].removeprefix(row["model"] + "_")]
                log_path = root / "rollout_logs" / (row["key"] + ".log")
                with log_path.open("x") as log:
                    result = run_with_signal_cleanup(command, seconds=600,
                        grace_seconds=10, cwd=manifest["cwd"], stdout=log)
                if result["timed_out"] or result["returncode"] != 0:
                    raise RuntimeError(f"rollout failed: {row['key']} result={result}")
                protocol = json.loads((target / "protocol.json").read_text())
                validate_protocol(row, protocol)
                if protocol.get("checkpoint") != model["checkpoint"] or protocol.get("model_sha") != model["sha256"]:
                    raise ValueError("protocol checkpoint provenance mismatch")
                capture = json.loads((target / "first_input.json").read_text())
                pair = (row["task_id"], row["kind"], row["init"])
                signature = environment_signature(capture)
                if pair in signatures and signature != signatures[pair]:
                    raise ValueError(f"paired environment mismatch: {row['key']}")
                signatures[pair] = signature
                if row["task_id"] == 0:
                    policy_key = (row["model"], row["task_id"], row["kind"], row["init"], row["repeat"])
                    policy_signature = {k: capture[k] for k in
                        ("input", "processor", "loaded_model", "rng")}
                    repeated_key = (row["model"], row["task_id"], row["init"])
                    previous = next((value for key, value in policy_signatures.items()
                                     if (key[0], key[1], key[3]) == repeated_key), None)
                    if previous is not None and previous != policy_signature:
                        raise ValueError(f"Repeated-init policy provenance mismatch: {row['key']}")
                    policy_signatures[policy_key] = policy_signature
                summaries = list(target.glob("*_summary.csv"))
                actions_files = list(target.glob("*_actions.csv"))
                if len(summaries) != 1 or len(actions_files) != 1:
                    raise ValueError("missing or ambiguous rollout CSV evidence")
                with summaries[0].open() as f:
                    summary_rows = list(csv.DictReader(f))
                with actions_files[0].open() as f:
                    actions = list(csv.DictReader(f))
                if len(summary_rows) != 1:
                    raise ValueError("expected exactly one summary row")
                success = validate_rollout(row, summary_rows[0], actions)
                record = dict(row, success=success, steps=int(summary_rows[0]["steps"]))
                with (root / (row["key"] + ".verified.json")).open("x") as f:
                    json.dump(record, f)
                records.append(record)
            # Task0 model pairs must share processor/input/RNG as well as the environment.
            for row in rows:
                if row["task_id"] == 0:
                    counterpart = "joint" if row["model"] == "single" else "single"
                    a = policy_signatures[(row["model"], 0, row["kind"], row["init"], row["repeat"])]
                    b = policy_signatures[(counterpart, 0, row["kind"], row["init"], row["repeat"])]
                    comparable = ("input", "processor", "rng")
                    if any(a[k] != b[k] for k in comparable):
                        raise ValueError(f"task0 paired policy input mismatch: {row['key']}")
            outcomes = {r["key"]: r["success"] for r in records}
            gates = {}
            for model, tasks in (("single", [0]), ("joint", [0, 1, 2, 3])):
                for task in tasks:
                    selected = [r for r in records if r["model"] == model and r["task_id"] == task]
                    paired = sum(r["success"] for r in selected if r["kind"] == "paired")
                    init3 = sum(r["success"] for r in selected if r["init"] == 3)
                    gates[f"{model}_task{task}"] = {"paired_successes": paired, "paired_total": 20,
                        "init3_successes": init3, "init3_total": 5,
                        "registered_gate_passed": paired >= (11 if task == 0 else 10)
                            and (task != 0 or init3 == 5)}
            single_outcomes = {r["init"]: r["success"] for r in records
                               if r["model"] == "single" and r["task_id"] == 0 and r["kind"] == "paired"}
            joint_outcomes = {r["init"]: r["success"] for r in records
                              if r["model"] == "joint" and r["task_id"] == 0 and r["kind"] == "paired"}
            gains = [i for i in range(20) if joint_outcomes[i] and not single_outcomes[i]]
            losses = [i for i in range(20) if single_outcomes[i] and not joint_outcomes[i]]
            discordant = len(gains) + len(losses)
            tail = sum(__import__("math").comb(discordant, k) for k in range(min(len(gains), len(losses)) + 1))
            exact_mcnemar_p = min(1.0, 2 * tail / (2 ** discordant)) if discordant else 1.0
            with (root / "completion.json").open("x") as f:
                json.dump({"records": records, "gates": gates,
                    "task0_paired_comparison": {"joint_gains": gains, "joint_losses": losses,
                        "exact_two_sided_mcnemar_p": exact_mcnemar_p}}, f, indent=2)
            terminal_status = "complete"
        finally:
            with (root / "exit_code").open("x") as f:
                json.dump({"status": terminal_status, "verified_rollouts": len(records)}, f)
    finally:
        os.close(fd)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    main(json.loads(Path(args.manifest).read_text()))
