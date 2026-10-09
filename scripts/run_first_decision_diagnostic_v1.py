"""Bounded foreground supervisor; caller must detach with retained PID/log."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(root)
    runner = Path(__file__).with_name("diagnose_first_decision_v1.py")
    helper = runner.with_name("first_decision_provenance_v1.py")
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == args.runner_sha256
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == args.helper_sha256
    with open("/root/smolvla-training-prep/teacher_control_gpu.lock", "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip()
        processes = subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
        conflicts = ("lerobot_train", "eval_v3_task0_state_capture_v1.py", "diagnose_first_decision_v1.py")
        assert not any(any(marker in line for marker in conflicts) for line in processes)
        assert os.statvfs(root.parent).f_bavail * os.statvfs(root.parent).f_frsize > 2_000_000_000
        root.mkdir()
        code = 1
        try:
            for i in range(1, 4):
                output = root / f"process_{i:02d}.json"
                with (root / f"process_{i:02d}.log").open("x") as log:
                    result = subprocess.run([sys.executable, str(runner),
                        "--checkpoint", args.checkpoint, "--evaluator", args.evaluator,
                        "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
                        timeout=600, cwd="/root/smolvla-eval-prep")
                assert result.returncode == 0, (i, result.returncode)
                record = json.loads(output.read_text())
                assert len(record["calls"]) == 2 and record["closed_loop_action_steps"] == 0
                assert record["calls"][0]["input_digest"] == record["calls"][1]["input_digest"]
                assert record["calls"][0]["rng_digest"] == record["calls"][1]["rng_digest"]
                print("VERIFIED_PROCESS", i, flush=True)
            code = 0
        finally:
            with (root / "exit_code").open("x") as f:
                f.write(str(code) + "\n")


if __name__ == "__main__":
    main()
