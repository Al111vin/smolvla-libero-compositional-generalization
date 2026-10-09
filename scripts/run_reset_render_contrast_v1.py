"""Six independent environment captures, exclusive lock, fail-stop."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from diagnostic_process_guard_v1 import conflicting_processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--capture-sha", required=True)
    parser.add_argument("--helper-sha", required=True)
    parser.add_argument("--guard-sha", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    assert not root.exists()
    base = Path(__file__).parent
    for name, expected in (("capture_reset_render_provenance_v1.py", args.capture_sha),
                           ("first_decision_provenance_v1.py", args.helper_sha),
                           ("diagnostic_process_guard_v1.py", args.guard_sha)):
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected
    with open("/root/smolvla-training-prep/teacher_control_gpu.lock", "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not conflicting_processes()
        assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip()
        root.mkdir()
        code = 1
        try:
            for mode in ("frozen_order", "preseed"):
                for i in range(3):
                    output = root / f"{mode}_{i}.json"
                    with (root / f"{mode}_{i}.log").open("x") as log:
                        result = subprocess.run([sys.executable,
                            str(base / "capture_reset_render_provenance_v1.py"),
                            "--evaluator", args.evaluator, "--mode", mode,
                            "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
                            timeout=120, cwd="/root/smolvla-eval-prep")
                    assert result.returncode == 0
                    d = json.loads(output.read_text())
                    assert d["policy_calls"] == d["predicted_actions"] == 0
                    assert d["wait_steps"] == 10 and d["mode"] == mode
                    print("VERIFIED", mode, i, flush=True)
            code = 0
        finally:
            with (root / "exit_code").open("x") as f:
                f.write(str(code) + "\n")


if __name__ == "__main__":
    main()
