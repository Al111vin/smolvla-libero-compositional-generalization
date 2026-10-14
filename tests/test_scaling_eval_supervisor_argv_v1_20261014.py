"""Regression fixture for the frozen runner's actual argv-based conflict scan."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_scaling_eval_v1 import scaling_conflicts


def scan_fixture(argv):
    with TemporaryDirectory() as temporary:
        proc_root = Path(temporary)
        (proc_root / "999").mkdir()
        (proc_root / "999/cmdline").write_bytes(b"caller\0")
        (proc_root / "1000").mkdir()
        (proc_root / "1000/cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
        return scaling_conflicts(proc_root=proc_root, caller_pid=999)


def test_timeout_parent_false_positive_is_reproduced():
    argv = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", "21600s",
            "/usr/local/bin/python", "/frozen/run_scaling_eval_v1.py", "--manifest", "/m.json"]
    assert scan_fixture(argv) == [1000]


def test_supervisor_argv_does_not_expose_runner_path():
    argv = ["/bin/bash", "/recovery/supervise_scaling_eval_attempt2_v1_20261014.sh",
            "/recovery/manifest.json", "/recovery/receipt.json",
            "/recovery/timed_out", "21600"]
    assert scan_fixture(argv) == []


def test_bash_c_topology_is_exempt_but_not_used_as_the_registered_supervisor():
    argv = ["/bin/bash", "-c", "python /frozen/run_scaling_eval_v1.py --manifest /m.json"]
    assert scan_fixture(argv) == []
