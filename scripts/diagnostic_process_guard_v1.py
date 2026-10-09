"""Match actual executable/script/module arguments, never inline source text."""
from pathlib import Path

TARGETS = {"diagnose_first_decision_v1.py", "eval_v3_task0_state_capture_v1.py",
           "lerobot_train", "lerobot_train.py", "lerobot_train_loco_compat.py",
           "scripts.lerobot_train_loco_compat", "task_balanced_train_wrapper_v1.py"}


def is_conflicting_argv(argv):
    if not argv:
        return False
    # python -c source / shell -c source are launchers, not running the named file.
    # Their actual GPU consumers remain protected by the compute-process check/lock.
    if "-c" in argv[:3]:
        return False
    for token in argv:
        if token.startswith("-"):
            continue
        if Path(token).name in TARGETS or token in TARGETS:
            return True
    return False


def conflicting_processes(proc_root=Path("/proc")):
    found = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = path.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            continue
        if is_conflicting_argv(argv):
            found.append(int(path.name))
    return found
