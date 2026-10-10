"""Hash-pinned v2 diagnostic source; preserves the failed v1 file."""
import hashlib
from pathlib import Path


def build_source(source):
    expected = "70310dc87f176920e0a0ee0b467e4c20b58b8c1bb5328fede36f0bd72c4d9ef6"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("v1 source drift; stop")
    if source.count("scheduler.last_epoch") != 2:
        raise ValueError("unexpected observation sites")
    corrected = source.replace("scheduler.last_epoch", 'scheduler.state_dict()["last_epoch"]')
    corrected = corrected.replace("one_actual_cpu_resume_update_passed", "one_actual_cpu_resume_update_v2_passed")
    compile(corrected, "actual_resume_update_probe_v2", "exec")
    return corrected


if __name__ == "__main__":
    print(build_source(Path(__file__).with_name("probe_actual_resume_update_v1.py").read_text()))
