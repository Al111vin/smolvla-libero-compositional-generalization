"""Build the explicitly authorized seed-2002 matched-pair configs."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "single": {
        "source": "configs/teacher_native_spatial_task0_freshseed2001_40k_batch2_v1_20261011.json",
        "source_sha256": "fabb744b04083ee643e3f7db34798e465f069379fcc1d57146d5dc70aaa43f0b",
        "destination": "configs/teacher_native_spatial_task0_freshseed2002_40k_batch2_v1_20261014.json",
        "steps": 40000,
        "job_name": "teacher_native_spatial_task0_freshseed2002_40k_batch2_v1_20261014",
        "output_dir": "/root/smolvla-training-prep/results/training/teacher_native_spatial_task0_freshseed2002_40k_batch2_v1_20261014",
    },
    "joint4": {
        "source": "configs/teacher_native_spatial_4task_freshseed2001_160k_batch2_v1_20261011.json",
        "source_sha256": "2ced9539c1ec358b47ce6ae20a2db979088f3e5bfc3e8399405bc2a5d2546ac3",
        "destination": "configs/teacher_native_spatial_4task_freshseed2002_160k_batch2_v1_20261014.json",
        "steps": 160000,
        "job_name": "teacher_native_spatial_4task_freshseed2002_160k_batch2_v1_20261014",
        "output_dir": "/root/smolvla-training-prep/results/training/teacher_native_spatial_4task_freshseed2002_160k_batch2_v1_20261014",
    },
}

def build(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unregistered arm: {arm}")
    spec = ARMS[arm]
    raw = (ROOT / spec["source"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["source_sha256"]:
        raise ValueError(f"pinned source hash mismatch: {spec['source']}")
    source = json.loads(raw)
    if (
        source.get("seed") != 2001
        or source.get("steps") != spec["steps"]
        or source.get("batch_size") != 2
        or source.get("resume") is not False
        or source.get("env") is not None
    ):
        raise ValueError(f"unexpected registered source recipe for {arm}")
    result = copy.deepcopy(source)
    result.update(seed=2002, job_name=spec["job_name"], output_dir=spec["output_dir"])
    changed = {key for key in source if source[key] != result[key]}
    if changed != {"seed", "job_name", "output_dir"}:
        raise ValueError(f"unexpected config changes: {sorted(changed)}")
    return result

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=tuple(ARMS))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = ARMS[args.arm]
    rendered = json.dumps(build(args.arm), indent=2) + "\n"
    if args.write:
        destination = ROOT / spec["destination"]
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(rendered)
        print(f"wrote {spec['destination']}")
    else:
        print(rendered, end="")

if __name__ == "__main__":
    main()
