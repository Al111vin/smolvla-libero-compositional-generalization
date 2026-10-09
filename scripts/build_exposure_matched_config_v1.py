"""Reproducible independent four-task config; exclusive-create output only."""
import copy
import json
from pathlib import Path

RUN_ID = "teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009"


def build(baseline):
    config = copy.deepcopy(baseline)
    if baseline["steps"] != 40000 or baseline["batch_size"] != 2 or baseline["env"] is not None:
        raise ValueError("unexpected baseline recipe")
    config["dataset"]["episodes"] = list(range(200))
    config["steps"] = 160000
    config["save_freq"] = 40000
    config["job_name"] = RUN_ID
    config["output_dir"] = "/root/smolvla-training-prep/results/training/" + RUN_ID
    return config


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/teacher_native_spatial_task0_single_40k_batch2_20261003.json"
    target = root / "configs" / (RUN_ID + ".json")
    config = build(json.loads(source.read_text()))
    with target.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
