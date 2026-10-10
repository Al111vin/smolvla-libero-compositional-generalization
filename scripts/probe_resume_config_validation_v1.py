"""Read-only installed config validation; never invokes trainer."""
import hashlib
import json
import sys
from pathlib import Path
import draccus
import lerobot.scripts.lerobot_train
from lerobot.configs.train import TrainPipelineConfig


def main():
    root = Path("/root/smolvla-training-prep/results/training")
    cases = [
        ("teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1", "020000", 40000),
        ("teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009", "080000", 160000),
    ]
    records = []
    original_argv = sys.argv
    try:
        for name, step, total in cases:
            checkpoint = root / name / "checkpoints" / step
            path = checkpoint / "pretrained_model/train_config.json"
            before = path.read_bytes()
            cfg = draccus.decode(TrainPipelineConfig, json.loads(before))
            cfg.resume = True
            cfg.output_dir = root / (name + "_resume_config_probe_not_launched_v1")
            assert not cfg.output_dir.exists()
            sys.argv = ["readonly_config_probe", f"--config_path={path}"]
            cfg.validate()
            assert cfg.checkpoint_path == checkpoint
            assert cfg.policy.pretrained_path == checkpoint / "pretrained_model"
            assert cfg.steps == total and cfg.env is None and not cfg.policy.push_to_hub
            assert not cfg.output_dir.exists() and path.read_bytes() == before
            records.append({"checkpoint_step": int(step), "total_steps": total,
                            "config_sha256_unchanged": hashlib.sha256(before).hexdigest(),
                            "checkpoint_path_correct": True, "output_not_created": True})
    finally:
        sys.argv = original_argv
    print(json.dumps({"status": "installed_resume_config_validation_passed",
                      "records": records, "trainer_invoked": False}))


if __name__ == "__main__":
    main()
