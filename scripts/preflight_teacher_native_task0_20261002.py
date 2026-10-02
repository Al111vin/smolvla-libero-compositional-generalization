"""No-gradient current official loader/processor and config preflight."""
import json
import sys
from pathlib import Path
import draccus
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from scripts.lerobot_loco_episode_compat import install_episode_filter_compat

cfg = draccus.parse(TrainPipelineConfig, config_path=sys.argv[1], args=[])
cfg.validate()
expected_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
assert not Path(cfg.output_dir).exists(), "output exists"
assert Path(cfg.policy.pretrained_path, "model.safetensors").is_file()
assert cfg.steps == 10000 and cfg.batch_size == 8
assert cfg.policy.chunk_size == 50 and cfg.policy.n_action_steps == 25
assert abs(cfg.policy.optimizer_lr - expected_lr) < 1e-12
assert abs(cfg.optimizer.lr - expected_lr) < 1e-12
assert abs(cfg.scheduler.peak_lr - expected_lr) < 1e-12
assert cfg.policy.scheduler_decay_steps == 90000
install_episode_filter_compat()
ds = make_dataset(cfg)
features = {**cfg.policy.input_features, **cfg.policy.output_features}
pre, post = make_pre_post_processors(
    policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
    dataset_stats=ds.meta.stats,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {"stats": ds.meta.stats, "features": features,
                                  "norm_map": cfg.policy.normalization_mapping},
        "rename_observations_processor": {"rename_map": cfg.rename_map}},
    postprocessor_overrides={"unnormalizer_processor": {
        "stats": ds.meta.stats, "features": cfg.policy.output_features,
        "norm_map": cfg.policy.normalization_mapping}})
batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8, num_workers=0)))
processed = pre(batch)
assert ds.num_frames == 5068 and ds.num_episodes == 50
for key in ("action", "observation.state"):
    assert torch.isfinite(processed[key]).all(), key
decoded = post(processed["action"])
diff = float((decoded.cpu() - batch["action"]).abs().max())
assert diff < 1e-5, diff
print(json.dumps({"passed": True, "frames": ds.num_frames, "episodes": ds.num_episodes,
                  "action_shape": list(processed["action"].shape),
                  "state_shape": list(processed["observation.state"].shape),
                  "action_normalize_roundtrip_max_abs_diff": diff,
                  "expected_learning_rate": expected_lr,
                  "output_dir": str(cfg.output_dir), "training_started": False,
                  "gradient_steps": 0}, indent=2))
