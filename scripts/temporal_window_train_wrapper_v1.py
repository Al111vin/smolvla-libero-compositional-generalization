"""Version-locked Phase M trainer wrapper; fails closed on source drift."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

TRAINER = Path("/usr/local/miniconda3/envs/py312/lib/python3.12/site-packages/lerobot/scripts/lerobot_train.py")

def build_source(source: str) -> str:
    markers = ["def update_policy(\n", "    # Let accelerator handle mixed precision\n", "    dataset = make_dataset(cfg)\n", "        else:\n            loss, output_dict = policy.forward(batch)\n"]
    for marker in markers:
        if marker not in source:
            raise RuntimeError(f"trainer source marker missing: {marker!r}")
    source = "import os\n" + source
    source = source.replace("def update_policy(\n", "TEMPORAL_EPISODE_LENGTHS = None\n\n\ndef update_policy(\n", 1)
    injection = ("    # Let accelerator handle mixed precision\n"
        "    temporal_weights = None\n"
        "    if TEMPORAL_EPISODE_LENGTHS is not None:\n"
        "        frame_index = torch.as_tensor(batch[\"frame_index\"])\n"
        "        episode_index = torch.as_tensor(batch[\"episode_index\"])\n"
        "        fraction = float(os.environ.get(\"TEMPORAL_TERMINAL_FRACTION\", \"0.15\"))\n"
        "        terminal_weight = float(os.environ.get(\"TEMPORAL_TERMINAL_WEIGHT\", \"2.0\"))\n"
        "        if not 0.0 < fraction < 1.0 or terminal_weight <= 0.0:\n"
        "            raise ValueError(\"invalid temporal weighting parameters\")\n"
        "        temporal_weights = torch.ones(frame_index.shape, dtype=torch.float32, device=frame_index.device)\n"
        "        for i, ep_value in enumerate(episode_index.detach().cpu().tolist()):\n"
        "            ep = int(ep_value)\n"
        "            if ep not in TEMPORAL_EPISODE_LENGTHS:\n"
        "                raise KeyError(f\"missing episode length for episode {ep}\")\n"
        "            start = max(0, int((1.0 - fraction) * TEMPORAL_EPISODE_LENGTHS[ep]))\n"
        "            if int(frame_index[i].item()) >= start:\n"
        "                temporal_weights[i] = terminal_weight\n"
        "        temporal_weights = temporal_weights * (temporal_weights.numel() / temporal_weights.sum().clamp_min(1e-12))\n")
    source = source.replace("    # Let accelerator handle mixed precision\n", injection, 1)
    replacement = ("        else:\n"
        "            if temporal_weights is not None:\n"
        "                per_sample_loss, output_dict = policy.forward(batch, reduction=\"none\")\n"
        "                loss = (per_sample_loss * temporal_weights).sum() / temporal_weights.sum().clamp_min(1e-6)\n"
        "                output_dict[\"temporal_terminal_weight_mean\"] = temporal_weights.mean().detach()\n"
        "            else:\n"
        "                loss, output_dict = policy.forward(batch)\n")
    source = source.replace("        else:\n            loss, output_dict = policy.forward(batch)\n", replacement, 1)
    dataset_patch = ("    dataset = make_dataset(cfg)\n"
        "    if cfg.use_rabc:\n"
        "        raise ValueError(\"Phase M temporal wrapper is incompatible with use_rabc\")\n"
        "    episodes = dataset.meta.episodes\n"
        "    TEMPORAL_EPISODE_LENGTHS = {\n"
        "        int(i): int(episodes[\"dataset_to_index\"][i] - episodes[\"dataset_from_index\"][i])\n"
        "        for i in range(len(episodes[\"dataset_from_index\"]))\n"
        "    }\n")
    return source.replace("    dataset = make_dataset(cfg)\n", dataset_patch, 1)

def main() -> None:
    if not TRAINER.exists():
        raise FileNotFoundError(TRAINER)
    patched = build_source(TRAINER.read_text())
    with tempfile.NamedTemporaryFile("w", suffix="_lerobot_train_temporal.py", delete=False) as handle:
        handle.write(patched)
        temp_path = handle.name
    os.execv(sys.executable, [sys.executable, temp_path, *sys.argv[1:]])

if __name__ == "__main__":
    main()
