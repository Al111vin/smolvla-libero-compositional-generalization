"""Run LeRobot training with the project-owned LOCO episode-filter patch."""

from __future__ import annotations

import runpy

from scripts.lerobot_loco_episode_compat import install_episode_filter_compat

install_episode_filter_compat()
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
