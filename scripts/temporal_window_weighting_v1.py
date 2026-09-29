"""Pure temporal-window sample weighting utilities for Phase M.

This module is deliberately independent of the trainer.  It computes per-sample
weights from episode-local frame positions, so callers cannot accidentally use
global dataset indices as episode progress.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def terminal_window_weights(
    batch: Mapping[str, Any],
    episode_lengths: Mapping[int, int],
    *,
    terminal_fraction: float = 0.15,
    terminal_weight: float = 2.0,
) -> torch.Tensor:
    """Return normalized per-sample weights for a batch.

    ``frame_index`` is episode-local.  The final ``terminal_fraction`` of each
    episode receives ``terminal_weight``; all earlier frames receive 1.0.
    We normalize the result to preserve the batch mean loss scale.
    """
    if not 0.0 < terminal_fraction < 1.0:
        raise ValueError("terminal_fraction must be between 0 and 1")
    if terminal_weight <= 0.0:
        raise ValueError("terminal_weight must be positive")
    if "frame_index" not in batch or "episode_index" not in batch:
        raise KeyError("batch must contain frame_index and episode_index")

    frame = torch.as_tensor(batch["frame_index"])
    episode = torch.as_tensor(batch["episode_index"])
    if frame.ndim != 1 or episode.ndim != 1 or frame.shape != episode.shape:
        raise ValueError("frame_index and episode_index must be 1-D with equal shape")

    weights = torch.ones(frame.shape, dtype=torch.float32, device=frame.device)
    for i, ep_value in enumerate(episode.detach().cpu().tolist()):
        ep = int(ep_value)
        if ep not in episode_lengths or int(episode_lengths[ep]) <= 0:
            raise KeyError(f"missing positive episode length for episode {ep}")
        length = int(episode_lengths[ep])
        terminal_start = max(0, int((1.0 - terminal_fraction) * length))
        if int(frame[i].item()) >= terminal_start:
            weights[i] = float(terminal_weight)
    return weights * (weights.numel() / weights.sum().clamp_min(1e-12))

