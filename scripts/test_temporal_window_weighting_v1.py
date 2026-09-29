from __future__ import annotations

import torch

from temporal_window_weighting_v1 import terminal_window_weights


def test_weights_use_episode_local_progress() -> None:
    batch = {"frame_index": torch.tensor([0, 8, 9, 0]), "episode_index": torch.tensor([0, 0, 0, 1])}
    weights = terminal_window_weights(batch, {0: 10, 1: 4}, terminal_fraction=0.2, terminal_weight=3.0)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5, 1.5, 0.5]))


def test_invalid_metadata_fails_closed() -> None:
    try:
        terminal_window_weights({"frame_index": torch.tensor([0])}, {0: 1})
    except KeyError:
        return
    raise AssertionError("missing episode_index must fail closed")

