"""Isolated SmolVLA temporal-window weighted-loss training wrapper.

This file is a draft entry point. It monkey-patches only the imported policy
for one process and requires an explicit POSE_WINDOW_PLAN environment
variable. It never edits LeRobot or the source dataset.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def load_window_plan(path: str) -> dict[int, tuple[int, int, float]]:
    payload = json.loads(Path(path).read_text())
    return {
        int(r["episode_index"]): (
            int(r["window_start"]),
            int(r["window_end_exclusive"]),
            float(r["sampling_multiplier"]),
        )
        for r in payload["records"]
        if r["split"] == "train"
    }


def sample_weights(episode_index, frame_index, plan, device):
    """Return one scalar weight per batch sample."""
    ep = episode_index.detach().to("cpu").tolist()
    fr = frame_index.detach().to("cpu").tolist()
    values = []
    for e, f in zip(ep, fr, strict=True):
        start, end, multiplier = plan.get(int(e), (0, 0, 1.0))
        values.append(multiplier if start <= int(f) < end else 1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def install_weighted_forward(plan_path: str) -> None:
    plan = load_window_plan(plan_path)
    original_forward = SmolVLAPolicy.forward

    def weighted_forward(self, batch, noise=None, time=None, reduction="mean"):
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        losses = losses[:, :, : self.config.action_feature.shape[0]]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        losses = losses[:, :, : self.config.max_action_dim]
        weights = sample_weights(batch["episode_index"], batch["frame_index"], plan, losses.device)
        weights = weights[:, None, None]
        weighted = losses * weights
        denom = weights.expand_as(losses).sum().clamp_min(1e-8)
        if reduction == "none":
            per_sample = weighted.sum(dim=(1, 2)) / weights.expand_as(losses).sum(dim=(1, 2)).clamp_min(1e-8)
            return per_sample, {"loss": per_sample.mean().item(), "pose_window_plan": plan_path}
        loss = weighted.sum() / denom
        return loss, {"loss": loss.item(), "pose_window_plan": plan_path}

    SmolVLAPolicy.forward = weighted_forward
    print(f"POSE_WINDOW_WEIGHTING_INSTALLED episodes={len(plan)} plan={plan_path}")


if __name__ == "__main__":
    plan_path = os.environ.get("POSE_WINDOW_PLAN")
    if not plan_path:
        raise RuntimeError("Set POSE_WINDOW_PLAN to an audited plan JSON before training")
    install_weighted_forward(plan_path)
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
