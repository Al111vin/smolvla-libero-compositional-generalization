#!/usr/bin/env python3
"""Task1 edge-pinch v2 fine-tune -- pure-vision closed-loop blind evaluation.

Evaluates the 6 checkpoints produced by the 3000-step edge-pinch v2
fine-tuning run (500/1000/1500/2000/2500/3000) on the 5 never-before-used
blind-eval seeds 555101-555105, using ONLY vision + proprioception input
(no privileged object pose) -- the same closed-loop rollout contract as
this project's existing Task1 diagnostic scripts
(chunk25_canary_diagnostic.py, receding_horizon_task1_diag.py,
frozen_fovy75_task1_validation.py, gripper_debounce_ablation.py), reused
here rather than reimplemented:

  - env construction: OffScreenRenderEnv(bddl_file_name=BDDL,
    camera_heights=128, camera_widths=128, horizon=1000,
    use_camera_obs=True) -- IDENTICAL to
    task1_edge_pinch_diverse_collection_recorder_v1.py's _make_env_fn(),
    which is what actually produced the training images. Deliberately does
    NOT call apply_frozen_agentview_camera() (unlike the diagnostic
    scripts above) -- the recorder never did either, so calling it here
    would introduce a camera-distribution mismatch against the training
    data.
  - seed -> initial state: np.random.seed(seed); obs = env.reset() --
    IDENTICAL to task1_edge_pinch_controller_v1.py's
    run_episode_edge_pinch() (line ~1473-1474), confirmed via GPU grep,
    NOT the diagnostic scripts' env.set_init_state(state)-from-stored-HDF5
    pattern (which has no analogue for a never-before-used seed like
    555101-555105).
  - frame()/qaxis(): byte-for-byte the same helper used in
    chunk25_canary_diagnostic.py / receding_horizon_task1_diag.py /
    frozen_fovy75_task1_validation.py / gripper_debounce_ablation.py --
    reads ONLY robot0_joint_pos, robot0_eef_pos, robot0_eef_quat (converted
    to axis-angle), robot0_gripper_qpos, agentview_image,
    robot0_eye_in_hand_image. No object-pose field is ever read.
  - rollout core: policy.select_action -> post-process -> env.step ->
    env.check_success()/reward>0, same pattern as the diagnostic scripts.

New in this script (none of the above precedents had these):
  - iterates over multiple checkpoints (500..3000) instead of one hardcoded
    path
  - a 10-step (default) zero-action settling period after env.reset(),
    NOT counted in steps_run, no model calls during it
    (--wait-steps, semantics confirmed with the user 2026-09-17)
  - hard seed gates: refuses anything outside 555101-555105, and
    explicitly refuses 555001-555005 even if accidentally requested
  - structured per-(checkpoint,seed) JSON output with checkpoint
    model.safetensors SHA-256 (not a path hash), eval script SHA-256,
    wait-phase outcome fields, and a summary aggregation
  - --smoke-test / --confirm-full-run mutually exclusive safety gate,
    mirroring convert_task1_edge_pinch_v2_to_lerobot.py's GATE A pattern

2026-09-17 update: added BLIND_EVAL_SEEDS_BATCH2_EXTENDED (555201-555220,
20 seeds) to ALLOWED_BLIND_EVAL_SEEDS, for a follow-up extended evaluation
of checkpoints 2500 and 3000 only (20 rollouts each), requested after the
initial 6x5 run showed a non-monotonic success pattern (peak at 2500,
regression at 3000, 5/30 overall) that a 5-seed-per-checkpoint sample is
too small to resolve. This does NOT change or re-run any batch1 seed's
prior result; batch1 (555101-555105) results remain exactly as recorded.

generation_only=False, not_a_model_evaluation_input=False for this
script's own output -- unlike the edge-pinch data-generation scripts, THIS
is the genuine model-performance evidence path. It remains a separate,
scoped result (this fine-tune's 6 checkpoints only) and does NOT itself
resolve this project's broader, still-unresolved Task1 strict closed-loop
evaluation status, and it is not a Fold02 gate signal either way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

DEFAULT_BDDL_PATH = (
    "/root/smolvla-training-prep/data/libero_36/bddl/"
    "task_001_layout_1_akita_black_bowl_put_on_top_middle.bddl"
)
LANGUAGE_INSTRUCTION = "pick up the akita black bowl and place it on the plate in the middle region"

BLIND_EVAL_SEEDS_BATCH1 = frozenset(range(555101, 555106))  # 555101-555105 (2026-09-17 initial 6x5 run)
BLIND_EVAL_SEEDS_BATCH2_EXTENDED = frozenset(range(555201, 555221))  # 555201-555220 (extended 2-checkpoint x 20-seed run)
ALLOWED_BLIND_EVAL_SEEDS = BLIND_EVAL_SEEDS_BATCH1 | BLIND_EVAL_SEEDS_BATCH2_EXTENDED
BLOCKED_HISTORICAL_SEEDS = frozenset(range(555001, 555006))  # 555001-555005
BLOCKED_TRAINING_SEED_PREFIX = 950000  # 950001-950025 collection seeds must never appear here

DEFAULT_CHECKPOINT_STEPS = [500, 1000, 1500, 2000, 2500, 3000]
DEFAULT_N_ACTION_STEPS = 25
DEFAULT_WAIT_STEPS = 10
DEFAULT_MAX_STEPS = 600
ACTION_DIM = 7


# --------------------------------------------------------------------------
# Pure helpers (no GPU/libero/torch import at module scope -- deferred, same
# convention as task1_edge_pinch_controller_v1.py, so --help/py_compile work
# without those packages installed).
# --------------------------------------------------------------------------

def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_seed(seed: int) -> None:
    """Raises ValueError loudly rather than silently skipping -- a wrong
    seed here must never be evaluated as if it were valid blind-eval
    evidence."""
    if seed in BLOCKED_HISTORICAL_SEEDS:
        raise ValueError(
            f"seed {seed} is in BLOCKED_HISTORICAL_SEEDS (555001-555005, "
            "already used historically -- not valid 'fresh' blind-eval "
            "evidence); refusing"
        )
    if 950001 <= seed <= 950025:
        raise ValueError(
            f"seed {seed} is a training-collection seed (950001-950025); "
            "these were used to generate the fine-tuning data and must "
            "never be used as blind-eval evidence; refusing"
        )
    if seed not in ALLOWED_BLIND_EVAL_SEEDS:
        raise ValueError(
            f"seed {seed} is not in ALLOWED_BLIND_EVAL_SEEDS "
            f"(555101-555105 batch1, 555201-555220 batch2_extended); refusing"
        )


def resolve_checkpoint_dir(checkpoints_root: str | Path, step: int) -> Path:
    d = Path(checkpoints_root) / f"{step:06d}" / "pretrained_model"
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {d}")
    return d


def qaxis(q) -> "np.ndarray":  # noqa: F821 -- np imported lazily by caller
    import numpy as np

    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    w = np.clip(w, -1, 1)
    a = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 0))
    axis = np.array([1.0, 0.0, 0.0]) if s < 1e-8 else (np.array([x, y, z]) / s)
    return (axis * a).astype("float32")


def frame(obs: dict, lang: str) -> dict:
    """Identical contract to the shared frame() used across this project's
    Task1 diagnostic scripts. Reads ONLY proprioceptive + vision fields --
    never any object-pose key."""
    import numpy as np
    import torch

    st = np.concatenate(
        [
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            qaxis(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    return {
        "observation.images.agentview": torch.as_tensor(obs["agentview_image"]).permute(2, 0, 1).float() / 255.0,
        "observation.images.wrist": torch.as_tensor(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).float() / 255.0,
        "observation.state": torch.from_numpy(st.astype("float32")),
        "task": lang,
    }


def make_env_fn(bddl_path: str):
    """Identical construction to
    task1_edge_pinch_diverse_collection_recorder_v1.py's _make_env_fn() --
    the function that actually produced the training images. Deliberately
    no apply_frozen_agentview_camera() call."""

    def _make():
        from libero.libero.envs import OffScreenRenderEnv  # noqa: WPS433

        return OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=128,
            camera_widths=128,
            horizon=1000,
            use_camera_obs=True,
        )

    return _make


def check_policy_api_compat(policy) -> None:
    """Fails loudly and immediately, before any rollout is attempted, if
    this lerobot version's SmolVLAPolicy doesn't expose the API this
    script depends on. Better than a confusing AttributeError mid-rollout
    30 minutes into a run."""
    if not hasattr(policy, "reset") or not callable(policy.reset):
        raise RuntimeError(
            "API_COMPAT_FAIL: policy has no callable .reset() method -- "
            "this script calls policy.reset() before every rollout to "
            "clear the action-chunk queue; cannot proceed safely without it"
        )
    if not hasattr(policy, "config") or not hasattr(policy.config, "n_action_steps"):
        raise RuntimeError(
            "API_COMPAT_FAIL: policy.config has no 'n_action_steps' "
            "attribute -- this script sets policy.config.n_action_steps "
            "before every rollout; cannot proceed safely without it"
        )
    if not hasattr(policy, "select_action") or not callable(policy.select_action):
        raise RuntimeError(
            "API_COMPAT_FAIL: policy has no callable .select_action() method"
        )


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def run_one_blind_rollout(
    policy,
    pre,
    post,
    env_fn,
    seed: int,
    lang: str,
    n_action_steps: int,
    wait_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    validate_seed(seed)

    env = env_fn()
    try:
        np.random.seed(seed)
        obs = env.reset()

        # --- wait phase: fixed-count settling, no model calls, not counted
        # in steps_run. If done fires during this phase, record it plainly
        # rather than silently continuing into model control. ---
        wait_terminated = False
        wait_success = False
        for _ in range(wait_steps):
            obs, wait_reward, wait_done, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            if bool(env.check_success()) or float(wait_reward) > 0:
                wait_success = True
            if wait_done:
                wait_terminated = True
                break

        rewards: list[float] = []
        success = False
        invalid_action = False
        termination_reason = "max_steps_reached"
        steps_run = 0

        if wait_terminated:
            termination_reason = "wait_phase_done"
        elif wait_success:
            # Extremely unlikely (settling into success with zero action)
            # but must be recorded truthfully, not discarded.
            success = True
            termination_reason = "wait_phase_success"
        else:
            policy.reset()
            policy.config.n_action_steps = n_action_steps

            for step in range(max_steps):
                with torch.inference_mode():
                    raw = policy.select_action(pre(frame(obs, lang)))
                processed = post(raw)
                if hasattr(processed, "detach"):
                    processed = processed.detach().cpu()
                a = np.asarray(processed, dtype=np.float32).squeeze()

                if a.shape != (ACTION_DIM,) or not np.all(np.isfinite(a)):
                    invalid_action = True
                    termination_reason = "invalid_action"
                    break

                obs, reward, done, _ = env.step(np.clip(a, -1, 1))
                rewards.append(float(reward))
                steps_run += 1

                if bool(env.check_success()) or reward > 0:
                    success = True
                    termination_reason = "success_early_termination"
                    break
                if done:
                    termination_reason = "env_done_no_success"
                    break

        return {
            "seed": seed,
            "success": success,
            "reward_max": max(rewards) if rewards else 0.0,
            "reward_final": rewards[-1] if rewards else 0.0,
            "steps_run": steps_run,
            "invalid_action_detected": invalid_action,
            "wait_steps": wait_steps,
            "wait_terminated": wait_terminated,
            "wait_success": wait_success,
            "termination_reason": termination_reason,
            "n_action_steps": n_action_steps,
            "max_steps": max_steps,
        }
    finally:
        env.close()


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finetune-checkpoints-dir", type=str, required=True)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS))
    p.add_argument("--seeds", type=int, nargs="+", default=sorted(ALLOWED_BLIND_EVAL_SEEDS))
    p.add_argument("--bddl-path", type=str, default=DEFAULT_BDDL_PATH)
    p.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    p.add_argument("--wait-steps", type=int, default=DEFAULT_WAIT_STEPS)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run only the FIRST checkpoint step and FIRST seed, to validate the full "
        "pipeline (env, policy loading, API compat, output writing) before committing "
        "to the full 6x5 run. Mutually exclusive with --confirm-full-run.",
    )
    p.add_argument(
        "--confirm-full-run", action="store_true",
        help="Required to run the full checkpoint x seed grid. Mutually exclusive with "
        "--smoke-test.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def gate_a_resolve_mode(args) -> str:
    if args.smoke_test and args.confirm_full_run:
        raise ValueError("--smoke-test and --confirm-full-run are mutually exclusive; pass exactly one")
    if not args.smoke_test and not args.confirm_full_run:
        raise ValueError(
            "refusing to run without an explicit mode: pass --smoke-test (1 checkpoint x 1 seed, "
            "pipeline validation only) or --confirm-full-run (the full grid)"
        )
    return "smoke_test" if args.smoke_test else "full_run"


def gate_b_validate_seeds(seeds: list[int]) -> None:
    for s in seeds:
        validate_seed(s)


def gate_c_check_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output_dir {output_dir} already exists and is non-empty; pass --overwrite "
            "to reuse it, or choose a new path (this script never deletes existing files "
            "itself)"
        )


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        mode = gate_a_resolve_mode(args)
        gate_b_validate_seeds(args.seeds)
        checkpoint_steps = list(args.checkpoint_steps)
        seeds = list(args.seeds)
        if mode == "smoke_test":
            checkpoint_steps = checkpoint_steps[:1]
            seeds = seeds[:1]
            print(
                f"SMOKE_TEST_MODE: restricting to checkpoint_step={checkpoint_steps[0]}, "
                f"seed={seeds[0]} only"
            )

        checkpoint_dirs = {
            step: resolve_checkpoint_dir(args.finetune_checkpoints_dir, step)
            for step in checkpoint_steps
        }
        output_dir = Path(args.output_dir)
        gate_c_check_output_dir(output_dir, args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports -- only past this point do we touch GPU/libero/lerobot.
    import torch
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401 -- import-availability check
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    eval_script_sha256 = sha256_of_file(__file__)
    bddl_sha256 = sha256_of_file(args.bddl_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_fn = make_env_fn(args.bddl_path)

    all_records: list[dict[str, Any]] = []

    for step in checkpoint_steps:
        ckpt_dir = checkpoint_dirs[step]
        model_file = ckpt_dir / "model.safetensors"
        if not model_file.is_file():
            print(f"REFUSING: checkpoint {step} missing model.safetensors at {model_file}")
            return 1
        ckpt_model_sha256 = sha256_of_file(model_file)

        print(f"Loading checkpoint step={step} from {ckpt_dir} ...")
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
        check_policy_api_compat(policy)
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        ckpt_out_dir = output_dir / f"checkpoint_{step:06d}"
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            t0 = time.time()
            result = run_one_blind_rollout(
                policy, pre, post, env_fn, seed, LANGUAGE_INSTRUCTION,
                n_action_steps=args.n_action_steps,
                wait_steps=args.wait_steps,
                max_steps=args.max_steps,
            )
            elapsed_s = time.time() - t0

            record = {
                "checkpoint_step": step,
                "checkpoint_path": str(ckpt_dir),
                "checkpoint_model_sha256": ckpt_model_sha256,
                "bddl_path": str(args.bddl_path),
                "bddl_sha256": bddl_sha256,
                "language_instruction": LANGUAGE_INSTRUCTION,
                "eval_script_sha256": eval_script_sha256,
                "pure_vision_no_privileged_pose": True,
                "not_used_for_training": True,
                "generation_only": False,
                "not_a_model_evaluation_input": False,
                "elapsed_s": round(elapsed_s, 3),
                "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **result,
            }
            all_records.append(record)

            rec_path = ckpt_out_dir / f"seed_{seed}.json"
            with open(rec_path, "w") as f:
                json.dump(record, f, indent=2)
            print(
                f"  seed={seed} success={record['success']} steps_run={record['steps_run']} "
                f"reward_max={record['reward_max']} elapsed_s={record['elapsed_s']} -> {rec_path}"
            )

        del policy, pre, post
        torch.cuda.empty_cache()

    summary_by_checkpoint = {}
    for step in checkpoint_steps:
        recs = [r for r in all_records if r["checkpoint_step"] == step]
        summary_by_checkpoint[str(step)] = {
            "num_rollouts": len(recs),
            "num_success": sum(1 for r in recs if r["success"]),
            "success_seeds": sorted(r["seed"] for r in recs if r["success"]),
            "mean_reward_max": (sum(r["reward_max"] for r in recs) / len(recs)) if recs else None,
        }

    summary = {
        "schema_version": "task1_edge_pinch_v2_blind_eval_v1",
        "mode": mode,
        "checkpoint_steps": checkpoint_steps,
        "seeds": seeds,
        "n_action_steps": args.n_action_steps,
        "wait_steps": args.wait_steps,
        "max_steps": args.max_steps,
        "output_dir": str(output_dir),
        "eval_script_sha256": eval_script_sha256,
        "bddl_sha256": bddl_sha256,
        "num_records": len(all_records),
        "summary_by_checkpoint": summary_by_checkpoint,
        "formal_evaluation": mode == "full_run",
        "smoke_test_only": mode == "smoke_test",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"BLIND_EVAL_{mode.upper()}_COMPLETE records={len(all_records)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
