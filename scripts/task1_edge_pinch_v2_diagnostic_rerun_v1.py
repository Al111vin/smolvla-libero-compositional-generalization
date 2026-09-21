#!/usr/bin/env python3
"""Task1 edge-pinch v2 fine-tune -- layer-2 diagnostic re-run for a small,
explicitly-named set of ALREADY-EVALUATED failed (checkpoint, seed) pairs
from the checkpoint 2500 / 3000 blind-eval.

Purpose (per user 2026-09-17): layer-1 (summary-field) failure analysis hit
its information ceiling -- every one of the 42 failures across checkpoints
2500/3000 was termination_reason=max_steps_reached, steps_run=600,
reward_max=0.0, with zero partial-reward signal, so failure subtype
(never touched the object / grasped then dropped / grasped but placed
wrong) cannot be distinguished from the existing records alone. This
script adds three diagnostic-only recordings to a SMALL re-run (10 total
rollouts: 5 named failed seeds x checkpoint 2500, 5 named failed seeds x
checkpoint 3000) to make that distinction visible:

  1. an agentview keyframe image every 50 control steps (not a full video,
     to control storage)
  2. the full per-step gripper state (robot0_gripper_qpos) sequence
  3. the final akita_black_bowl_1 <-> plate_1 distance, read from
     obs['akita_black_bowl_1_pos'] / obs['plate_1_pos'] -- the same
     observation-dict object-pose keys this project's own
     calibrate_libero_36_push.py already uses (confirmed real keys, not a
     guess: see claude/task1_strict_closedloop_diagnosis_20260915.json
     revision 23, "object_pose_observation_dict_access").

CRITICAL: object-pose reads in this script are DIAGNOSTIC-ONLY, logged
after the fact for a human to read. They are NEVER passed to policy.
select_action() or included in any model input -- the rollout control loop
below is byte-for-byte the same pure-vision (agentview+wrist images +
proprioception only) contract as task1_edge_pinch_v2_blind_eval_v1.py.
Every diagnostic record is tagged object_pose_diagnostic_only=True /
object_pose_not_used_as_model_input=True so this can never be mistaken for
a privileged-input evaluation.

This is a RE-RUN of already-evaluated (checkpoint, seed) pairs -- these
seeds are not "new blind-eval evidence" and this script's results do NOT
change, supersede, or get merged into the existing blind-eval success/
failure counts or the official conclusion already recorded in
claude/task1_edge_pinch_v2_official_conclusion_and_fold02_decision_20260917.json.
Its only purpose is to explain WHY the named failures failed. Fold02
remains locked regardless of what this script finds.

Imports validate_seed/frame/qaxis/make_env_fn/check_policy_api_compat/
sha256_of_file/LANGUAGE_INSTRUCTION/DEFAULT_BDDL_PATH/ACTION_DIM directly
from the already-deployed, already-tested
task1_edge_pinch_v2_blind_eval_v1.py (same directory) rather than
duplicating them, so the control-loop contract and seed gate are
byte-identical/shared, not re-implemented and possibly drifted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task1_edge_pinch_v2_blind_eval_v1 import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_BDDL_PATH,
    LANGUAGE_INSTRUCTION,
    check_policy_api_compat,
    frame,
    make_env_fn,
    resolve_checkpoint_dir,
    sha256_of_file,
    validate_seed,
)

OBJECT_POSE_KEYS = {"bowl": "akita_black_bowl_1_pos", "plate": "plate_1_pos"}
KEYFRAME_INTERVAL_STEPS = 50


def parse_pair(s: str) -> tuple[int, int]:
    step_str, seed_str = s.split(":")
    return int(step_str), int(seed_str)


def _save_keyframe(png_module, obs: dict, out_path: Path) -> bool:
    """Best-effort PNG save of obs['agentview_image']. Returns True on
    success. Never raises -- a missing Pillow install must not crash the
    whole diagnostic run; falls back to a .npy save instead."""
    import numpy as np

    img = np.asarray(obs["agentview_image"], dtype="uint8")
    if png_module is not None:
        png_module.fromarray(img).save(out_path.with_suffix(".png"))
        return True
    else:
        np.save(out_path.with_suffix(".npy"), img)
        return False


def run_one_diagnostic_rollout(
    policy,
    pre,
    post,
    env_fn,
    seed: int,
    lang: str,
    n_action_steps: int,
    wait_steps: int,
    max_steps: int,
    keyframe_dir: Path,
    png_module,
) -> dict[str, Any]:
    import numpy as np
    import torch

    validate_seed(seed)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    env = env_fn()
    try:
        np.random.seed(seed)
        obs = env.reset()

        initial_bowl_pos = list(map(float, obs[OBJECT_POSE_KEYS["bowl"]]))
        initial_plate_pos = list(map(float, obs[OBJECT_POSE_KEYS["plate"]]))

        gripper_qpos_sequence: list[list[float]] = []
        keyframes_saved: list[str] = []
        keyframe_used_png = True

        wait_terminated = False
        wait_success = False
        for _ in range(wait_steps):
            obs, wait_reward, wait_done, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            gripper_qpos_sequence.append(list(map(float, obs["robot0_gripper_qpos"])))
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
            success = True
            termination_reason = "wait_phase_success"
        else:
            policy.reset()
            policy.config.n_action_steps = n_action_steps

            kf_path = keyframe_dir / f"step_{0:04d}"
            ok = _save_keyframe(png_module, obs, kf_path)
            keyframe_used_png = keyframe_used_png and ok
            keyframes_saved.append(str(kf_path.with_suffix(".png" if ok else ".npy").name))

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
                gripper_qpos_sequence.append(list(map(float, obs["robot0_gripper_qpos"])))

                if steps_run % KEYFRAME_INTERVAL_STEPS == 0:
                    kf_path = keyframe_dir / f"step_{steps_run:04d}"
                    ok = _save_keyframe(png_module, obs, kf_path)
                    keyframe_used_png = keyframe_used_png and ok
                    keyframes_saved.append(str(kf_path.with_suffix(".png" if ok else ".npy").name))

                if bool(env.check_success()) or reward > 0:
                    success = True
                    termination_reason = "success_early_termination"
                    break
                if done:
                    termination_reason = "env_done_no_success"
                    break

            # final keyframe regardless of interval, so the last frame is
            # always available even if steps_run isn't a multiple of 50
            kf_path = keyframe_dir / f"step_{steps_run:04d}_final"
            ok = _save_keyframe(png_module, obs, kf_path)
            keyframe_used_png = keyframe_used_png and ok
            keyframes_saved.append(str(kf_path.with_suffix(".png" if ok else ".npy").name))

        final_bowl_pos = list(map(float, obs[OBJECT_POSE_KEYS["bowl"]]))
        final_plate_pos = list(map(float, obs[OBJECT_POSE_KEYS["plate"]]))
        final_bowl_plate_distance_m = float(
            np.linalg.norm(np.asarray(final_bowl_pos) - np.asarray(final_plate_pos))
        )
        bowl_displacement_from_reset_m = float(
            np.linalg.norm(np.asarray(final_bowl_pos) - np.asarray(initial_bowl_pos))
        )

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
            "gripper_qpos_sequence": gripper_qpos_sequence,
            "keyframes_saved": keyframes_saved,
            "keyframes_are_png": keyframe_used_png,
            "keyframe_interval_steps": KEYFRAME_INTERVAL_STEPS,
            "object_pose_diagnostic_only": True,
            "object_pose_not_used_as_model_input": True,
            "initial_bowl_pos": initial_bowl_pos,
            "initial_plate_pos": initial_plate_pos,
            "final_bowl_pos": final_bowl_pos,
            "final_plate_pos": final_plate_pos,
            "final_bowl_plate_distance_m": final_bowl_plate_distance_m,
            "bowl_displacement_from_reset_m": bowl_displacement_from_reset_m,
        }
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finetune-checkpoints-dir", type=str, required=True)
    p.add_argument(
        "--pair", type=str, nargs="+", required=True,
        help="One or more CHECKPOINT_STEP:SEED pairs, e.g. --pair 2500:555103 2500:555105 3000:555101",
    )
    p.add_argument("--bddl-path", type=str, default=DEFAULT_BDDL_PATH)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run only the FIRST pair, to validate the pipeline before committing to the full set. "
        "Mutually exclusive with --confirm-full-run.",
    )
    p.add_argument(
        "--confirm-full-run", action="store_true",
        help="Required to run all named pairs. Mutually exclusive with --smoke-test.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def gate_a_resolve_mode(args) -> str:
    if args.smoke_test and args.confirm_full_run:
        raise ValueError("--smoke-test and --confirm-full-run are mutually exclusive; pass exactly one")
    if not args.smoke_test and not args.confirm_full_run:
        raise ValueError(
            "refusing to run without an explicit mode: pass --smoke-test (first pair only, "
            "pipeline validation) or --confirm-full-run (all named pairs)"
        )
    return "smoke_test" if args.smoke_test else "full_run"


def gate_c_check_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output_dir {output_dir} already exists and is non-empty; pass --overwrite to reuse "
            "it, or choose a new path (this script never deletes existing files itself)"
        )


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        mode = gate_a_resolve_mode(args)
        pairs = [parse_pair(s) for s in args.pair]
        for step, seed in pairs:
            validate_seed(seed)
        if mode == "smoke_test":
            pairs = pairs[:1]
            print(f"SMOKE_TEST_MODE: restricting to pair={pairs[0]}")

        unique_steps = sorted({step for step, _ in pairs})
        checkpoint_dirs = {
            step: resolve_checkpoint_dir(args.finetune_checkpoints_dir, step)
            for step in unique_steps
        }
        output_dir = Path(args.output_dir)
        gate_c_check_output_dir(output_dir, args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image as png_module
    except ImportError:
        png_module = None
        print("WARNING: Pillow not importable -- keyframes will be saved as .npy instead of .png")

    import torch
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    eval_script_sha256 = sha256_of_file(__file__)
    bddl_sha256 = sha256_of_file(args.bddl_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_fn = make_env_fn(args.bddl_path)

    all_records: list[dict[str, Any]] = []
    policies_cache: dict[int, Any] = {}

    for step, seed in pairs:
        if step not in policies_cache:
            ckpt_dir = checkpoint_dirs[step]
            model_file = ckpt_dir / "model.safetensors"
            if not model_file.is_file():
                print(f"REFUSING: checkpoint {step} missing model.safetensors at {model_file}")
                return 1
            print(f"Loading checkpoint step={step} from {ckpt_dir} ...")
            policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
            check_policy_api_compat(policy)
            pre, post = make_pre_post_processors(
                policy.config, pretrained_path=str(ckpt_dir),
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            policies_cache[step] = (policy, pre, post, sha256_of_file(model_file))
        policy, pre, post, ckpt_model_sha256 = policies_cache[step]

        ckpt_out_dir = output_dir / f"checkpoint_{step:06d}"
        keyframe_dir = ckpt_out_dir / f"seed_{seed}_keyframes"

        t0 = time.time()
        result = run_one_diagnostic_rollout(
            policy, pre, post, env_fn, seed, LANGUAGE_INSTRUCTION,
            n_action_steps=args.n_action_steps, wait_steps=args.wait_steps,
            max_steps=args.max_steps, keyframe_dir=keyframe_dir, png_module=png_module,
        )
        elapsed_s = time.time() - t0

        record = {
            "checkpoint_step": step,
            "checkpoint_model_sha256": ckpt_model_sha256,
            "bddl_sha256": bddl_sha256,
            "eval_script_sha256": eval_script_sha256,
            "diagnostic_rerun_of_known_failed_pair": True,
            "elapsed_s": round(elapsed_s, 3),
            "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **result,
        }
        all_records.append(record)

        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = ckpt_out_dir / f"seed_{seed}_diagnostic.json"
        with open(rec_path, "w") as f:
            json.dump(record, f, indent=2)
        print(
            f"  checkpoint={step} seed={seed} success={record['success']} "
            f"steps_run={record['steps_run']} final_bowl_plate_distance_m="
            f"{record['final_bowl_plate_distance_m']:.4f} bowl_displacement_m="
            f"{record['bowl_displacement_from_reset_m']:.4f} elapsed_s={record['elapsed_s']} "
            f"-> {rec_path}"
        )

    for policy, pre, post, _ in policies_cache.values():
        del policy, pre, post
    import torch as _torch
    _torch.cuda.empty_cache()

    summary = {
        "schema_version": "task1_edge_pinch_v2_diagnostic_rerun_v1",
        "mode": mode,
        "pairs": [f"{s}:{se}" for s, se in pairs],
        "output_dir": str(output_dir),
        "eval_script_sha256": eval_script_sha256,
        "bddl_sha256": bddl_sha256,
        "num_records": len(all_records),
        "records_summary": [
            {"checkpoint_step": r["checkpoint_step"], "seed": r["seed"], "success": r["success"],
             "final_bowl_plate_distance_m": r["final_bowl_plate_distance_m"],
             "bowl_displacement_from_reset_m": r["bowl_displacement_from_reset_m"]}
            for r in all_records
        ],
        "not_new_blind_eval_evidence": True,
        "does_not_change_official_conclusion": True,
        "fold02_gate": "LOCKED -- unaffected by this diagnostic run",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"DIAGNOSTIC_RERUN_{mode.upper()}_COMPLETE records={len(all_records)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
