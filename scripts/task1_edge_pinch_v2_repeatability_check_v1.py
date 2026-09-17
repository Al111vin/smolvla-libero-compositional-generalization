#!/usr/bin/env python3
"""Task1 edge-pinch v2 fine-tune -- repeatability check for the two
(checkpoint, seed) pairs whose success/failure flipped between independent
runs of the SAME protocol (2026-09-17 finding, see
claude/task1_edge_pinch_v2_diagnostic_rerun_result_and_nondeterminism_finding_20260917.json):

  - checkpoint 2500, seed 555103: False (original blind eval) -> False (this
    project's diagnostic-rerun smoke test) -> True (this project's
    diagnostic-rerun full run)
  - checkpoint 3000, seed 555102: False (original blind eval) -> True (this
    project's diagnostic-rerun full run)

Purpose (per user 2026-09-17): repeat EACH of these two pairs 5 times, with
the SAME seed every repeat (not seed+i -- the point is to test whether an
identical (checkpoint, seed, protocol) setup produces different outcomes
across repeated executions, which would indicate execution-level
non-determinism rather than intentional seed variation), and explicitly
fix + record the determinism-relevant settings the user asked for:
torch.manual_seed(seed) + torch.cuda.manual_seed_all(seed),
policy.eval() (already the case via SmolVLAPolicy.from_pretrained(...).eval()
in the existing blind-eval/diagnostic scripts), torch.inference_mode()
(already used at the select_action call site),
torch.backends.cudnn.benchmark=False, and an attempt at
torch.use_deterministic_algorithms(True) (best-effort -- some ops may not
have a deterministic CUDA implementation; failure is caught and recorded,
never silently swallowed).

Per repeat, records: success, steps_run, final bowl/plate diagnostic
(reusing the same diagnostic-only object-pose read as
task1_edge_pinch_v2_diagnostic_rerun_v1.py), and the raw post-processed
action vector at a fixed set of control-step indices (1, 50, 100, 200, 400,
and the final step actually reached) so the trajectories can be compared
step-by-step across repeats, not just by final outcome.

If outcomes still differ across the 5 repeats even with cudnn.benchmark
disabled and deterministic algorithms requested, this is reported as
"execution-level randomness" (per user's phrasing: 报告为'执行级随机性').
If all 5 repeats agree with each other (whether or not they match the
original blind-eval label), that indicates the earlier flip was a
determinism-CONFIGURATION issue (e.g. cudnn.benchmark=True was in effect
during the original runs) that this script's settings fix.

This script does NOT change, supersede, or get merged into any existing
blind-eval success/failure counts, the Wilson CI analysis, or the official
conclusion already recorded. It also does NOT recompute or re-report the
existing 25-seed-per-checkpoint Wilson intervals -- per explicit user
instruction, those remain valid as historical per-rollout observations and
must not be reinterpreted as fixed per-seed determinism labels. Fold02
remains locked regardless of what this script finds.

Reuses validate_seed/frame/qaxis/make_env_fn/check_policy_api_compat/
resolve_checkpoint_dir/sha256_of_file/LANGUAGE_INSTRUCTION/
DEFAULT_BDDL_PATH/ACTION_DIM from the already-deployed
task1_edge_pinch_v2_blind_eval_v1.py (same directory), and the same
diagnostic-only object-pose keys (akita_black_bowl_1_pos / plate_1_pos)
already used and confirmed working in
task1_edge_pinch_v2_diagnostic_rerun_v1.py.
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
# Must be set before the first CUDA context/op (i.e. before torch is ever
# imported and touches the GPU) -- torch.use_deterministic_algorithms(True)
# alone does NOT make CuBLAS matmul deterministic on CUDA>=10.2; this repo's
# smoke test surfaced PyTorch's own warning confirming exactly this gap
# (att_weights = torch.matmul(...) in smolvlm_with_expert.py is a CuBLAS op).
# See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
ACTION_SNAPSHOT_STEPS = (1, 50, 100, 200, 400)


def parse_pair(s: str) -> tuple[int, int]:
    step_str, seed_str = s.split(":")
    return int(step_str), int(seed_str)


def apply_determinism_settings() -> dict[str, Any]:
    """Best-effort: fixes torch.manual_seed/cuda seed, disables
    cudnn.benchmark, and attempts torch.use_deterministic_algorithms(True).
    Returns a dict recording exactly what was applied/observed, so this is
    never a silent no-op -- a failure to enforce determinism is data, not
    something to hide."""
    import torch

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["cudnn_version"] = torch.backends.cudnn.version()

    report["cudnn_benchmark_before"] = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    report["cudnn_benchmark_after"] = torch.backends.cudnn.benchmark

    report["cudnn_deterministic_before"] = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    report["cudnn_deterministic_after"] = torch.backends.cudnn.deterministic

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        report["use_deterministic_algorithms_requested"] = True
        report["use_deterministic_algorithms_error"] = None
    except Exception as exc:  # pragma: no cover -- environment-dependent
        report["use_deterministic_algorithms_requested"] = False
        report["use_deterministic_algorithms_error"] = f"{type(exc).__name__}: {exc}"

    return report


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_repeatability_rollout(
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
        # np.random.seed AND torch seeds are both re-applied here,
        # immediately before env.reset(), so every repeat starts from the
        # identical seeded state -- not just at process start.
        seed_everything(seed)
        obs = env.reset()

        initial_bowl_pos = list(map(float, obs[OBJECT_POSE_KEYS["bowl"]]))
        initial_plate_pos = list(map(float, obs[OBJECT_POSE_KEYS["plate"]]))

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
        action_snapshots: dict[str, list[float]] = {}

        if wait_terminated:
            termination_reason = "wait_phase_done"
        elif wait_success:
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

                if steps_run in ACTION_SNAPSHOT_STEPS:
                    action_snapshots[str(steps_run)] = [round(float(x), 6) for x in a.tolist()]

                if bool(env.check_success()) or reward > 0:
                    success = True
                    termination_reason = "success_early_termination"
                    break
                if done:
                    termination_reason = "env_done_no_success"
                    break

            action_snapshots[f"final_{steps_run}"] = (
                [round(float(x), 6) for x in a.tolist()] if steps_run > 0 else None
            )

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
            "steps_run": steps_run,
            "invalid_action_detected": invalid_action,
            "termination_reason": termination_reason,
            "action_snapshots": action_snapshots,
            "object_pose_diagnostic_only": True,
            "object_pose_not_used_as_model_input": True,
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
        help="One or more CHECKPOINT_STEP:SEED pairs to repeat, e.g. --pair 2500:555103 3000:555102",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--bddl-path", type=str, default=DEFAULT_BDDL_PATH)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run only 1 repeat of the FIRST pair. Mutually exclusive with --confirm-full-run.",
    )
    p.add_argument(
        "--confirm-full-run", action="store_true",
        help="Required to run all repeats of all named pairs. Mutually exclusive with --smoke-test.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def gate_a_resolve_mode(args) -> str:
    if args.smoke_test and args.confirm_full_run:
        raise ValueError("--smoke-test and --confirm-full-run are mutually exclusive; pass exactly one")
    if not args.smoke_test and not args.confirm_full_run:
        raise ValueError(
            "refusing to run without an explicit mode: pass --smoke-test (1 repeat of the first "
            "pair, pipeline validation) or --confirm-full-run (all repeats of all named pairs)"
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
        repeats = args.repeats
        if mode == "smoke_test":
            pairs = pairs[:1]
            repeats = 1
            print(f"SMOKE_TEST_MODE: restricting to pair={pairs[0]}, 1 repeat")

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

    import torch
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    determinism_report = apply_determinism_settings()
    print("DETERMINISM_SETTINGS:", json.dumps(determinism_report, indent=2))

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

        pair_records = []
        for repeat_idx in range(repeats):
            t0 = time.time()
            result = run_one_repeatability_rollout(
                policy, pre, post, env_fn, seed, LANGUAGE_INSTRUCTION,
                n_action_steps=args.n_action_steps, wait_steps=args.wait_steps,
                max_steps=args.max_steps,
            )
            elapsed_s = time.time() - t0

            record = {
                "checkpoint_step": step,
                "seed": seed,
                "repeat_index": repeat_idx,
                "checkpoint_model_sha256": ckpt_model_sha256,
                "bddl_sha256": bddl_sha256,
                "eval_script_sha256": eval_script_sha256,
                "determinism_settings": determinism_report,
                "elapsed_s": round(elapsed_s, 3),
                "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **result,
            }
            all_records.append(record)
            pair_records.append(record)

            ckpt_out_dir = output_dir / f"checkpoint_{step:06d}"
            ckpt_out_dir.mkdir(parents=True, exist_ok=True)
            rec_path = ckpt_out_dir / f"seed_{seed}_repeat_{repeat_idx}.json"
            with open(rec_path, "w") as f:
                json.dump(record, f, indent=2)
            print(
                f"  checkpoint={step} seed={seed} repeat={repeat_idx} success={record['success']} "
                f"steps_run={record['steps_run']} final_bowl_plate_distance_m="
                f"{record['final_bowl_plate_distance_m']:.4f} elapsed_s={record['elapsed_s']} "
                f"-> {rec_path}"
            )

        successes = [r["success"] for r in pair_records]
        print(
            f"PAIR_SUMMARY checkpoint={step} seed={seed} repeats={len(pair_records)} "
            f"successes={sum(successes)} all_agree={len(set(successes)) == 1} "
            f"success_pattern={successes}"
        )

    del policy, pre, post
    torch.cuda.empty_cache()

    per_pair_summary = {}
    for step, seed in {(r["checkpoint_step"], r["seed"]) for r in all_records}:
        recs = [r for r in all_records if r["checkpoint_step"] == step and r["seed"] == seed]
        successes = [r["success"] for r in recs]
        per_pair_summary[f"{step}:{seed}"] = {
            "repeats": len(recs),
            "successes": sum(successes),
            "success_pattern": successes,
            "all_repeats_agree": len(set(successes)) == 1,
        }

    summary = {
        "schema_version": "task1_edge_pinch_v2_repeatability_check_v1",
        "mode": mode,
        "pairs": [f"{s}:{se}" for s, se in pairs],
        "repeats_per_pair": repeats,
        "determinism_settings": determinism_report,
        "output_dir": str(output_dir),
        "eval_script_sha256": eval_script_sha256,
        "bddl_sha256": bddl_sha256,
        "num_records": len(all_records),
        "per_pair_summary": per_pair_summary,
        "not_new_blind_eval_evidence": True,
        "does_not_recompute_existing_wilson_ci": True,
        "does_not_change_official_conclusion": True,
        "fold02_gate": "LOCKED -- unaffected by this repeatability check",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"REPEATABILITY_CHECK_{mode.upper()}_COMPLETE records={len(all_records)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
