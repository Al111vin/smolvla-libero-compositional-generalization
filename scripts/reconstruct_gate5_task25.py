"""Task-25 closed-loop reconstruction with a validated single-pad prefix.

This is an isolated diagnostic runner.  It never writes to formal evidence.
The prefix brings only the right fingerpad to alphabet soup from the safe
north-east lane; after that, each command is derived from the live target pose
and a bounded contact offset.  The action budget, contact rules, terminal
hold, and deterministic replay are the unchanged Gate-5 checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c


TASK_ID = 25
SELECTED_SIDE = "right"
GOAL_XY = np.array([-0.12, 0.0], dtype=np.float64)
ROOT = Path("/tmp/libero36_closedloop_task25_reanchor_v1")
CANONICAL = Path(
    "results/libero36_source_xy_redesign_diagnostic/gate5/"
    "task26_layout1_two_leg.npz"
)


def task_row() -> dict:
    rows = c.reset_validator.read_layout_spec(
        Path("data/libero_36/layout_spec.csv")
    )
    return next(
        row
        for row in rows
        if int(row["task_id"]) == TASK_ID and int(row["layout_id"]) == 1
    )


def trace_safety(arrays: dict[str, np.ndarray]) -> dict[str, bool]:
    return {
        "no_bilateral_or_grasp": not bool(
            np.any(arrays["left_contact"] & arrays["right_contact"])
            or arrays["grasp_proxy"].any()
        ),
        "right_pad_only": not bool(
            arrays["nonselected_robot_target_contact"].any()
        ),
        "no_robot_distractor": not bool(
            arrays["robot_distractor_contact"].any()
        ),
        "no_unexpected_object": not bool(
            arrays["unexpected_object_contact"].any()
        ),
        "table_supported": bool(arrays["table_support"].all()),
    }


def main() -> None:
    if not CANONICAL.is_file():
        raise FileNotFoundError(CANONICAL)
    ROOT.mkdir(parents=True, exist_ok=True)
    row = task_row()
    seed = c.reset_validator.seed_for(
        row, c.GATE3_CALIBRATION_RESET_INDEX, c.reset_validator.FULL_SEED_BASE
    )
    reference = np.load(CANONICAL, allow_pickle=True)
    prefix_mask = np.asarray(reference["phase"]).astype(str) == "leg1_replay"
    prefix_actions = np.asarray(reference["actions"], dtype=np.float32)[
        prefix_mask
    ].copy()
    prefix_target = np.asarray(reference["target_xyz"], dtype=np.float64)[
        prefix_mask
    ]
    if len(prefix_actions) != 238:
        raise RuntimeError("canonical single-pad prefix shape changed")

    target = c.target_instance(row)
    env = c.reset_validator.make_environment(Path(row["bddl_path"]))
    try:
        settled = c.settle_environment(env, row, seed)
        obs = settled["obs"]
        geometry = c.contact_geometry(env, target)
        recorder = c.AttemptRecorder(
            env, row, target, geometry, SELECTED_SIDE, settled["stabilized_z"]
        )
        start = c.target_xyz(obs, target).copy()
        pads = c.finger_positions(env.env, geometry)
        # The +18 mm north shift has a deterministic, independently replayed
        # safety record and keeps the gripper hand clear of the mug.
        translation = start - prefix_target[0]
        translation[1] += 0.018
        obs, shift_ok, shift_diagnostics = c.move_pad_to(
            recorder,
            obs,
            geometry,
            SELECTED_SIDE,
            pads[SELECTED_SIDE] + translation,
            settled["desired_rotation"],
            "safe_prefix_shift",
            60,
            0.14,
        )
        if not shift_ok:
            raise RuntimeError(f"safe prefix shift failed: {shift_diagnostics}")
        # Retain the known safe contact acquisition but not its former
        # open-loop continuation.  This is a task-local access primitive.
        prefix_actions[100:, 1] *= 1.75
        for action in prefix_actions:
            obs = recorder.step(obs, action, "safe_prefix_contact")

        pad = c.finger_positions(env.env, geometry)[SELECTED_SIDE]
        contact_offset = pad[:2] - c.target_xyz(obs, target)[:2]
        # Do not command the wrist back to the reset orientation here.  The
        # prefix acquired its safe right-pad contact at a rotated wrist pose;
        # returning to reset posture was sufficient to lose that contact.
        contact_rotation = c.eef_rotation(obs).copy()
        contact_offset_norm = float(np.linalg.norm(contact_offset))
        if not (0.020 <= contact_offset_norm <= 0.050):
            raise RuntimeError(f"unsafe reanchor offset: {contact_offset_norm}")

        # 249 prefix actions + 640 feedback actions + 20 holds <= 930.
        for _ in range(640):
            live = c.target_xyz(obs, target)
            relation, xy_ok = c.target_region_status(env, obs, row)
            if relation and xy_ok:
                break
            delta = GOAL_XY - live[:2]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6:
                break
            direction = delta / distance
            desired_pad = pad.copy()
            desired_pad[:2] = live[:2] + contact_offset + direction * 0.0025
            # Keep the contact on the right *pad*, rather than allowing the
            # finger body to climb over the can as the target reacts to push
            # forces.  The 10 mm offset is the same safe contact height used
            # by the prefix, expressed against the live object height.
            desired_pad[2] = live[2] + 0.010
            desired_eef = c.desired_eef_for_pad(
                env, obs, geometry, SELECTED_SIDE, desired_pad
            )
            action = c.osc_action(
                obs, desired_eef, contact_rotation, 0.25
            )
            # The offset feedback alone reacts one simulator step late and
            # lets the target drift toward the opposite (left) finger.  A
            # bounded feed-forward term keeps the selected pad travelling in
            # the same direction as the intended target displacement.
            action[:2] = np.clip(
                action[:2] + direction * 0.12, -0.35, 0.35
            )
            obs = recorder.step(obs, action, "live_reanchor_push")
            pad = c.finger_positions(env.env, geometry)[SELECTED_SIDE]
            contact_offset = pad[:2] - c.target_xyz(obs, target)[:2]
            # Stop before a safety violation can turn into an extended trace.
            if (
                recorder.nonselected_robot_target_contact[-1]
                or recorder.robot_distractor_contact[-1]
                or recorder.unexpected_object_contact[-1]
                or recorder.grasp_proxy[-1]
            ):
                break

        for _ in range(c.TERMINAL_HOLD_STEPS):
            obs = recorder.step(
                obs, np.zeros(7, dtype=np.float32), "terminal_hold"
            )
        arrays = recorder.arrays()
    finally:
        env.close()

    replay = c.replay_attempt(row, seed, arrays["actions"], SELECTED_SIDE)
    replay_states = np.asarray(replay["states"], dtype=np.float64)
    state_exact = bool(
        arrays["states"].shape == replay_states.shape
        and np.array_equal(arrays["states"], replay_states)
    )
    trace_exact = c.trace_matches(arrays, replay)
    hold = slice(-c.TERMINAL_HOLD_STEPS, None)
    checks = trace_safety(arrays)
    checks.update(
        {
            "selected_right_contact": bool(arrays["right_contact"].any()),
            "terminal_relation": bool(arrays["relation"][hold].all()),
            "terminal_xy": bool(arrays["xy_in_target"][hold].all()),
            "terminal_support": bool(arrays["table_support"][hold].all()),
            "exact_replay": state_exact and trace_exact,
            "within_action_budget": len(arrays["actions"]) <= c.MAX_ACTIVE_ACTIONS,
        }
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    np.savez_compressed(ROOT / "trajectory.npz", **arrays)
    np.savez_compressed(
        ROOT / "replay.npz",
        **{key: value for key, value in replay.items() if isinstance(value, np.ndarray)},
    )
    summary = {
        "task_id": TASK_ID,
        "actions": len(arrays["actions"]),
        "final_target_xyz": arrays["target_xyz"][-1].tolist(),
        "checks": checks,
        "failed_checks": failed,
        "diagnostic_only": True,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
