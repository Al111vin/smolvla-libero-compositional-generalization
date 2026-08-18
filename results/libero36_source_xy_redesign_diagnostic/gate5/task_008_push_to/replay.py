from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as calibrator
from scripts import validate_libero_36_envs as validator


ROOT = Path(__file__).resolve().parent
TRAJECTORY = ROOT / "trajectory.npz"
REPLAY = ROOT / "replay.npz"
SUMMARY = ROOT / "replay_summary.json"

TASK_ID = 8
LAYOUT_ID = 1
SELECTED_SIDE = "left"
MAX_ACTIVE_ACTIONS = 930
TERMINAL_HOLD_STEPS = 20


rows = validator.read_layout_spec(
    Path("data/libero_36/layout_spec.csv")
)

matches = [
    row
    for row in rows
    if int(row["task_id"]) == TASK_ID
    and int(row["layout_id"]) == LAYOUT_ID
]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one Task 8 layout-1 row, "
        f"found {len(matches)}"
    )

row = matches[0]

seed = validator.seed_for(
    row,
    calibrator.GATE3_CALIBRATION_RESET_INDEX,
    validator.FULL_SEED_BASE,
)

reference = np.load(
    TRAJECTORY,
    allow_pickle=True,
)

actions = np.asarray(
    reference["actions"],
    dtype=np.float32,
)

replayed = calibrator.replay_attempt(
    row,
    seed,
    actions,
    SELECTED_SIDE,
)

array_keys = [
    "actions",
    "states",
    "left_contact",
    "right_contact",
    "grasp_proxy",
    "table_support",
    "unexpected_object_contact",
    "robot_distractor_contact",
    "nonselected_robot_target_contact",
    "relation",
    "xy_in_target",
]

missing_reference = [
    key
    for key in array_keys
    if key not in reference.files
]

missing_replay = [
    key
    for key in array_keys
    if key not in replayed
]

if missing_reference or missing_replay:
    raise RuntimeError(
        "Missing replay arrays: "
        f"reference={missing_reference}, "
        f"replay={missing_replay}"
    )

trace_matches = {}

for key in array_keys:
    expected = np.asarray(reference[key])
    actual = np.asarray(replayed[key])

    trace_matches[key] = bool(
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and np.array_equal(expected, actual)
    )

reference_states = np.asarray(
    reference["states"],
    dtype=np.float64,
)
replay_states = np.asarray(
    replayed["states"],
    dtype=np.float64,
)

if reference_states.shape != replay_states.shape:
    initial_state_diff = float("inf")
    trajectory_state_diff = float("inf")
else:
    initial_state_diff = float(
        np.max(
            np.abs(
                reference_states[0]
                - replay_states[0]
            )
        )
    )

    trajectory_state_diff = float(
        np.max(
            np.abs(
                reference_states
                - replay_states
            )
        )
    )

left = np.asarray(
    replayed["left_contact"],
    dtype=bool,
)
right = np.asarray(
    replayed["right_contact"],
    dtype=bool,
)
grasp = np.asarray(
    replayed["grasp_proxy"],
    dtype=bool,
)
support = np.asarray(
    replayed["table_support"],
    dtype=bool,
)
unexpected = np.asarray(
    replayed["unexpected_object_contact"],
    dtype=bool,
)
distractor = np.asarray(
    replayed["robot_distractor_contact"],
    dtype=bool,
)
nonselected = np.asarray(
    replayed[
        "nonselected_robot_target_contact"
    ],
    dtype=bool,
)
relation = np.asarray(
    replayed["relation"],
    dtype=bool,
)
xy_in_target = np.asarray(
    replayed["xy_in_target"],
    dtype=bool,
)

hold_slice = slice(
    -TERMINAL_HOLD_STEPS,
    None,
)

checks = {
    "actions_within_safety_horizon": bool(
        len(actions) <= MAX_ACTIVE_ACTIONS
    ),
    "actions_exact": bool(
        trace_matches["actions"]
    ),
    "initial_state_exact": bool(
        initial_state_diff == 0.0
    ),
    "trajectory_state_exact": bool(
        trajectory_state_diff == 0.0
    ),
    "all_recorded_traces_exact": bool(
        all(trace_matches.values())
    ),
    "selected_pad_contact_observed": bool(
        left.any()
    ),
    "opposite_pad_never_contacts": bool(
        not right.any()
    ),
    "no_bilateral_contact_or_grasp": bool(
        not grasp.any()
    ),
    "table_support_all_actions": bool(
        support.all()
    ),
    "no_unexpected_object_contact": bool(
        not unexpected.any()
    ),
    "no_robot_distractor_contact": bool(
        not distractor.any()
    ),
    "no_nonselected_robot_target_contact": bool(
        not nonselected.any()
    ),
    "goal_acquired": bool(
        relation.any()
    ),
    "terminal_relation_hold": bool(
        len(relation) >= TERMINAL_HOLD_STEPS
        and relation[hold_slice].all()
    ),
    "terminal_xy_hold": bool(
        len(xy_in_target)
        >= TERMINAL_HOLD_STEPS
        and xy_in_target[hold_slice].all()
    ),
    "terminal_support_hold": bool(
        len(support) >= TERMINAL_HOLD_STEPS
        and support[hold_slice].all()
    ),
}

failed_checks = [
    key
    for key, value in checks.items()
    if not value
]

replay_passed = not failed_checks

save_arrays = {
    key: np.asarray(replayed[key])
    for key in array_keys
}

np.savez_compressed(
    REPLAY,
    **save_arrays,
)

summary = {
    "task_id": TASK_ID,
    "layout_id": LAYOUT_ID,
    "seed": int(seed),
    "selected_side": SELECTED_SIDE,
    "actions": int(len(actions)),
    "terminal_hold_steps": (
        TERMINAL_HOLD_STEPS
    ),
    "initial_state_max_abs_diff": (
        initial_state_diff
    ),
    "trajectory_state_max_abs_diff": (
        trajectory_state_diff
    ),
    "selected_pad_contact_steps": int(
        left.sum()
    ),
    "opposite_pad_contact_steps": int(
        right.sum()
    ),
    "support_failure_steps": int(
        (~support).sum()
    ),
    "unexpected_object_contact_steps": int(
        unexpected.sum()
    ),
    "robot_distractor_contact_steps": int(
        distractor.sum()
    ),
    "nonselected_robot_target_contact_steps": int(
        nonselected.sum()
    ),
    "relation_true_steps": int(
        relation.sum()
    ),
    "xy_true_steps": int(
        xy_in_target.sum()
    ),
    "trace_matches": trace_matches,
    "checks": checks,
    "failed_checks": failed_checks,
    "replay_passed": replay_passed,
    "trajectory": str(
        TRAJECTORY.resolve()
    ),
    "replay": str(
        REPLAY.resolve()
    ),
}

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
)
print("replay:", REPLAY)
print("summary:", SUMMARY)

if not replay_passed:
    raise RuntimeError(
        f"Task 8 replay failed: {failed_checks}"
    )
