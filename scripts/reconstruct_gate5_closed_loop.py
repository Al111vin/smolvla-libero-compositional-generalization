"""Run isolated Gate-5 closed-loop reconstruction candidates.

This runner deliberately keeps its artifacts outside the formal Gate-5
evidence directory.  A candidate is promotable only after the normal
calibrator has recorded, replayed, and accepted every safety and terminal
check.  The 29 already-passed task records are never opened or overwritten.

The remaining source layouts form a vertical obstacle corridor.  A direct
source-to-goal push is therefore not a safe primitive: it eventually brings
the wrist or opposite finger into a neighbouring object.  Each candidate here
uses an explicit three-stage route: make lateral clearance, travel through
the clear lane, then make the small terminal correction.  The underlying
controller re-anchors the selected pad to the live target pose at every push
step, rather than continuing an open-loop end-effector displacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts import calibrate_libero_36_push as calibrator


# These are controller waypoints, not goal bounds.  Goal validation remains in
# calibrate_libero_36_push and uses the frozen Gate-4 task semantics.
TASK_SPECS = {
    # The mug is tall.  Its route first clears the bowl/can corridor, then
    # returns to the requested x lane only after it is past the neighbour.
    15: dict(
        # Entry probe: the left pad reaches the mug body at this low,
        # rotated pose (11 mm maximum target lift) while every right-pad
        # candidate reached only the handle / finger body.
        pusher_finger="left", pad_height_offset_m=0.010,
        route=[[-0.135, 0.065], [-0.135, -0.205], [-0.120, -0.210]],
        yaw=-45.0,
        allow_approximate_behind=True,
        approach_standoff_m=0.025,
        rotation_aware_pad_target=True,
        side_step_approach=True,
    ),
    16: dict(
        pusher_finger="right", pad_height_offset_m=0.035,
        route=[[-0.125, 0.075], [-0.125, 0.000], [-0.120, 0.000]],
        yaw=-35.0,
    ),
    17: dict(
        pusher_finger="left", pad_height_offset_m=0.035,
        route=[[-0.135, 0.065], [-0.135, 0.205], [-0.120, 0.210]],
        yaw=35.0,
    ),
    # Alphabet soup begins above the corridor.  The first node moves it west
    # while it is still clear of the mug, avoiding the old diagonal collision.
    24: dict(
        pusher_finger="right", pad_height_offset_m=0.010,
        route=[[-0.080, 0.140], [-0.190, 0.040], [-0.180, -0.160]],
        yaw=-60.0,
        rotate_high_before_descend=True,
        allow_approximate_behind=True,
        penetration=0.008,
    ),
    25: dict(
        pusher_finger="right", pad_height_offset_m=0.005,
        # Measured from v1: a direct westward contact path reaches this
        # clearance point before the wrist enters the mug corridor.  Making
        # the turn here keeps the second segment physically reachable.
        route=[[-0.080, 0.170], [-0.135, 0.000], [-0.120, 0.000]],
        yaw=-90.0,
        rotate_high_before_descend=True,
        penetration=0.001,
    ),
    33: dict(
        pusher_finger="right", pad_height_offset_m=0.010,
        route=[[-0.120, -0.212], [-0.120, -0.210]],
        yaw=-32.0,
        rotate_high_before_descend=True,
        penetration=0.003,
    ),
    35: dict(
        pusher_finger="right", pad_height_offset_m=0.010,
        route=[[-0.125, -0.212], [-0.135, 0.205], [-0.120, 0.210]],
        yaw=-32.0,
    ),
}


def candidate_for(task_id: int) -> dict:
    spec = TASK_SPECS[task_id]
    route = spec["route"]
    # Trigger each transition slightly before the nominal waypoint.  This is
    # intentional: the live target state, not a presumed displacement, decides
    # when the route changes.
    triggers = [route[0], route[1]]
    candidate = {
        "candidate_index": 0,
        "pusher_finger": spec["pusher_finger"],
        "pad_height_offset_m": spec["pad_height_offset_m"],
        "desired_yaw_degrees": spec["yaw"],
        "candidate_order_key": f"closed_loop_corridor_task_{task_id}",
        "controller_route_xy": route,
        "route_transition_trigger_xy": triggers,
        "route_transition_trigger_tolerance_m": 0.018,
        "controller_destination_tolerance_m": 0.010,
        "contact_follow": True,
        "contact_follow_feedback": True,
        "feedback_destination_xy": spec.get("feedback_route", route),
        "contact_follow_penetration_m": spec.get("penetration", 0.003),
        "route_contact_follow_penetration_m": spec.get("penetration", 0.003),
        "push_iterations": 650,
        "push_inner_steps": 1,
        # v6 keeps the simulator's native horizon, rather than applying the
        # v5 diagnostic controller cap.  Auxiliary non-grasping target
        # contact is recorded and later judged by the v6 verifier; grasp,
        # lifting, distractor contact, and object contact remain hard stops.
        "active_action_limit": None,
        "allow_auxiliary_target_contact": True,
        # At a corner the physical contact normal must rotate.  The base
        # runner changes it only after the live waypoint trigger is met.
        "route_transition_yaw_delta_degrees": [0.0, 0.0],
    }
    if "approach_lateral_offset_xy" in spec:
        candidate["approach_lateral_offset_xy"] = spec[
            "approach_lateral_offset_xy"
        ]
    if spec.get("rotate_high_before_descend", False):
        candidate["rotate_high_before_descend"] = True
    if spec.get("allow_approximate_behind", False):
        candidate["allow_approximate_behind"] = True
    if spec.get("side_step_approach", False):
        candidate["side_step_approach"] = True
    if "approach_standoff_m" in spec:
        candidate["approach_standoff_m"] = spec["approach_standoff_m"]
    if spec.get("rotation_aware_pad_target", False):
        candidate["rotation_aware_pad_target"] = True
    if "push_direction_yaw_degrees" in spec:
        candidate["push_direction_yaw_degrees"] = spec[
            "push_direction_yaw_degrees"
        ]
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated Gate-5 task-level closed-loop candidate"
    )
    parser.add_argument("--task-id", type=int, choices=sorted(TASK_SPECS))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def candidates(row: dict) -> list[dict]:
        if int(row["task_id"]) != args.task_id:
            raise RuntimeError(f"unexpected task row: {row['task_id']}")
        return [candidate_for(args.task_id)]

    calibrator.candidates_for = candidates
    sys.argv = [
        "reconstruct_gate5_closed_loop.py",
        "--mode", "smoke",
        "--layout-spec", "data/libero_36/layout_spec.csv",
        "--reset-manifest", "results/libero_36_reset_audit.manifest.json",
        "--goal-manifest", "results/libero_36_goal_audit.manifest.json",
        "--task-ids", str(args.task_id),
        "--output", str(root / "calibration.csv"),
        "--manifest", str(root / "calibration.manifest.json"),
        "--attempts-dir", str(root / "attempts"),
        "--overwrite",
    ]
    calibrator.main()


if __name__ == "__main__":
    main()
