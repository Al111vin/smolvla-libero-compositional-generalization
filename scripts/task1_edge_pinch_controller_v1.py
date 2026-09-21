#!/usr/bin/env python3
"""task1_edge_pinch_controller_v1.py

Edge/single-point grasp controller for Task1 (akita_black_bowl_1), designed
from the official-successful-demo audit evidence in
claude/task1_privileged_pickplace_controller_design_v1.json revision 11:
  - the official demo uses a real, large, sustained orientation change
    (NOT a fixed top-down pose, which is what task1_privileged_pickplace_
    controller_v1.py's compute_action() always used, action[3:6]=0.0);
  - the official demo achieves bilateral left_fingerpad+right_fingerpad
    contact at a final spacing of ~0.019-0.024m, far narrower than the
    bowl's rim diameter (0.1359m, v9) -- consistent with grasping a local
    edge/segment of the rim, not bracketing the whole object.

STATUS: NEW SCRIPT. Not yet executed against the simulator anywhere.
Per explicit user instruction (2026-09-16): this delivery is CODE ONLY --
py_compile + offline/mocked unit tests. No simulation steps are to be run
on the GPU with this script yet. --probe-orientation and --calibrate are
both separate, explicitly-confirmed future steps.

============================================================================
TWO HARD PROTECTIONS REQUIRED BY THE USER BEFORE ANY CODE STAGE (2026-09-16)
============================================================================

PROTECTION 1 -- ORIENTATION-FRAME VERIFICATION GATE
  action[3:6] is a delta-orientation command consumed by the env's OSC_POSE
  controller (confirmed convention, see OUTPUT_MAX_ROT_RAD below). The
  observation's own orientation field (whatever key holds it, e.g.
  'robot0_eef_quat') is NOT assumed to be expressed in the same axis-angle
  frame/convention as that delta, until a LIVE single-step probe
  (probe_orientation_action_response(), a separate explicit --probe-
  orientation CLI mode, not run by this delivery) has been executed and its
  result reviewed by the operator. Every orientation-targeting code path in
  this file is gated behind cfg.orientation_control_verified (default
  False): if False, run_episode_edge_pinch() and compute_action_edge_pinch()
  refuse to compute target_ee_ori - current_ee_ori at all and return/raise
  a clearly labeled failure instead of silently trusting an unverified
  frame match. This is enforced in code (see _require_orientation_verified
  below), not just documented.

PROTECTION 2 -- REACHABILITY / FAILURE-ABORT GATE
  Before each critical phase (APPROACH, DESCEND) computes and steps toward
  a target, the target is checked against a configurable workspace bounding
  box (check_target_reachable()). An out-of-bounds target aborts the
  episode IMMEDIATELY -- zero wasted simulation steps -- rather than
  stepping toward something unreachable. During APPROACH/DESCEND, a stall
  detector (_StallTracker) aborts the episode early if the position/
  orientation error has not improved meaningfully over a rolling window,
  rather than waiting out the full per-phase step budget. A hard per-phase
  step-count timeout remains as a last-resort backstop. In ALL abort cases
  (unreachable target, stall, hard timeout) for a critical phase, the
  episode is terminated immediately: CLOSE and LIFT are NEVER entered for
  that candidate, and the specific reason is recorded in
  EdgePinchEpisodeResult.abort_reason / phase_abort_reason. This differs
  from task1_privileged_pickplace_controller_v1.py's original run_episode(),
  which recorded phase_timed_out but still advanced phase_idx and continued
  into CLOSE/LIFT regardless -- that loophole is deliberately closed here.

Design reference: claude/task1_privileged_pickplace_controller_design_v1.json
(revision 11 documents the audit; the design proposal accepted by the user,
plus these two protections, is what this file implements).

============================================================================
POST-PROBE RECALIBRATION (2026-09-16): QUATERNION-BASED ORIENTATION ERROR
============================================================================
Three single-axis live probes (probe_axis_index=0/1/2) confirmed
action[3:6]'s three channels correctly map to world X/Y/Z rotation axes
respectively (AXIS_CORRESPONDENCE_MATCH=True for all three, via
task1_orientation_probe_extract.py's quaternion-relative-rotation
analysis). Per the user's explicit instruction, --orientation-verified may
now be used for --calibrate. Before running the 40-candidate sweep, this
revision applies the requested "最后一次参数核算" (final parameter
recalibration):

  1. compute_action_edge_pinch()'s orientation-error computation has been
     changed from raw axis-angle VECTOR SUBTRACTION (target_ee_ori -
     current_ori_axisangle) to the QUATERNION RELATIVE-ROTATION method
     (q_rel = target_quat * conj(current_quat) -> axis,angle) -- the SAME
     representation-independent method the probes themselves required,
     because the raw axis-angle subtraction is unreliable in exactly the
     region this task's reset orientation sits in (angle_rad ~= pi). This
     was NOT explicitly requested by name, but is necessary for the
     requested numeric recalibration (below) to be meaningful: tuning
     max_ori_step_rad/orientation_tolerance_rad against a degenerate error
     metric would not fix the underlying issue.
  2. max_ori_step_rad: 0.08 -> 0.4 rad/step, grounded in the measured
     magnitude ratio (observed physical angle / nominal commanded angle)
     of 0.16-0.21 across the three probes: commanding up to ~0.4 rad/step
     (near OUTPUT_MAX_ROT_RAD=0.5's practical ceiling) is expected to yield
     a physically-realized ~0.07-0.08 rad/step of actual rotation once the
     impedance controller's convergence lag is accounted for.
  3. orientation_tolerance_rad: 0.15 -> 0.1 rad (~5.7deg), tightened now
     that the error metric itself is representation-correct.
  4. New EdgePinchConfig.max_phase_steps_orientation_convergence=200,
     applied to APPROACH/DESCEND specifically via max_steps_for_phase()
     (MOVE keeps max_phase_steps_large; CLOSE/LIFT/DESCEND2/RELEASE/RETREAT
     keep max_phase_steps_small) -- grounded in the computed world-frame
     geodesic rotation gap from the probed reset orientation to
     target_ee_ori: ~1.57 rad (~90deg, about an axis close to world -Z),
     which at an estimated ~0.08 rad/step realized rate would take ~20
     steps for orientation alone in the best case, but the budget is set
     far more generously (200) to tolerate slow initial convergence,
     simultaneous position tracking, and imperfect axis alignment.
  5. Hard-timeout and stall abort-reason strings in run_episode_edge_pinch()
     now explicitly report position_error_m and orientation_error_rad (with
     their pass/fail status against position_tolerance_m/
     orientation_tolerance_rad) at the moment of abort, instead of a bare
     phase name -- satisfying "明确姿态未收敛时的超时原因".
  6. probe_combined_axes_action_response() added: a low-cost, single-env,
     single-reset, small-simultaneous-delta-on-all-3-channels probe, as a
     supplementary coupling sanity check. It explicitly does NOT
     re-verify or re-litigate the three already-established single-axis
     correspondence conclusions.

None of the above runs any simulation by itself -- this revision is CODE
ONLY until re-verified via py_compile + the offline mock-test suite. The
actual 40-candidate --calibrate --orientation-verified sweep remains a
separate, explicitly-confirmed future step. No formal data collection, no
training, and no Fold02 unlock discussion are in scope here.

============================================================================
POST-CALIBRATION DIAGNOSIS FIX (2026-09-16, round 2): METRIC TIMING + STALL
============================================================================
A 16-candidate --calibrate --orientation-verified run (azimuth in
{0,45,90,135,180,225,270,315} x standoff in {0.03,0.05}) surfaced two
implementation bugs, diagnosed from the actual edge_pinch_calibration_
summary.json output (not from theory):

  1. finger_distance_at_lift_end_m and grasp_lift_delta_m were measured
     AFTER run_episode_edge_pinch()'s terminal_hold_steps loop, which
     unconditionally commands the gripper OPEN
     (action[6]=cfg.gripper_open_value) -- a leftover from the original
     controller's full RELEASE-at-plate pipeline, wrong for this truncated
     APPROACH->DESCEND->CLOSE->LIFT calibration subset. Evidence: all 8
     non-aborted candidates reported finger_distance_at_lift_end_m
     ~0.0936m (bitwise near-identical) regardless of their very different
     finger_distance_at_close_end_m (0.017-0.039m), and grasp_lift_delta_m
     was 0.0 or negative in every candidate -- both symptoms of measuring
     AFTER the gripper was forced back open. FIX: these metrics (plus a
     new bilateral_contact_at_lift_end field) are now captured the instant
     the LIFT phase itself completes, immediately after `phase_idx += 1`
     for phase=="LIFT", strictly BEFORE the terminal_hold loop runs.
  2. Several DESCEND candidates were aborted by the orientation stall
     detector even though orientation_error_rad was ALREADY well under
     orientation_tolerance_rad (e.g. 0.0232, 0.0220 vs tol=0.1) -- the
     stall trigger was small noise/oscillation in an already-converged
     quantity, while position was still improving steadily and only a few
     mm from its own tolerance. FIX: orientation-stall now only fires when
     orientation is NOT yet converged (>= orientation_tolerance_rad) AND
     genuinely worsening (window improvement < 0.0), not merely
     "improving slower than stall_min_orientation_improvement_rad".
     Position-stall logic is unchanged and still aborts on its own.

Per explicit user instruction, only azimuth in {45, 90, 135, 315} (both
standoffs, 8 candidates total) are to be re-run after these fixes -- NOT
the full 16/40-candidate sweep. Still no formal data collection, training,
or Fold02 unlock in scope.

CRITICAL LABELING CONSTRAINT (unchanged from the original controller):
  This controller reads object/gripper pose via PRIVILEGED SIMULATOR ACCESS
  ONLY to compute its own actions while GENERATING demonstration
  trajectories. This privileged information must NEVER be fed to SmolVLA as
  a model input at training or evaluation time. Every output artifact is
  stamped controller_source="privileged_simulator_pose" and
  generation_only=true.

This script never touches any existing dataset, checkpoint, training run,
or evaluation result, and never modifies training/eval protocols. Fold02
gate status is entirely unaffected by anything in this file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_BDDL_PATH = (
    "/root/smolvla-training-prep/data/libero_36/bddl/"
    "task_001_layout_1_akita_black_bowl_put_on_top_middle.bddl"
)
DEFAULT_TARGET_OBJECT = "akita_black_bowl_1"
DEFAULT_TARGET_CONTAINER = "plate_1"
DEFAULT_CONTROLLER_SCRIPT = (
    "/root/smolvla-training-prep/scripts/task1_privileged_pickplace_controller_v1.py"
)

# Same OSC_POSE convention as the original controller (confirmed in
# task1_strict_closedloop_diagnosis_20260915.json, revisions 17-18):
# action[:3] = position delta / OUTPUT_MAX_POS_M; action[3:6] = orientation
# delta / OUTPUT_MAX_ROT_RAD; action[6] = gripper.
OUTPUT_MAX_POS_M = 0.05
OUTPUT_MAX_ROT_RAD = 0.5
ACTION_DIM = 7

GRIPPER_OPEN_VALUE = -1.0
DEFAULT_GRIPPER_CLOSE_VALUE = 1.0

# Reuses the ORIGINAL controller's 8-phase names (per the user's explicit
# instruction to keep the 8-phase state machine). Orientation control is
# integrated INTO the existing APPROACH/DESCEND phases (converging every
# step alongside position, not as a separate phase) rather than adding a
# 9th phase -- this keeps the phase count and names identical to
# task1_privileged_pickplace_controller_v1.py.
PHASE_NAMES_EDGE = [
    "APPROACH",
    "DESCEND",
    "CLOSE",
    "LIFT",
    "MOVE",
    "DESCEND2",
    "RELEASE",
    "RETREAT",
]

# Phases before CLOSE where an unreachable/stalled target must abort the
# WHOLE episode rather than let execution fall through into CLOSE/LIFT
# (Protection 2). MOVE/DESCEND2/RELEASE/RETREAT are not exercised by the
# calibration subset (see run_calibration_subset_edge_pinch, which truncates
# to APPROACH->DESCEND->CLOSE->LIFT exactly like the original controller's
# _run_calibration_subset), but are included here for completeness/future
# full-episode use and use the same abort semantics.
CRITICAL_PHASES = ("APPROACH", "DESCEND", "MOVE", "DESCEND2")

DEFAULT_WORKSPACE_BOUNDS_M = {
    # Generously sized around the officially-audited eef_pos range
    # (x: 0.09-0.13, y: -0.07..-0.02, z: 0.92-1.07, per
    # task1_official_demo_audit_v1.json) plus margin for the edge-pinch
    # controller's wider approach/lift excursions. Configurable via CLI.
    # USED FOR: APPROACH/DESCEND only (2026-09-17 phase-separation, see
    # DEFAULT_WORKSPACE_BOUNDS_TRANSPORT_M below) -- this is the bowl-area
    # box and is intentionally left UNCHANGED by the transport-bounds fix.
    "x": (-0.10, 0.40),
    "y": (-0.40, 0.40),
    "z": (0.70, 1.30),
}

# 2026-09-17: MOVE/DESCEND2-specific workspace bounds, added after
# task1_workspace_bounds_audit_v1.py's GPU run confirmed plate_1's actual x
# range across 5 seeds is [-0.1565, -0.0868] -- below DEFAULT_WORKSPACE_
# BOUNDS_M's x lower bound of -0.10, causing 8/10 --smoke-test runs to abort
# during MOVE (see claude/task1_edge_pinch_workspace_bounds_audit_and_
# seed950004_diagnosis_20260916.json). The x lower bound here is set to
# -0.25 rather than just covering the audited -0.1565, because the
# grasp-offset compensation added in this same revision (see
# grasp_object_offset_m / _compensated_place_target()) shifts the MOVE/
# DESCEND2 EEF TARGET further negative than plate_x itself (diagnosed
# offset_x ~= +0.033m from claude/task1_edge_pinch_seed950004_phase_
# breakdown_and_offset_diagnosis_20260916.json), so the bounds must cover
# "audited plate range + compensation shift + margin", not just the plate
# range alone. y/z are DELIBERATELY IDENTICAL to DEFAULT_WORKSPACE_BOUNDS_M
# -- the audited plate y=[-0.0273,0.0325] and z=0.97 are comfortably inside
# the existing bounds, so there is no evidence to widen those axes, and
# doing so anyway would be an unjustified, ungrounded change.
DEFAULT_WORKSPACE_BOUNDS_TRANSPORT_M = {
    "x": (-0.25, 0.40),
    "y": (-0.40, 0.40),
    "z": (0.70, 1.30),
}


# --------------------------------------------------------------------------
# Controller-module reuse (avoids duplicating already-confirmed helpers)
# --------------------------------------------------------------------------


def _load_controller_module(controller_script_path: str):
    """Imports task1_privileged_pickplace_controller_v1.py as a module so
    this file can reuse its already-confirmed helpers (object_xyz, eef_xyz,
    clip_position_delta, _eef_pos_with_fallback, _gripper_part_contact_
    with_object, _gripper_contact_with_object, gripper_geometry_diagnostics,
    object_horizontal_geometry_diagnostics) instead of re-implementing them.
    Same importlib pattern already used and confirmed working in
    task1_official_demo_audit_v1.py (spec_from_file_location +
    sys.modules["ctrl"]=m before exec_module, required for dataclass
    introspection).
    """
    spec = importlib.util.spec_from_file_location("ctrl", controller_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load controller module from {controller_script_path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ctrl"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# Orientation representation handling (Protection 1)
# --------------------------------------------------------------------------


def quat_to_axisangle(quat: np.ndarray) -> dict:
    """Converts a quaternion to an axis-angle (rotation) vector.

    UNVERIFIED CONVENTION WARNING: this does NOT know, by itself, whether
    `quat` is scalar-first [w,x,y,z] (MuJoCo's own internal convention) or
    scalar-last [x,y,z,w] (robosuite's documented transform_utils
    convention for observation-dict quaternions). Per Protection 1, this
    function is a fallback ONLY -- the preferred path (see
    axisangle_from_quat_preferring_robosuite below) imports and uses
    robosuite's own robosuite.utils.transform_utils.quat2axisangle when
    available, since that is the authoritative, already-tested converter
    for whatever convention robosuite's own observation dict actually uses.
    This manual fallback assumes scalar-last [x,y,z,w] (robosuite's
    documented convention) and is only meant for offline unit testing when
    robosuite is not installed locally -- it must NOT be trusted as the
    final word on GPU without cross-checking against the robosuite
    converter's output in probe_orientation_action_response().
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    x, y, z, w = q  # ASSUMED scalar-last convention, see docstring warning
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arctan2(np.linalg.norm([x, y, z]), w)
    axis_norm = np.linalg.norm([x, y, z])
    if axis_norm < 1e-9:
        axis = np.zeros(3)
    else:
        axis = np.array([x, y, z]) / axis_norm
    return {
        "axisangle_vec": (axis * angle).tolist(),
        "angle_rad": float(angle),
        "axis": axis.tolist(),
        "convention_assumed": "scalar-last [x,y,z,w] (UNVERIFIED for this env; "
        "fallback only, see docstring)",
    }


def axisangle_from_quat_preferring_robosuite(quat: np.ndarray) -> dict:
    """Prefers robosuite's own quat2axisangle (authoritative for whatever
    convention robosuite's observation dict actually uses) over the manual
    fallback above. Falls back only if robosuite is not importable, and
    labels the result accordingly so it is never silently mistaken for a
    verified conversion.
    """
    try:
        from robosuite.utils.transform_utils import quat2axisangle  # noqa: WPS433

        vec = np.asarray(quat2axisangle(np.asarray(quat, dtype=np.float64)), dtype=np.float64)
        return {
            "axisangle_vec": vec.tolist(),
            "angle_rad": float(np.linalg.norm(vec)),
            "axis": (vec / np.linalg.norm(vec)).tolist() if np.linalg.norm(vec) > 1e-9 else [0.0, 0.0, 0.0],
            "convention_assumed": "robosuite.utils.transform_utils.quat2axisangle (authoritative)",
        }
    except Exception as exc:  # noqa: BLE001 -- fall back to the manual, unverified path
        result = quat_to_axisangle(quat)
        result["robosuite_import_error"] = f"{type(exc).__name__}: {exc}"
        return result


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q1*q2, scalar-last convention (x,y,z,w), matching
    robosuite's documented obs quaternion convention (confirmed via
    axisangle_from_quat_preferring_robosuite's use of
    robosuite.utils.transform_utils.quat2axisangle on the GPU probes).
    Identical formula to task1_orientation_probe_extract.py's quat_mul
    (kept in sync deliberately -- this is now the project's single
    established method for world-frame relative-rotation analysis).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


def quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def axisangle_vec_to_quat(vec: np.ndarray) -> np.ndarray:
    """Inverse of the axisangle_vec = axis*angle representation used
    throughout this file (quat_to_axisangle / axisangle_from_quat_
    preferring_robosuite's output, and EdgePinchConfig.target_ee_ori).
    Returns a scalar-last quaternion (x,y,z,w). Verified by hand
    (2026-09-16) against target_ee_ori=(2.2587,-2.2587,-0.0700): norm=
    3.19505..., a valid self-consistent axis-angle vector (it was itself
    derived via quat2axisangle from the audited official-demo data, so
    round-tripping through this inverse is expected to be exact).
    """
    vec = np.asarray(vec, dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    half = angle / 2.0
    xyz = axis * np.sin(half)
    return np.array([xyz[0], xyz[1], xyz[2], np.cos(half)])


def quat_relative_rotation_axisangle(q_from: np.ndarray, q_to: np.ndarray) -> tuple[np.ndarray, float]:
    """Computes the WORLD-FRAME rotation needed to go FROM q_from TO q_to,
    as (axis, angle_rad), via q_rel = q_to * conj(q_from) -- the same
    representation-independent method established and user-confirmed
    (2026-09-16) for all orientation-frame analysis in this project
    (task1_orientation_probe_extract.py), used here instead of raw
    axis-angle VECTOR SUBTRACTION (target - current), which is unreliable
    near angle_rad~=pi -- exactly the region this task's reset orientation
    sits in (per the probe reports' start_axisangle.angle_rad ~= 3.14).

    Resolves the quaternion double-cover sign ambiguity by choosing the
    representative with w>=0, i.e. the MINIMAL-angle (shorter-path)
    rotation (angle_rad in [0, pi]) -- verified by hand (2026-09-16) for
    the start-pose -> target_ee_ori gap: the w>=0 branch gives ~1.57rad
    (~90deg, axis close to world -Z), the alternate (w<0) branch gives the
    non-minimal ~4.71rad path, which is correctly discarded.
    """
    q_rel = quat_mul(np.asarray(q_to, dtype=np.float64), quat_conj(np.asarray(q_from, dtype=np.float64)))
    n = np.linalg.norm(q_rel)
    if n > 1e-12:
        q_rel = q_rel / n
    x, y, z, w = q_rel
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arctan2(np.linalg.norm([x, y, z]), w)
    axis_norm = np.linalg.norm([x, y, z])
    axis = np.array([x, y, z]) / axis_norm if axis_norm > 1e-9 else np.zeros(3)
    return axis, float(angle)


def probe_orientation_action_response(
    make_env_fn,
    ctrl_module,
    probe_axis_index: int = 1,
    probe_delta: float = 0.3,
    probe_steps: int = 5,
) -> dict:
    """LIVE single-step probe (Protection 1's verification mechanism). Runs
    env.reset(), records the starting orientation observation, then applies
    `probe_steps` steps of a KNOWN, constant small orientation-delta action
    (probe_delta on action[3+probe_axis_index], all other action dims 0
    except the gripper held at its open value) and records the resulting
    orientation observation after each step.

    Reports the observed axis-angle delta (via
    axisangle_from_quat_preferring_robosuite, both frame candidates) versus
    the commanded cumulative delta, so the operator can visually confirm
    whether the observed orientation change is consistent (same axis,
    plausible magnitude given OUTPUT_MAX_ROT_RAD scaling) with the commanded
    action -- this is the confirmation step required before
    cfg.orientation_control_verified may be set True for --calibrate.

    NOT invoked by default; only runs when the operator passes
    --probe-orientation on the CLI. This function itself is implemented and
    unit-testable (with a FakeEnv), but per the user's explicit instruction
    this delivery does not execute it against the real simulator.
    """
    report: dict = {
        "method": "live probe: env.reset() then probe_steps steps of a constant known "
        "orientation-delta action, comparing observed vs commanded rotation",
        "probe_axis_index": probe_axis_index,
        "probe_delta_per_step": probe_delta,
        "probe_steps": probe_steps,
    }
    env = None
    try:
        env = make_env_fn()
        obs = env.reset()
        quat_key = None
        for candidate_key in ("robot0_eef_quat", "eef_quat", "robot0_eef_quat_site"):
            if candidate_key in obs:
                quat_key = candidate_key
                break
        report["orientation_obs_key_found"] = quat_key
        if quat_key is None:
            report["error"] = (
                f"No known quaternion key found in obs; available keys: {sorted(obs.keys())}"
            )
            return report

        start_quat = np.asarray(obs[quat_key], dtype=np.float64)
        start_axisangle = axisangle_from_quat_preferring_robosuite(start_quat)
        report["start_quat"] = start_quat.tolist()
        report["start_axisangle"] = start_axisangle

        cumulative_commanded = 0.0
        step_reports = []
        for i in range(probe_steps):
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[3 + probe_axis_index] = float(probe_delta)
            action[6] = GRIPPER_OPEN_VALUE
            obs, _, done, _ = env.step(action)
            cumulative_commanded += probe_delta * OUTPUT_MAX_ROT_RAD
            cur_quat = np.asarray(obs[quat_key], dtype=np.float64)
            cur_axisangle = axisangle_from_quat_preferring_robosuite(cur_quat)
            step_reports.append(
                {
                    "step": i + 1,
                    "cumulative_commanded_rad_if_frame_matches": cumulative_commanded,
                    "observed_quat": cur_quat.tolist(),
                    "observed_axisangle": cur_axisangle,
                    "done": bool(done),
                }
            )
            if done:
                break
        report["step_reports"] = step_reports
        report["interpretation_note"] = (
            "Compare observed_axisangle.angle_rad growth per step against "
            "cumulative_commanded_rad_if_frame_matches: if they track each other "
            "(similar magnitude, and the dominant axis of observed_axisangle.axis "
            "stays consistent with probe_axis_index), the delta-action frame and "
            "the observation frame are consistent and "
            "cfg.orientation_control_verified=True is justified. If they diverge "
            "substantially (wrong axis dominant, magnitude ratio far from 1), do "
            "NOT set orientation_control_verified=True -- report the discrepancy "
            "instead so the frame mapping can be corrected first."
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, never fatal
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return report


def probe_combined_axes_action_response(
    make_env_fn,
    ctrl_module,
    probe_delta: float = 0.15,
    probe_steps: int = 5,
) -> dict:
    """Low-cost SUPPLEMENTARY probe (user-requested 2026-09-16): applies a
    small, SIMULTANEOUS, non-zero action on all three orientation channels
    at once (action[3]=action[4]=action[5]=probe_delta), and checks whether
    the resulting world-frame rotation direction (via the same quaternion
    relative-rotation method as the single-axis probes) is broadly
    consistent with the vector sum of the three already-probed single-axis
    world directions (X~=[1,0,0], Y~=[0,1,0], Z~=[0,0,1], per the
    2026-09-16 axis=0/1/2 probes' AXIS_CORRESPONDENCE_MATCH=True results).

    This is explicitly NOT a re-verification of those three single-axis
    conclusions -- it is a cheap cross-coupling sanity check only (does
    commanding all three channels together produce roughly the expected
    combined direction, or is there unexpected coupling between channels).
    A low cosine-similarity here would warrant follow-up, but per the
    user's instruction this probe does not by itself re-open or falsify
    the single-axis findings.
    """
    report: dict = {
        "method": "live probe: env.reset() then probe_steps steps of a constant small "
        "SIMULTANEOUS delta on all 3 orientation channels, quaternion relative-rotation "
        "analysis (same method as the single-axis probes) -- supplementary coupling "
        "check, does not re-verify the single-axis correspondence conclusions",
        "probe_delta_per_axis_per_step": probe_delta,
        "probe_steps": probe_steps,
    }
    env = None
    try:
        env = make_env_fn()
        obs = env.reset()
        quat_key = None
        for candidate_key in ("robot0_eef_quat", "eef_quat", "robot0_eef_quat_site"):
            if candidate_key in obs:
                quat_key = candidate_key
                break
        report["orientation_obs_key_found"] = quat_key
        if quat_key is None:
            report["error"] = (
                f"No known quaternion key found in obs; available keys: {sorted(obs.keys())}"
            )
            return report

        start_quat = np.asarray(obs[quat_key], dtype=np.float64)
        report["start_quat"] = start_quat.tolist()

        step_reports = []
        for i in range(probe_steps):
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[3] = float(probe_delta)
            action[4] = float(probe_delta)
            action[5] = float(probe_delta)
            action[6] = GRIPPER_OPEN_VALUE
            obs, _, done, _ = env.step(action)
            cur_quat = np.asarray(obs[quat_key], dtype=np.float64)
            axis, angle = quat_relative_rotation_axisangle(start_quat, cur_quat)
            step_reports.append(
                {
                    "step": i + 1,
                    "observed_quat": cur_quat.tolist(),
                    "world_frame_relative_axis": axis.tolist(),
                    "world_frame_relative_angle_rad": angle,
                    "done": bool(done),
                }
            )
            if done:
                break
        report["step_reports"] = step_reports

        # Expected direction under simple linear superposition of the three
        # already-probed single-axis world directions (2026-09-16 probes:
        # axis=0 -> world +X, axis=1 -> world +Y, axis=2 -> world +Z, all
        # AXIS_CORRESPONDENCE_MATCH=True), normalized.
        expected_dir = np.array([1.0, 1.0, 1.0])
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        report["expected_combined_axis_direction_if_linear_superposition"] = expected_dir.tolist()
        if step_reports:
            last_axis = np.array(step_reports[-1]["world_frame_relative_axis"])
            cos_sim = (
                float(np.dot(last_axis, expected_dir)) if np.linalg.norm(last_axis) > 1e-9 else None
            )
            report["observed_vs_expected_cosine_similarity"] = cos_sim
            report["coupling_consistent"] = bool(cos_sim is not None and cos_sim > 0.7)
        report["interpretation_note"] = (
            "cosine_similarity close to 1.0 indicates the three channels combine "
            "roughly as expected under simple superposition (no strong unexpected "
            "cross-coupling); this is a low-cost supplementary sanity check only and "
            "does NOT by itself re-verify or override the three single-axis "
            "correspondence conclusions already established via probe_axis_index=0/1/2."
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, never fatal
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return report


def _require_orientation_verified(cfg: "EdgePinchConfig") -> Optional[str]:
    """Protection 1's enforcement point. Returns None if orientation control
    may proceed, else a human-readable reason string explaining why it must
    not. Called at the START of run_episode_edge_pinch() and again inside
    compute_action_edge_pinch() (defense in depth: a caller who somehow
    bypasses the episode-level check still cannot get a single action
    computed with unverified orientation targeting).
    """
    if not cfg.orientation_control_verified:
        return (
            "ORIENTATION_CONTROL_NOT_VERIFIED: cfg.orientation_control_verified is "
            "False. Refusing to compute target_ee_ori - current_ee_ori without "
            "confirmed evidence that action[3:6]'s delta frame matches the "
            "observation's orientation frame. Run probe_orientation_action_response() "
            "(--probe-orientation), review its report, then re-run with "
            "--orientation-verified only if the observed and commanded rotations "
            "are consistent."
        )
    return None


# --------------------------------------------------------------------------
# Config / records
# --------------------------------------------------------------------------


@dataclass
class EdgePinchConfig:
    target_object: str = DEFAULT_TARGET_OBJECT
    target_container: str = DEFAULT_TARGET_CONTAINER

    # Edge-point selection (design section 3)
    edge_azimuth_deg: float = 0.0  # calibrated candidate, see build_arg_parser
    standoff_m: float = 0.05  # height above the edge point during APPROACH
    descend_edge_height_offset_m: float = 0.0  # relative to the edge point's own world z

    # Orientation target (design section 2). Axis-angle vector, SAME
    # representation as action[3:6] deltas once Protection 1 is satisfied.
    # Defaults anchor the audited official-demo ee_ori progression
    # (task1_official_demo_audit_v1.json): axis0 ~2.23, axis1 ~-2.26 (the
    # dominant rotation), axis2 ~-0.04..-0.11. These are CANDIDATE values
    # for calibration, not asserted-correct targets -- see design doc
    # section 5's calibration range.
    target_ee_ori: tuple[float, float, float] = (2.2587, -2.2587, -0.0700)
    # Recalibrated 2026-09-16 against the quaternion-based orientation-error
    # metric (see module docstring "POST-PROBE RECALIBRATION" section),
    # grounded in the three-probe measured magnitude ratio of 0.16-0.21:
    max_ori_step_rad: float = 0.4
    orientation_tolerance_rad: float = 0.1

    # Position control (reused convention from the original controller)
    position_tolerance_m: float = 0.01
    lift_height_m: float = 0.15
    place_height_offset_m: float = 0.05
    retreat_height_m: float = 0.15

    grasp_hold_steps: int = 20
    release_hold_steps: int = 20
    terminal_hold_steps: int = 20

    max_phase_steps_large: int = 150
    max_phase_steps_small: int = 80
    # New 2026-09-16: dedicated, larger step budget for APPROACH/DESCEND,
    # whose target pose now requires substantial orientation convergence
    # (~1.57rad geodesic gap from the reset pose to target_ee_ori, computed
    # by hand via quat_relative_rotation_axisangle) in addition to position
    # convergence. Grounded in an estimated ~0.08rad/step realized rotation
    # rate (extrapolated from the probes' measured per-step increments),
    # sized generously to tolerate slower real convergence, simultaneous
    # position tracking, and imperfect axis alignment. See
    # max_steps_for_phase() below for routing.
    max_phase_steps_orientation_convergence: int = 200
    max_episode_steps: int = 600

    gripper_open_value: float = GRIPPER_OPEN_VALUE
    gripper_close_value: float = DEFAULT_GRIPPER_CLOSE_VALUE

    # Target gripper spacing band at LIFT (design section 4). Pass/fail
    # judgment field, not a control input -- the gripper is commanded fully
    # closed (+1.0); this band is what we check the RESULT against.
    target_finger_distance_band_m: tuple[float, float] = (0.015, 0.028)

    # Protection 1
    orientation_control_verified: bool = False

    # Protection 2
    workspace_bounds_m: dict = field(default_factory=lambda: dict(DEFAULT_WORKSPACE_BOUNDS_M))
    # 2026-09-17: separate, MOVE/DESCEND2-only bounds (see
    # DEFAULT_WORKSPACE_BOUNDS_TRANSPORT_M for the data-grounded rationale).
    # APPROACH/DESCEND continue to use workspace_bounds_m above, unchanged.
    workspace_bounds_transport_m: dict = field(
        default_factory=lambda: dict(DEFAULT_WORKSPACE_BOUNDS_TRANSPORT_M)
    )
    stall_window_steps: int = 15
    stall_min_position_improvement_m: float = 0.002
    stall_min_orientation_improvement_rad: float = 0.02

    # 2026-09-17: grasp-offset estimation window (see grasp_object_offset_m
    # in EdgePinchEpisodeResult / _estimate_grasp_object_offset()). Averaging
    # over the last N LIFT-phase step records rather than a single step,
    # to damp any residual noise right as LIFT completes.
    grasp_offset_estimation_window_steps: int = 5

    def max_steps_for_phase(self, phase: str) -> int:
        # APPROACH/DESCEND get the dedicated, larger orientation-convergence
        # budget (2026-09-16 recalibration) since target_ee_ori's tilted
        # pose requires substantial orientation convergence in addition to
        # position convergence during these two phases specifically.
        if phase in ("APPROACH", "DESCEND"):
            return self.max_phase_steps_orientation_convergence
        if phase == "MOVE":
            return self.max_phase_steps_large
        return self.max_phase_steps_small


@dataclass
class EdgePinchStepRecord:
    step: int
    phase: str
    action: np.ndarray
    eef_pos: np.ndarray
    target_pos: np.ndarray
    position_error_m: float
    orientation_error_rad: Optional[float]
    target_object_pos: np.ndarray
    gripper_contact: Optional[bool] = None
    gripper_part_contact: dict = field(default_factory=dict)
    finger_distance_m: Optional[float] = None


@dataclass
class EdgePinchEpisodeResult:
    seed: int
    steps_run: int = 0
    phase_transition_steps: dict = field(default_factory=dict)
    phase_timed_out: dict = field(default_factory=dict)
    phase_abort_reason: dict = field(default_factory=dict)
    aborted: bool = False
    abort_reason: Optional[str] = None
    grasp_lift_delta_m: Optional[float] = None
    finger_distance_at_close_end_m: Optional[float] = None
    finger_distance_at_lift_end_m: Optional[float] = None
    bilateral_contact_at_close_end: Optional[bool] = None
    bilateral_contact_at_lift_end: Optional[bool] = None
    bilateral_contact_lift_fraction: Optional[float] = None
    edge_target_point_world_m: Optional[list] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    records: list = field(default_factory=list)

    # 2026-09-17: grasp-offset compensation (see _estimate_grasp_object_offset()
    # / _compensated_place_target()). grasp_object_offset_m is the estimated
    # (bowl - eef) vector at LIFT end, used to compensate MOVE/DESCEND2/
    # RELEASE targets so the BOWL (not the EEF) is what gets centered over
    # the plate. Recorded here for transparency/debuggability of every run.
    grasp_object_offset_m: Optional[list] = None
    grasp_object_offset_std_m: Optional[list] = None
    grasp_object_offset_reliable: Optional[bool] = None
    grasp_object_offset_fallback_used: bool = False

    # 2026-09-17 diagnostic addition: when env.step() returns done=True mid-
    # phase, the code raises RuntimeError WITHOUT knowing whether that means
    # "task succeeded and the env terminated itself" or "a genuine physical
    # failure/instability" -- diagnosed after the offset-compensation smoke
    # test made 10/10 runs hit this exact path during RELEASE, all with
    # env_check_success left at None because check_success() was never
    # reached. These fields capture what's knowable at the moment done=True
    # fires, BEFORE raising, so that question can actually be answered
    # instead of guessed at. Does NOT change control flow, target
    # computation, bounds, or when the RuntimeError is raised -- purely
    # additive diagnostics on the same exception path.
    done_early_termination_phase: Optional[str] = None
    done_early_termination_step: Optional[int] = None
    done_early_termination_check_success: Optional[bool] = None
    done_early_termination_check_success_error: Optional[str] = None
    done_early_termination_info: Optional[dict] = None

    # 2026-09-17 (design: claude/task1_edge_pinch_release_success_termination_
    # design_v1_20260917.json): unified record of how the episode ended.
    # Populated at exactly two additional points beyond the pre-existing
    # 'normal_completion' path: 'success_early_termination' when RELEASE's
    # own done=True fires with check_success()==True (see
    # _handle_release_done_termination), and 'environment_failure_done_true'
    # when RELEASE's done=True fires with check_success() False/unknown.
    # CLOSE and all other phases' done=True handling (via
    # _diagnose_and_raise_on_done) does NOT set this field -- out of scope
    # for this design, see the design doc's scope_boundary section.
    termination_reason: Optional[str] = None


# --------------------------------------------------------------------------
# Edge-point selection (design section 3): local-frame + azimuth
# --------------------------------------------------------------------------


def compute_rim_local_points(make_env_fn, ctrl_module, target_object_name: str) -> dict:
    """One-time (per fresh env instance), read-only computation of the rim
    ring's member geoms' BODY-LOCAL positions (model.geom_pos -- a static,
    body-frame-relative offset, NOT a world position), so the edge target
    point can be recomputed every step purely from the object's LIVE
    body_xpos + body_xmat (Protection-1-independent: this uses a 3x3
    rotation matrix directly from MuJoCo, sidestepping any quaternion
    convention ambiguity entirely) without re-deriving which geoms belong
    to the rim at every step.

    Reuses the same world-z clustering heuristic as the original
    controller's object_horizontal_geometry_diagnostics() (>=3 box geoms
    sharing a world z rounded to 1mm at env.reset() default pose; topmost
    such cluster reported) to select rim member geoms, then reads each
    selected geom's LOCAL (body-frame) position from model.geom_pos instead
    of (or in addition to) its world position.

    Defensive by construction like every diagnostic in the original
    controller file: any failure is captured in the returned dict rather
    than raised. No simulation steps are taken beyond a single env.reset().
    """
    result: dict = {
        "method": "rim-ring geoms selected via world-z clustering at reset (reused "
        "heuristic from object_horizontal_geometry_diagnostics), local (body-frame) "
        "positions read from model.geom_pos, no simulation steps beyond env.reset()"
    }
    env = None
    try:
        env = make_env_fn()
        env.reset()
        inner = getattr(env, "env", env)
        model = inner.sim.model
        data = inner.sim.data

        objects_dict = getattr(inner, "objects_dict", None)
        if objects_dict is None:
            result["error"] = "inner.objects_dict not found."
            return result
        obj = objects_dict.get(target_object_name)
        if obj is None:
            result["error"] = f"'{target_object_name}' not found in objects_dict."
            return result

        root_body = obj.root_body
        body_id = model.body_name2id(root_body)

        geom_bodyid = np.asarray(model.geom_bodyid)
        attached_geom_ids = [
            int(gid) for gid in range(len(geom_bodyid)) if geom_bodyid[gid] == body_id
        ]
        box_geoms = []
        for gid in attached_geom_ids:
            if int(model.geom_type[gid]) != 6:  # box only, same rationale as the
                continue  # original horizontal-geometry diagnostic (mesh over-estimates)
            gxyz = np.array(data.geom_xpos[gid], dtype=np.float64)
            glocal = np.array(model.geom_pos[gid], dtype=np.float64)
            box_geoms.append(
                {
                    "geom_id": gid,
                    "geom_name": model.geom_id2name(gid) or f"geom_id_{gid}",
                    "world_xyz": gxyz.tolist(),
                    "local_xyz": glocal.tolist(),
                }
            )
        if not box_geoms:
            result["error"] = "No geom_type==6 (box) geoms found attached to this body."
            return result

        clusters: dict = {}
        for g in box_geoms:
            zr = round(g["world_xyz"][2], 3)
            clusters.setdefault(zr, []).append(g)
        ring_clusters = [(zr, members) for zr, members in clusters.items() if len(members) >= 3]
        if not ring_clusters:
            result["error"] = "No world-z cluster with >=3 box geoms found; cannot isolate a rim."
            return result

        rim_z, rim_members = max(ring_clusters, key=lambda kv: kv[0])
        result["rim_cluster_world_z"] = rim_z
        result["num_geoms_in_rim_cluster"] = len(rim_members)
        result["root_body"] = root_body
        result["rim_local_points"] = [
            {
                "geom_name": g["geom_name"],
                "local_xyz": g["local_xyz"],
                "local_azimuth_deg": float(
                    np.degrees(np.arctan2(g["local_xyz"][1], g["local_xyz"][0]))
                ),
            }
            for g in rim_members
        ]
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return result


def select_edge_local_point(rim_local_points: dict, edge_azimuth_deg: float) -> Optional[np.ndarray]:
    """Picks the rim-member local point whose local_azimuth_deg is closest
    (circular distance) to edge_azimuth_deg. Returns None if
    rim_local_points has no usable points (e.g. compute_rim_local_points
    failed) -- callers must treat None as an immediate abort condition, not
    silently fall back to the object's root-body center (that would
    silently regress to the already-falsified whole-object-center strategy).
    """
    points = rim_local_points.get("rim_local_points")
    if not points:
        return None
    target = edge_azimuth_deg % 360.0

    def circ_dist(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    best = min(points, key=lambda p: circ_dist(p["local_azimuth_deg"], target))
    return np.array(best["local_xyz"], dtype=np.float64)


def edge_target_point_world(
    env, target_object_name: str, edge_local_point: np.ndarray, ctrl_module
) -> Optional[np.ndarray]:
    """Transforms the (fixed, body-local) edge point to WORLD coordinates
    using the object's CURRENT (live, per-step) body_xpos + body_xmat. This
    is what makes the edge target track the bowl's real-time pose (design
    requirement: '接近、闭合和抬升阶段都保留实时位姿反馈') without needing
    any quaternion conversion -- body_xmat is a direct 3x3 rotation matrix
    from MuJoCo, no convention ambiguity.

    Defensive: returns None on any failure (caller must treat as abort).
    """
    try:
        inner = getattr(env, "env", env)
        model = inner.sim.model
        data = inner.sim.data
        objects_dict = getattr(inner, "objects_dict", None)
        obj = objects_dict.get(target_object_name)
        root_body = obj.root_body
        body_id = model.body_name2id(root_body)
        body_xpos = np.array(data.body_xpos[body_id], dtype=np.float64)
        body_xmat = np.array(data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
        world_point = body_xpos + body_xmat @ edge_local_point
        return world_point
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Protection 2: reachability + stall detection
# --------------------------------------------------------------------------


def check_target_reachable(target_pos: np.ndarray, bounds: dict) -> Optional[str]:
    """Returns None if target_pos is inside the configured workspace
    bounding box, else a human-readable reason string. Called BEFORE any
    simulation step is taken toward that target (zero wasted steps on an
    unreachable candidate).
    """
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = bounds[axis]
        v = float(target_pos[i])
        if not (lo <= v <= hi):
            return (
                f"target_out_of_workspace_bounds: {axis}={v:.4f} not in "
                f"[{lo:.4f}, {hi:.4f}]"
            )
    return None


class _StallTracker:
    """Rolling-window stall detector for a single phase. Call update() each
    step with the current scalar error (position_error_m, or
    orientation_error_rad); is_stalled() returns True once `window` samples
    have been seen and the error has not improved by at least `min_improve`
    since the start of the window.
    """

    def __init__(self, window: int, min_improve: float):
        self.window = window
        self.min_improve = min_improve
        self._buf: list[float] = []

    def update(self, error: float) -> None:
        self._buf.append(float(error))
        if len(self._buf) > self.window:
            self._buf.pop(0)

    def is_stalled(self) -> bool:
        if len(self._buf) < self.window:
            return False
        improvement = self._buf[0] - self._buf[-1]
        return improvement < self.min_improve

    def improvement(self) -> Optional[float]:
        if len(self._buf) < self.window:
            return None
        return self._buf[0] - self._buf[-1]


# --------------------------------------------------------------------------
# Action computation (Protection 1 enforced here too, defense in depth)
# --------------------------------------------------------------------------


def orientation_error_rad_between(
    current_ori_axisangle: np.ndarray, target_ori_axisangle: np.ndarray
) -> float:
    """DEPRECATED / UNRELIABLE near angle_rad~=pi -- raw axis-angle VECTOR
    SUBTRACTION, the same degenerate representation whose unreliability
    motivated switching to the quaternion relative-rotation method (see
    quat_relative_rotation_axisangle and the module docstring's
    'POST-PROBE RECALIBRATION' section). Kept only for any external
    callers/tests that may still reference it; compute_action_edge_pinch
    no longer uses this function as of the 2026-09-16 recalibration.
    """
    return float(np.linalg.norm(np.asarray(target_ori_axisangle) - np.asarray(current_ori_axisangle)))


def compute_action_edge_pinch(
    cfg: EdgePinchConfig,
    target_pos: np.ndarray,
    eef_pos: np.ndarray,
    current_quat: Optional[np.ndarray],
    gripper_value: float,
    ctrl_module,
) -> tuple[np.ndarray, float, Optional[float]]:
    """Returns (7-dim action, position_error_m, orientation_error_rad).

    ORIENTATION ERROR (2026-09-16 recalibration): computed via the
    quaternion relative-rotation method (quat_relative_rotation_axisangle),
    NOT raw axis-angle vector subtraction -- the latter is unreliable near
    angle_rad~=pi, exactly the region this task's reset orientation sits
    in (see module docstring). `current_quat` is the RAW observed
    end-effector quaternion (scalar-last [x,y,z,w]), not a pre-converted
    axis-angle vector -- see _current_quat() below.

    orientation_error_rad is None if current_quat is None (e.g. the
    orientation observation key could not be found this step) -- in that
    case action[3:6] is held at 0.0 for this step and the caller should
    treat it as an orientation-tracking failure for stall-detection
    purposes, NOT as "orientation reached".

    PROTECTION 1 (re-checked here, defense in depth): if
    cfg.orientation_control_verified is False, this raises RuntimeError
    immediately -- it will never silently compute an orientation delta on
    an unverified frame, even if a caller bypassed the episode-level check.
    """
    veto_reason = _require_orientation_verified(cfg)
    if veto_reason is not None:
        raise RuntimeError(veto_reason)

    delta_pos = target_pos - eef_pos
    position_error_m = float(np.linalg.norm(delta_pos))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = ctrl_module.clip_position_delta(delta_pos, OUTPUT_MAX_POS_M)

    orientation_error_rad: Optional[float] = None
    if current_quat is not None:
        target_quat = axisangle_vec_to_quat(np.asarray(cfg.target_ee_ori, dtype=np.float64))
        axis, angle = quat_relative_rotation_axisangle(
            np.asarray(current_quat, dtype=np.float64), target_quat
        )
        orientation_error_rad = float(angle)
        step_angle = min(angle, cfg.max_ori_step_rad)
        delta_ori_vec = axis * step_angle
        action[3:6] = np.clip(delta_ori_vec / OUTPUT_MAX_ROT_RAD, -1.0, 1.0)
    else:
        action[3:6] = 0.0

    action[6] = float(gripper_value)
    if action.shape != (ACTION_DIM,):
        raise RuntimeError(f"Action shape mismatch: {action.shape}")
    if np.any(action[:6] < -1.0) or np.any(action[:6] > 1.0):
        raise RuntimeError(f"Action out of [-1,1] range: {action}")
    return action, position_error_m, orientation_error_rad


def _current_ori_axisangle(env, obs: dict) -> Optional[np.ndarray]:
    """Reads the live end-effector orientation and converts it to axis-angle
    via the robosuite-preferring converter. Returns None (never raises) if
    no known quaternion key is present. NOT used by compute_action_edge_pinch
    as of the 2026-09-16 recalibration (which uses the raw quaternion via
    _current_quat() instead, to avoid the axis-angle degeneracy near
    angle_rad~=pi) -- kept for any diagnostic/reporting use and for offline
    unit-test compatibility.
    """
    for key in ("robot0_eef_quat", "eef_quat", "robot0_eef_quat_site"):
        if key in obs:
            result = axisangle_from_quat_preferring_robosuite(np.asarray(obs[key]))
            return np.array(result["axisangle_vec"], dtype=np.float64)
    return None


def _current_quat(env, obs: dict) -> Optional[np.ndarray]:
    """Reads the live end-effector quaternion (scalar-last [x,y,z,w],
    robosuite's obs-dict convention) directly, WITHOUT converting to
    axis-angle -- required by compute_action_edge_pinch's quaternion
    relative-rotation orientation-error computation (2026-09-16
    recalibration; raw axis-angle vector subtraction is unreliable near
    angle_rad~=pi). Returns None (never raises) if no known quaternion key
    is present, mirroring _current_ori_axisangle.
    """
    for key in ("robot0_eef_quat", "eef_quat", "robot0_eef_quat_site"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64)
    return None


# --------------------------------------------------------------------------
# Episode rollout (state machine driver, Protections 1+2 enforced)
# --------------------------------------------------------------------------

# 2026-09-17: a run without a reliable offset estimate (LIFT never produced
# any records) falls back to a zero offset -- i.e. exactly the OLD,
# uncompensated behavior -- rather than an undefined/guessed value. This
# constant names that fallback explicitly so it is never confused with a
# genuinely-measured all-zero offset.
ZERO_GRASP_OFFSET_M = (0.0, 0.0, 0.0)

# Diagnostic-only reliability threshold for grasp_object_offset_std_m (see
# _estimate_grasp_object_offset). Crossing this does NOT abort the episode
# or change any target computation -- it only sets
# grasp_object_offset_reliable=False in the result for later review, per the
# design doc's explicit choice not to introduce a new, untested abort
# condition alongside the offset-compensation fix itself.
GRASP_OFFSET_STD_RELIABLE_THRESHOLD_M = 0.01


def _estimate_grasp_object_offset(
    records: list, window_steps: int
) -> tuple[np.ndarray, Optional[np.ndarray], bool, bool]:
    """Estimates the constant (bowl - eef) spatial offset introduced by the
    edge-pinch grasp, from the LAST `window_steps` LIFT-phase step records
    (`target_object_pos` is the bowl's own position, `eef_pos` is the
    end-effector position -- both already recorded per-step by
    run_episode_edge_pinch, no new instrumentation needed).

    Averaging over the tail of LIFT (rather than a single step) damps any
    residual settling noise right as LIFT completes; the bowl should already
    be static in the closed gripper with no table contact by then.

    Returns (offset_m, std_m_or_None, reliable, fallback_used). When there
    are no LIFT records at all (record_trace=False, or LIFT itself was
    somehow skipped), returns (ZERO_GRASP_OFFSET_M, None, False, True) --
    the zero-offset fallback is IDENTICAL to the pre-2026-09-17 uncompensated
    behavior, so an estimation failure degrades to the old behavior rather
    than to an undefined new one.
    """
    lift_records = [r for r in records if r.phase == "LIFT"]
    if not lift_records:
        return np.array(ZERO_GRASP_OFFSET_M, dtype=np.float64), None, False, True

    tail = lift_records[-min(window_steps, len(lift_records)):]
    per_step_offsets = np.array(
        [np.asarray(r.target_object_pos, dtype=np.float64) - np.asarray(r.eef_pos, dtype=np.float64) for r in tail]
    )
    offset_mean = per_step_offsets.mean(axis=0)
    offset_std = per_step_offsets.std(axis=0)
    reliable = bool(np.all(offset_std < GRASP_OFFSET_STD_RELIABLE_THRESHOLD_M))
    return offset_mean, offset_std, reliable, False


def _compensated_place_target(
    plate: np.ndarray, offset_m: np.ndarray, cfg: EdgePinchConfig
) -> np.ndarray:
    """Shared DESCEND2/RELEASE target formula (2026-09-17 offset-compensation
    fix): since bowl_pos ~= eef_pos + offset_m (a constant, per-episode
    estimate from _estimate_grasp_object_offset), driving the EEF to
    `plate + offset-compensated delta` is what actually centers the BOWL
    over the plate -- not driving the EEF directly to the plate's own
    coordinates (the pre-fix behavior, which left the bowl consistently
    off-target by the grasp offset; see claude/task1_edge_pinch_seed950004_
    phase_breakdown_and_offset_diagnosis_20260916.json). The z component is
    compensated too (unlike MOVE's transit height): the goal is for the BOWL
    to sit at plate_z + place_height_offset_m, so the EEF target must be
    that height MINUS offset_z.
    """
    return np.array(
        [
            plate[0] - offset_m[0],
            plate[1] - offset_m[1],
            plate[2] + cfg.place_height_offset_m - offset_m[2],
        ]
    )


def phase_target_pos_edge_pinch(
    phase: str,
    cfg: EdgePinchConfig,
    env,
    ctrl_module,
    obs: dict,
    eef_pos: np.ndarray,
    lift_reference_z: Optional[float],
    edge_local_point: Optional[np.ndarray],
    grasp_object_offset_m: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Returns None if the edge target point cannot be computed this step
    (caller must treat as an immediate abort -- Protection 2 requires this,
    NOT a silent fallback to the object's root-body center).

    `grasp_object_offset_m` (2026-09-17): the per-episode estimated
    (bowl - eef) offset from _estimate_grasp_object_offset(), used by MOVE/
    DESCEND2/RELEASE to compensate their targets so the BOWL (not the EEF)
    is what gets centered on the plate. Defaults to None (treated as zero
    offset, i.e. the pre-fix uncompensated behavior) purely so existing
    callers/tests that construct this function's arguments directly, without
    knowledge of the new parameter, keep working unchanged.
    """
    off = (
        np.asarray(grasp_object_offset_m, dtype=np.float64)
        if grasp_object_offset_m is not None
        else np.array(ZERO_GRASP_OFFSET_M, dtype=np.float64)
    )
    plate = ctrl_module.object_xyz(obs, cfg.target_container)

    if phase in ("APPROACH", "DESCEND", "CLOSE"):
        if edge_local_point is None:
            return None
        edge_world = edge_target_point_world(env, cfg.target_object, edge_local_point, ctrl_module)
        if edge_world is None:
            return None
        if phase == "APPROACH":
            return np.array([edge_world[0], edge_world[1], edge_world[2] + cfg.standoff_m])
        # DESCEND and CLOSE both target the edge point itself (+ configurable offset)
        return np.array(
            [edge_world[0], edge_world[1], edge_world[2] + cfg.descend_edge_height_offset_m]
        )
    if phase == "LIFT":
        assert lift_reference_z is not None
        return np.array([eef_pos[0], eef_pos[1], lift_reference_z + cfg.lift_height_m])
    if phase == "MOVE":
        # 2026-09-17: xy compensated so the BOWL (not the EEF) ends up over
        # the plate; z (transit height) is intentionally NOT compensated --
        # it is a safe transport height derived from lift_reference_z, not a
        # plate-relative target, so offset_z has no bearing on it.
        assert lift_reference_z is not None
        return np.array([plate[0] - off[0], plate[1] - off[1], lift_reference_z + cfg.lift_height_m])
    if phase in ("DESCEND2", "RELEASE"):
        # 2026-09-17: both phases share the identical compensated target
        # (unchanged from the pre-fix design's "RELEASE targets the same
        # point as DESCEND2" -- only the formula for that shared point
        # changed, per the design doc's explicit instruction to keep
        # DESCEND2/RELEASE using one consistent target).
        return _compensated_place_target(plate, off, cfg)
    if phase == "RETREAT":
        assert lift_reference_z is not None
        return np.array([eef_pos[0], eef_pos[1], lift_reference_z + cfg.retreat_height_m])
    raise ValueError(f"Unknown phase: {phase}")


def phase_gripper_value_edge_pinch(phase: str, cfg: EdgePinchConfig) -> float:
    if phase in ("APPROACH", "DESCEND", "RELEASE", "RETREAT"):
        return cfg.gripper_open_value
    if phase in ("CLOSE", "LIFT", "MOVE", "DESCEND2"):
        return cfg.gripper_close_value
    raise ValueError(f"Unknown phase: {phase}")


def _finger_distance_m(env, ctrl_module) -> Optional[float]:
    try:
        inner = getattr(env, "env", env)
        gripper = inner.robots[0].gripper
        model = inner.sim.model
        data = inner.sim.data
        important_geoms = dict(gripper.important_geoms)

        def _avg_world_pos(geom_names: list) -> Optional[np.ndarray]:
            pts = []
            for gname in geom_names:
                gid = model.geom_name2id(gname)
                pts.append(np.array(data.geom_xpos[gid], dtype=np.float64))
            if not pts:
                return None
            return np.mean(pts, axis=0)

        left_pt = _avg_world_pos(important_geoms.get("left_finger", []))
        right_pt = _avg_world_pos(important_geoms.get("right_finger", []))
        if left_pt is None or right_pt is None:
            return None
        return float(np.linalg.norm(left_pt - right_pt))
    except Exception:  # noqa: BLE001
        return None


def _diagnose_and_raise_on_done(env, phase: str, step_count: int, info, result) -> None:
    """2026-09-17 diagnostic addition: called at the exact moment env.step()
    returns done=True mid-phase, BEFORE the RuntimeError that has always
    been raised here (unchanged: this still always raises, control flow is
    identical to before). The only change is that result's
    done_early_termination_* fields are populated first, so the caught
    exception's result object tells us whether check_success() was already
    True at that instant (env terminating itself on task success -- a
    behavior some LIBERO/robosuite env wrappers have) versus some other
    condition, instead of leaving env_check_success as an unexplained None.

    env.check_success() is called defensively (a wrapped try/except) since
    an env that just set done=True mid-episode may itself be in a state
    where check_success() raises -- that possibility is itself diagnostic
    information (done_early_termination_check_success_error), not a reason
    to let this function raise something other than the RuntimeError it
    always raised.
    """
    result.done_early_termination_phase = phase
    result.done_early_termination_step = step_count
    try:
        result.done_early_termination_check_success = bool(env.check_success())
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, must not mask the real raise below
        result.done_early_termination_check_success = None
        result.done_early_termination_check_success_error = f"{type(exc).__name__}: {exc}"
    try:
        result.done_early_termination_info = dict(info) if info else {}
    except Exception:  # noqa: BLE001 -- info may not be dict-like in some env wrappers
        result.done_early_termination_info = {"unconvertible_info_repr": repr(info)}
    raise RuntimeError(
        f"Environment ended unexpectedly during {phase} "
        f"(check_success()={result.done_early_termination_check_success}, "
        f"info={result.done_early_termination_info})"
    )


def _handle_release_done_termination(env, step_count: int, info, result: "EdgePinchEpisodeResult") -> bool:
    """2026-09-17 (design: claude/task1_edge_pinch_release_success_termination_
    design_v1_20260917.json). RELEASE-phase-only counterpart to
    _diagnose_and_raise_on_done(): unlike that function, this one does NOT
    unconditionally raise. It is called at the exact moment env.step()
    returns done=True during the RELEASE phase specifically (never CLOSE or
    any other phase -- those remain on _diagnose_and_raise_on_done,
    unchanged).

    Populates the same done_early_termination_* diagnostic fields as
    _diagnose_and_raise_on_done (identical defensive check_success()/info
    handling), plus the outcome fields requested by the confirmed design:
    result.success, result.termination_reason, result.steps_run (so both
    the success and the failure path carry steps_run without depending on
    the caller's except-block fallback). result.grasp_object_offset_m is
    NOT touched here -- it is already populated at LIFT-end (see
    _estimate_grasp_object_offset call site) and remains set regardless of
    which path this function takes.

    Returns True when this is a genuine success-early-termination (task
    completed, environment ended itself): the caller must `return result`
    immediately, WITHOUT running terminal_hold_steps or any further
    env.step() calls, and WITHOUT raising -- per the confirmed design
    decision, stepping an already-done environment is undefined behavior
    and is deliberately avoided.

    Returns False when this is a real failure (check_success() is False,
    or check_success() itself raised and the outcome is therefore unknown
    -- per the confirmed design decision, an exception during check_success()
    is treated as "unknown", NOT silently upgraded to a success, and
    result.success is left as None rather than being forced to False): the
    caller must raise RuntimeError exactly as _diagnose_and_raise_on_done
    would have (same message shape), preserving existing error-reporting
    behavior/format for this failure path.
    """
    result.done_early_termination_phase = "RELEASE"
    result.done_early_termination_step = step_count
    check_success_value: Optional[bool] = None
    try:
        check_success_value = bool(env.check_success())
        result.done_early_termination_check_success = check_success_value
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, must not mask outcome handling below
        result.done_early_termination_check_success = None
        result.done_early_termination_check_success_error = f"{type(exc).__name__}: {exc}"
    try:
        result.done_early_termination_info = dict(info) if info else {}
    except Exception:  # noqa: BLE001 -- info may not be dict-like in some env wrappers
        result.done_early_termination_info = {"unconvertible_info_repr": repr(info)}

    result.steps_run = step_count

    if check_success_value is True:
        result.success = True
        result.termination_reason = "success_early_termination"
        return True

    # check_success_value is False, or check_success() itself raised
    # (check_success_value stays None in that case -- "unknown", not
    # "confirmed failure", per the confirmed design decision).
    result.success = check_success_value  # False, or None if check_success() raised
    result.termination_reason = "environment_failure_done_true"
    return False


def run_episode_edge_pinch(
    env,
    cfg: EdgePinchConfig,
    ctrl_module,
    seed: int,
    rim_local_points: dict,
    record_trace: bool = True,
    obs_capture_callback=None,
) -> EdgePinchEpisodeResult:
    """Drives the edge-pinch state machine for one episode, with Protections
    1 and 2 enforced throughout.

    `env` must already be an initialized LIBERO env. `rim_local_points`
    must be the output of compute_rim_local_points() for this same object
    (computed once, outside the episode loop, since it only needs the
    object's asset geometry, not a live pose).

    obs_capture_callback (2026-09-17, additive, default None -- ZERO effect
    on existing behavior/tests when omitted): optional callable
    `fn(step_count: int, phase: str, obs: dict, action: Optional[np.ndarray],
    done: bool, info: Optional[dict]) -> None`, invoked once right after
    env.reset() (step_count=0, phase="RESET", action=None, done=False,
    info=None) and once after every subsequent env.step() call (both call
    sites: the CLOSE/RELEASE hold-loop branch and the general phase-advance
    branch), with that step's actual obs/action/done/info. This is the only
    hook point task1_edge_pinch_diverse_collection_recorder_v1.py uses to
    capture full per-step observation fields (images, joint/eef/gripper
    state) for HDF5 recording, WITHOUT duplicating this state machine. Must
    never raise -- any exception inside the callback propagates and aborts
    the episode, matching the same fail-loud behavior as any other step in
    this loop; the recorder's own callback is responsible for being
    defensive if that is undesired. Does not alter control flow, targets,
    or any existing field in EdgePinchEpisodeResult/EdgePinchStepRecord.
    """
    result = EdgePinchEpisodeResult(seed=seed)

    # PROTECTION 1: episode-level gate, checked before any env.reset()/step.
    veto_reason = _require_orientation_verified(cfg)
    if veto_reason is not None:
        result.error = veto_reason
        result.aborted = True
        result.abort_reason = "orientation_control_not_verified"
        return result

    edge_local_point = select_edge_local_point(rim_local_points, cfg.edge_azimuth_deg)
    if edge_local_point is None:
        result.error = (
            "EDGE_POINT_UNAVAILABLE: rim_local_points did not yield a usable edge "
            "point (see compute_rim_local_points' 'error' field). Refusing to fall "
            "back to the object's root-body center -- that would silently regress "
            "to the already-falsified whole-object-center strategy."
        )
        result.aborted = True
        result.abort_reason = "edge_point_unavailable"
        return result

    np.random.seed(seed)
    obs = env.reset()
    if obs_capture_callback is not None:
        obs_capture_callback(0, "RESET", obs, None, False, None)

    lift_reference_z: Optional[float] = None
    grasp_start_bowl_z: Optional[float] = None
    step_count = 0
    phase_idx = 0
    # 2026-09-17: zero until LIFT completes and _estimate_grasp_object_offset()
    # fills it in (see the `if phase == "LIFT":` block below) -- MOVE/
    # DESCEND2/RELEASE all run strictly after LIFT in PHASE_NAMES_EDGE, so by
    # the time any of them calls phase_target_pos_edge_pinch(), this is
    # always the real estimate (or the explicit zero-offset fallback).
    grasp_object_offset_m = np.array(ZERO_GRASP_OFFSET_M, dtype=np.float64)

    try:
        edge_world_at_reset = edge_target_point_world(env, cfg.target_object, edge_local_point, ctrl_module)
        result.edge_target_point_world_m = (
            edge_world_at_reset.tolist() if edge_world_at_reset is not None else None
        )

        while phase_idx < len(PHASE_NAMES_EDGE):
            phase = PHASE_NAMES_EDGE[phase_idx]
            phase_step = 0
            max_phase_steps = cfg.max_steps_for_phase(phase)
            stall_pos = _StallTracker(cfg.stall_window_steps, cfg.stall_min_position_improvement_m)
            stall_ori = _StallTracker(cfg.stall_window_steps, cfg.stall_min_orientation_improvement_rad)

            if phase == "CLOSE":
                bowl_pos = ctrl_module.object_xyz(obs, cfg.target_object)
                grasp_start_bowl_z = float(bowl_pos[2])

            while True:
                eef_pos = ctrl_module._eef_pos_with_fallback(env, obs)
                target_pos = phase_target_pos_edge_pinch(
                    phase, cfg, env, ctrl_module, obs, eef_pos, lift_reference_z, edge_local_point,
                    grasp_object_offset_m,
                )

                # PROTECTION 2a: edge-target computation failure -> immediate abort.
                if target_pos is None:
                    reason = f"{phase}: edge_target_point_world returned None (live transform failed)"
                    result.phase_abort_reason[phase] = reason
                    result.aborted = True
                    result.abort_reason = reason
                    break

                # PROTECTION 2b: reachability precheck, BEFORE stepping.
                if phase in CRITICAL_PHASES:
                    # 2026-09-17: MOVE/DESCEND2 use the wider, plate-transport-
                    # area bounds (data-grounded in task1_workspace_bounds_
                    # audit_v1.py's GPU run); APPROACH/DESCEND are UNCHANGED,
                    # still using the original bowl-area workspace_bounds_m.
                    active_bounds = (
                        cfg.workspace_bounds_transport_m
                        if phase in ("MOVE", "DESCEND2")
                        else cfg.workspace_bounds_m
                    )
                    reach_err = check_target_reachable(target_pos, active_bounds)
                    if reach_err is not None:
                        reason = f"{phase}: {reach_err}"
                        result.phase_abort_reason[phase] = reason
                        result.aborted = True
                        result.abort_reason = reason
                        break

                gripper_value = phase_gripper_value_edge_pinch(phase, cfg)
                current_quat = _current_quat(env, obs)

                if phase in ("CLOSE", "RELEASE"):
                    hold_steps = cfg.grasp_hold_steps if phase == "CLOSE" else cfg.release_hold_steps
                    action, position_error_m, orientation_error_rad = compute_action_edge_pinch(
                        cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module
                    )
                    obs, _, done, info = env.step(action)
                    step_count += 1
                    phase_step += 1
                    if obs_capture_callback is not None:
                        obs_capture_callback(step_count, phase, obs, action, done, info)
                    if record_trace:
                        result.records.append(
                            EdgePinchStepRecord(
                                step=step_count,
                                phase=phase,
                                action=action.copy(),
                                eef_pos=eef_pos.copy(),
                                target_pos=target_pos.copy(),
                                position_error_m=position_error_m,
                                orientation_error_rad=orientation_error_rad,
                                target_object_pos=ctrl_module.object_xyz(obs, cfg.target_object),
                                gripper_contact=ctrl_module._gripper_contact_with_object(
                                    env, cfg.target_object
                                ),
                                gripper_part_contact=ctrl_module._gripper_part_contact_with_object(
                                    env, cfg.target_object
                                ),
                                finger_distance_m=_finger_distance_m(env, ctrl_module),
                            )
                        )
                    if done:
                        if phase == "RELEASE":
                            is_success = _handle_release_done_termination(env, step_count, info, result)
                            if is_success:
                                # Success-early-termination (confirmed design,
                                # 2026-09-17): the environment ended itself
                                # because the task succeeded. Return
                                # immediately WITHOUT running
                                # terminal_hold_steps or entering RETREAT --
                                # stepping an already-done environment is
                                # undefined behavior. result.success,
                                # result.termination_reason, result.steps_run
                                # and result.grasp_object_offset_m (set at
                                # LIFT-end, unaffected) are all already
                                # populated by _handle_release_done_termination.
                                return result
                            raise RuntimeError(
                                f"Environment ended unexpectedly during RELEASE "
                                f"(check_success()={result.done_early_termination_check_success}, "
                                f"info={result.done_early_termination_info})"
                            )
                        else:
                            _diagnose_and_raise_on_done(env, phase, step_count, info, result)
                    if step_count >= cfg.max_episode_steps:
                        raise RuntimeError("Episode step budget exhausted")
                    if phase_step >= hold_steps:
                        result.phase_transition_steps[phase] = step_count
                        result.phase_timed_out[phase] = False
                        if phase == "CLOSE":
                            lift_reference_z = float(ctrl_module._eef_pos_with_fallback(env, obs)[2])
                            result.finger_distance_at_close_end_m = _finger_distance_m(env, ctrl_module)
                            gpc = ctrl_module._gripper_part_contact_with_object(env, cfg.target_object)
                            result.bilateral_contact_at_close_end = bool(
                                gpc.get("left_fingerpad") and gpc.get("right_fingerpad")
                            )
                        break
                    continue

                action, position_error_m, orientation_error_rad = compute_action_edge_pinch(
                    cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module
                )
                obs, _, done, info = env.step(action)
                step_count += 1
                phase_step += 1
                if obs_capture_callback is not None:
                    obs_capture_callback(step_count, phase, obs, action, done, info)
                if record_trace:
                    result.records.append(
                        EdgePinchStepRecord(
                            step=step_count,
                            phase=phase,
                            action=action.copy(),
                            eef_pos=eef_pos.copy(),
                            target_pos=target_pos.copy(),
                            position_error_m=position_error_m,
                            orientation_error_rad=orientation_error_rad,
                            target_object_pos=ctrl_module.object_xyz(obs, cfg.target_object),
                            gripper_contact=ctrl_module._gripper_contact_with_object(
                                env, cfg.target_object
                            ),
                            gripper_part_contact=ctrl_module._gripper_part_contact_with_object(
                                env, cfg.target_object
                            ),
                            finger_distance_m=_finger_distance_m(env, ctrl_module),
                        )
                    )

                if done:
                    _diagnose_and_raise_on_done(env, phase, step_count, info, result)
                if step_count >= cfg.max_episode_steps:
                    raise RuntimeError("Episode step budget exhausted")

                # PROTECTION 2c: stall detection during critical phases.
                # 2026-09-16 FIX (diagnosed from the 16-candidate calibration
                # run): several DESCEND candidates were aborted with
                # orientation_error_rad ALREADY well under
                # orientation_tolerance_rad (e.g. 0.0232, 0.0220 vs tol=0.1),
                # purely because the orientation error was oscillating by a
                # tiny amount below the stall_min_orientation_improvement_rad
                # threshold (or even trending slightly upward within noise),
                # while position was still improving steadily and just a few
                # mm from converging. Once orientation has already converged,
                # further requiring monotonic improvement in an already-small,
                # noisy quantity is not a meaningful failure signal. Per the
                # user's explicit instruction: orientation-stall now only
                # fires when orientation is NOT YET converged AND it is
                # GENUINELY WORSENING (improvement < 0, not just
                # sub-threshold), not merely "not improving enough".
                # Position-stall logic is UNCHANGED.
                if phase in CRITICAL_PHASES:
                    stall_pos.update(position_error_m)
                    ori_already_converged = (
                        orientation_error_rad is not None
                        and orientation_error_rad < cfg.orientation_tolerance_rad
                    )
                    if orientation_error_rad is not None:
                        stall_ori.update(orientation_error_rad)
                    pos_stalled = stall_pos.is_stalled()
                    ori_improvement = stall_ori.improvement()
                    ori_stalled = (
                        not ori_already_converged
                        and orientation_error_rad is not None
                        and stall_ori.is_stalled()
                        and ori_improvement is not None
                        and ori_improvement < 0.0
                    )
                    if pos_stalled or ori_stalled:
                        reason = (
                            f"{phase}: stalled after {phase_step} steps "
                            f"(pos_improvement={stall_pos.improvement()} over "
                            f"{cfg.stall_window_steps} steps [threshold="
                            f"{cfg.stall_min_position_improvement_m}], "
                            f"pos_stalled={pos_stalled}; "
                            f"ori_improvement={ori_improvement} over "
                            f"{cfg.stall_window_steps} steps [threshold="
                            f"{cfg.stall_min_orientation_improvement_rad}], "
                            f"ori_already_converged={ori_already_converged}, "
                            f"ori_stalled={ori_stalled} "
                            f"(requires NOT converged AND genuinely worsening); "
                            f"current position_error_m={position_error_m:.4f} "
                            f"[tol={cfg.position_tolerance_m}], current "
                            f"orientation_error_rad="
                            f"{'N/A' if orientation_error_rad is None else f'{orientation_error_rad:.4f}'} "
                            f"[tol={cfg.orientation_tolerance_rad}])"
                        )
                        result.phase_abort_reason[phase] = reason
                        result.aborted = True
                        result.abort_reason = reason
                        break

                pos_ok = position_error_m < cfg.position_tolerance_m
                ori_ok = (
                    orientation_error_rad is None
                    or orientation_error_rad < cfg.orientation_tolerance_rad
                )
                if pos_ok and ori_ok:
                    result.phase_transition_steps[phase] = step_count
                    result.phase_timed_out[phase] = False
                    break
                if phase_step >= max_phase_steps:
                    result.phase_transition_steps[phase] = step_count
                    result.phase_timed_out[phase] = True
                    if phase in CRITICAL_PHASES:
                        # PROTECTION 2d: hard timeout on a critical phase is an
                        # abort, NOT a fall-through into CLOSE/LIFT (this is the
                        # loophole closed relative to the original controller).
                        # 2026-09-16: explicitly reports WHICH of position/
                        # orientation failed to converge, with actual numeric
                        # values, per the user's "明确姿态未收敛时的超时原因".
                        ori_err_str = (
                            "N/A" if orientation_error_rad is None else f"{orientation_error_rad:.4f}"
                        )
                        reason = (
                            f"{phase}: hard timeout after {max_phase_steps} steps "
                            f"(position_error_m={position_error_m:.4f} "
                            f"[{'OK' if pos_ok else 'NOT_CONVERGED'}, tol={cfg.position_tolerance_m}], "
                            f"orientation_error_rad={ori_err_str} "
                            f"[{'OK' if ori_ok else 'NOT_CONVERGED'}, tol={cfg.orientation_tolerance_rad}])"
                        )
                        result.phase_abort_reason[phase] = reason
                        result.aborted = True
                        result.abort_reason = reason
                    break

            if result.aborted:
                break  # PROTECTION 2: never proceed to the next phase after an abort.

            # 2026-09-16 FIX (diagnosed from the 16-candidate calibration run:
            # finger_distance_at_lift_end_m was identical ~0.0936m across every
            # non-aborted candidate regardless of finger_distance_at_close_end_m,
            # because it was being measured AFTER the terminal_hold loop below,
            # which unconditionally commands the gripper OPEN
            # (zero_action[6]=cfg.gripper_open_value) -- a leftover from the
            # original controller's full RELEASE-at-plate pipeline, inappropriate
            # for this truncated APPROACH->DESCEND->CLOSE->LIFT calibration
            # subset. Capture the LIFT-end grasp-quality metrics HERE, the
            # instant LIFT itself completes (success or per-phase timeout), i.e.
            # BEFORE the gripper is ever forced open again -- this is true
            # regardless of whether the full 8-phase episode or the truncated
            # calibration subset is being run.
            if phase == "LIFT":
                result.finger_distance_at_lift_end_m = _finger_distance_m(env, ctrl_module)
                lift_end_gpc = ctrl_module._gripper_part_contact_with_object(env, cfg.target_object)
                result.bilateral_contact_at_lift_end = bool(
                    lift_end_gpc.get("left_fingerpad") and lift_end_gpc.get("right_fingerpad")
                )
                if grasp_start_bowl_z is not None:
                    bowl_pos_at_lift_end = ctrl_module.object_xyz(obs, cfg.target_object)
                    result.grasp_lift_delta_m = float(bowl_pos_at_lift_end[2] - grasp_start_bowl_z)

                # 2026-09-17 FIX (diagnosed from the seed=950004 full-episode
                # smoke-test diagnosis: MOVE/DESCEND2/RELEASE driving the EEF
                # directly to the plate's own coordinates left the bowl
                # consistently ~3.3-3.6cm horizontally and ~5.6cm vertically
                # off-target, because edge-pinch holds the bowl off-center
                # from the EEF by a roughly constant amount). Estimate that
                # constant offset HERE, at the same point LIFT-end metrics are
                # already captured, using only records already gathered
                # during LIFT -- no new instrumentation, no extra sim steps.
                (
                    grasp_object_offset_m,
                    grasp_object_offset_std_m,
                    grasp_object_offset_reliable,
                    grasp_object_offset_fallback_used,
                ) = _estimate_grasp_object_offset(result.records, cfg.grasp_offset_estimation_window_steps)
                result.grasp_object_offset_m = grasp_object_offset_m.tolist()
                result.grasp_object_offset_std_m = (
                    grasp_object_offset_std_m.tolist() if grasp_object_offset_std_m is not None else None
                )
                result.grasp_object_offset_reliable = grasp_object_offset_reliable
                result.grasp_object_offset_fallback_used = grasp_object_offset_fallback_used

            phase_idx += 1

        if result.aborted:
            result.steps_run = step_count
            return result

        for _ in range(cfg.terminal_hold_steps):
            zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
            zero_action[6] = cfg.gripper_open_value
            obs, _, done, info = env.step(zero_action)
            step_count += 1
            if obs_capture_callback is not None:
                obs_capture_callback(step_count, "TERMINAL_HOLD", obs, zero_action, done, info)
            if done or step_count >= cfg.max_episode_steps:
                break

        # bilateral_contact_lift_fraction is computed from records captured
        # DURING the LIFT phase's own steps (unaffected by the terminal_hold
        # bug above) -- left here, after terminal_hold, purely because it only
        # needs result.records, not live env/obs state.
        lift_records = [r for r in result.records if r.phase == "LIFT"]
        if lift_records:
            bilateral_count = sum(
                1
                for r in lift_records
                if r.gripper_part_contact.get("left_fingerpad")
                and r.gripper_part_contact.get("right_fingerpad")
            )
            result.bilateral_contact_lift_fraction = bilateral_count / len(lift_records)

        result.success = bool(env.check_success())
        result.steps_run = step_count
        # 2026-09-17 (design: claude/task1_edge_pinch_release_success_
        # termination_design_v1_20260917.json): normal_completion is the
        # third value in the unified termination_reason vocabulary,
        # alongside success_early_termination and
        # environment_failure_done_true (both set in
        # _handle_release_done_termination). This is the pre-existing path
        # where all 8 phases + terminal_hold ran without done=True ever
        # firing during CLOSE/RELEASE.
        result.termination_reason = "normal_completion"

    except Exception as exc:  # noqa: BLE001 -- smoke-test driver, report cleanly
        result.error = f"{type(exc).__name__}: {exc}"
        result.steps_run = step_count

    return result


def run_calibration_subset_edge_pinch(
    env, cfg: EdgePinchConfig, ctrl_module, seed: int, rim_local_points: dict
) -> EdgePinchEpisodeResult:
    """Same driver as run_episode_edge_pinch but truncates PHASE_NAMES_EDGE
    to APPROACH->DESCEND->CLOSE->LIFT, mirroring the original controller's
    _run_calibration_subset. Implemented by temporarily swapping the module-
    level PHASE_NAMES_EDGE list (same pattern as the original), restored in
    a finally block.
    """
    global PHASE_NAMES_EDGE
    original = PHASE_NAMES_EDGE[:]
    try:
        PHASE_NAMES_EDGE[:] = ["APPROACH", "DESCEND", "CLOSE", "LIFT"]
        return run_episode_edge_pinch(env, cfg, ctrl_module, seed, rim_local_points, record_trace=True)
    finally:
        PHASE_NAMES_EDGE[:] = original


# --------------------------------------------------------------------------
# Calibration sweep
# --------------------------------------------------------------------------


@dataclass
class EdgePinchCalibrationResult:
    edge_azimuth_deg: float
    target_ee_ori: tuple
    standoff_m: float
    grasp_lift_delta_m: Optional[float]
    finger_distance_at_close_end_m: Optional[float]
    finger_distance_at_lift_end_m: Optional[float]
    bilateral_contact_at_close_end: Optional[bool]
    bilateral_contact_at_lift_end: Optional[bool]
    bilateral_contact_lift_fraction: Optional[float]
    aborted: bool
    abort_reason: Optional[str]
    error: Optional[str]
    phase_transition_steps: dict = field(default_factory=dict)
    phase_timed_out: dict = field(default_factory=dict)

    @property
    def looks_successful(self) -> bool:
        if self.aborted or self.error is not None:
            return False
        if self.grasp_lift_delta_m is None or self.grasp_lift_delta_m <= 0.02:
            return False
        if not self.bilateral_contact_at_close_end:
            return False
        if self.bilateral_contact_lift_fraction is None or self.bilateral_contact_lift_fraction < 0.8:
            return False
        if self.finger_distance_at_lift_end_m is None:
            return False
        lo, hi = (0.015, 0.028)
        return lo <= self.finger_distance_at_lift_end_m <= hi


def calibrate_edge_pinch(
    make_env_fn,
    ctrl_module,
    seed: int,
    edge_azimuth_candidates: list[float],
    target_ori_candidates: list[tuple[float, float, float]],
    standoff_candidates: list[float],
    orientation_control_verified: bool,
) -> list[EdgePinchCalibrationResult]:
    """Runs one short (APPROACH->DESCEND->CLOSE->LIFT) episode per
    (edge_azimuth, target_ee_ori, standoff) candidate. NOT invoked by this
    delivery -- only reachable via --calibrate, which itself is refused
    (see build_arg_parser/main) unless --orientation-verified was also
    passed, per Protection 1.
    """
    results: list[EdgePinchCalibrationResult] = []
    rim_local_points = compute_rim_local_points(make_env_fn, ctrl_module, DEFAULT_TARGET_OBJECT)
    for azimuth in edge_azimuth_candidates:
        for target_ori in target_ori_candidates:
            for standoff in standoff_candidates:
                cfg = EdgePinchConfig(
                    edge_azimuth_deg=azimuth,
                    target_ee_ori=tuple(target_ori),
                    standoff_m=standoff,
                    orientation_control_verified=orientation_control_verified,
                )
                env = make_env_fn()
                try:
                    partial = run_calibration_subset_edge_pinch(env, cfg, ctrl_module, seed, rim_local_points)
                    results.append(
                        EdgePinchCalibrationResult(
                            edge_azimuth_deg=azimuth,
                            target_ee_ori=tuple(target_ori),
                            standoff_m=standoff,
                            grasp_lift_delta_m=partial.grasp_lift_delta_m,
                            finger_distance_at_close_end_m=partial.finger_distance_at_close_end_m,
                            finger_distance_at_lift_end_m=partial.finger_distance_at_lift_end_m,
                            bilateral_contact_at_close_end=partial.bilateral_contact_at_close_end,
                            bilateral_contact_at_lift_end=partial.bilateral_contact_at_lift_end,
                            bilateral_contact_lift_fraction=partial.bilateral_contact_lift_fraction,
                            aborted=partial.aborted,
                            abort_reason=partial.abort_reason,
                            error=partial.error,
                            phase_transition_steps=dict(partial.phase_transition_steps),
                            phase_timed_out=dict(partial.phase_timed_out),
                        )
                    )
                finally:
                    env.close()
    return results


# --------------------------------------------------------------------------
# Smoke test (multi-seed, FULL 8-phase episode -- NOT the truncated
# calibration subset) for a single already-calibrated (azimuth, standoff)
# candidate. Added 2026-09-16 per the user's explicit request, following the
# original design's planned sequence ("先做单 seed 校准，再做少量 smoke
# test"): run_calibration_subset_edge_pinch truncates to
# APPROACH->DESCEND->CLOSE->LIFT, so env.check_success() is never meaningful
# there (MOVE/DESCEND2/RELEASE never run, so the bowl is never actually
# placed on the plate). This smoke test runs the FULL PHASE_NAMES_EDGE
# sequence instead, so env.check_success() reflects the real task-success
# criterion.
# --------------------------------------------------------------------------


def run_smoke_test_edge_pinch(
    make_env_fn,
    ctrl_module,
    seeds: list[int],
    edge_azimuth_deg: float,
    standoff_candidates: list[float],
    target_ee_ori: tuple[float, float, float],
    orientation_control_verified: bool,
) -> list[dict]:
    """Runs one FULL (all 8 phases) episode per (standoff, seed) pair for a
    FIXED edge_azimuth_deg/target_ee_ori (the already-calibrated candidate
    under test), across multiple NEW seeds. NOT invoked by this delivery --
    only reachable via --smoke-test, refused unless --orientation-verified
    is also passed, per Protection 1 (same gate as --calibrate).

    Returns a list of plain dicts (not a dataclass) since the reporting
    need here is per-run summary fields only -- no reason to introduce a
    third result type alongside EdgePinchEpisodeResult/EdgePinchCalibrationResult.
    """
    results: list[dict] = []
    rim_local_points = compute_rim_local_points(make_env_fn, ctrl_module, DEFAULT_TARGET_OBJECT)
    for standoff in standoff_candidates:
        for seed in seeds:
            cfg = EdgePinchConfig(
                edge_azimuth_deg=edge_azimuth_deg,
                target_ee_ori=tuple(target_ee_ori),
                standoff_m=standoff,
                orientation_control_verified=orientation_control_verified,
            )
            env = make_env_fn()
            try:
                result = run_episode_edge_pinch(
                    env, cfg, ctrl_module, seed, rim_local_points, record_trace=True
                )
                results.append(
                    {
                        "edge_azimuth_deg": edge_azimuth_deg,
                        "standoff_m": standoff,
                        "seed": seed,
                        "aborted": result.aborted,
                        "abort_reason": result.abort_reason,
                        "error": result.error,
                        "steps_run": result.steps_run,
                        "grasp_lift_delta_m": result.grasp_lift_delta_m,
                        "finger_distance_at_lift_end_m": result.finger_distance_at_lift_end_m,
                        "bilateral_contact_at_close_end": result.bilateral_contact_at_close_end,
                        "bilateral_contact_at_lift_end": result.bilateral_contact_at_lift_end,
                        "bilateral_contact_lift_fraction": result.bilateral_contact_lift_fraction,
                        "env_check_success": result.success,
                        "phase_transition_steps": dict(result.phase_transition_steps),
                        "phase_timed_out": dict(result.phase_timed_out),
                        # 2026-09-17: surfaced regardless of aborted/error status --
                        # None simply means LIFT never completed (offset was never
                        # estimated), which is itself informative, not an omission.
                        "grasp_object_offset_m": result.grasp_object_offset_m,
                        "grasp_object_offset_std_m": result.grasp_object_offset_std_m,
                        "grasp_object_offset_reliable": result.grasp_object_offset_reliable,
                        "grasp_object_offset_fallback_used": result.grasp_object_offset_fallback_used,
                        # 2026-09-17 diagnostic addition: populated only when env.step()
                        # returned done=True mid-phase (see _diagnose_and_raise_on_done);
                        # None in the normal (no early termination) case.
                        "done_early_termination_phase": result.done_early_termination_phase,
                        "done_early_termination_step": result.done_early_termination_step,
                        "done_early_termination_check_success": result.done_early_termination_check_success,
                        "done_early_termination_check_success_error": result.done_early_termination_check_success_error,
                        "done_early_termination_info": result.done_early_termination_info,
                        # 2026-09-17 (design: claude/task1_edge_pinch_release_success_
                        # termination_design_v1_20260917.json): unified termination
                        # classification -- 'normal_completion', 'success_early_
                        # termination', 'environment_failure_done_true', or None if
                        # the episode was aborted/raised via some other path (e.g.
                        # PROTECTION 1/2 aborts, CLOSE-phase done=True, step-budget
                        # exhaustion) that this design intentionally left out of scope.
                        "termination_reason": result.termination_reason,
                    }
                )
            finally:
                env.close()
    return results


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def write_smoke_test_outputs_edge_pinch(
    output_dir: Path,
    results: list[dict],
    bddl_path: str,
    orientation_control_verified: bool,
) -> Path:
    payload = {
        "controller_source": "privileged_simulator_pose",
        "generation_only": True,
        "not_a_model_evaluation_input": True,
        "protocol": "task1_edge_pinch_controller_smoke_test_v1",
        "note": (
            "FULL 8-phase episode per run (APPROACH..RETREAT), NOT the truncated "
            "calibration subset -- env_check_success is meaningful here since "
            "MOVE/DESCEND2/RELEASE actually run and the bowl is actually placed "
            "toward the plate. This is a PRIVILEGED demo-generation controller "
            "result, NOT a SmolVLA model evaluation and NOT evidence toward Task1 "
            "strict closed-loop model eval status."
        ),
        "bddl_path": bddl_path,
        "orientation_control_verified": orientation_control_verified,
        "protections_applied": [
            "orientation_control_verification_gate",
            "workspace_reachability_check",
            "stall_detection_abort",
            "hard_timeout_abort_on_critical_phase",
        ],
        "num_runs": len(results),
        "runs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "edge_pinch_smoke_test_summary.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"edge_pinch_smoke_test_summary written to: {out_path}")
    return out_path


def write_calibration_outputs_edge_pinch(
    output_dir: Path,
    results: list[EdgePinchCalibrationResult],
    seed: int,
    bddl_path: str,
    orientation_probe_report: Optional[dict] = None,
    orientation_control_verified: bool = False,
) -> Path:
    payload = {
        "controller_source": "privileged_simulator_pose",
        "generation_only": True,
        "not_a_model_evaluation_input": True,
        "protocol": "task1_edge_pinch_controller_calibrate_v1",
        "bddl_path": bddl_path,
        "calibrate_seed": seed,
        "protections_applied": [
            "orientation_control_verification_gate",
            "workspace_reachability_check",
            "stall_detection_abort",
            "hard_timeout_abort_on_critical_phase",
        ],
        "orientation_control_verified": orientation_control_verified,
        "orientation_probe_report": orientation_probe_report,
        "num_candidates": len(results),
        "candidates": [
            {
                "edge_azimuth_deg": r.edge_azimuth_deg,
                "target_ee_ori": list(r.target_ee_ori),
                "standoff_m": r.standoff_m,
                "grasp_lift_delta_m": r.grasp_lift_delta_m,
                "finger_distance_at_close_end_m": r.finger_distance_at_close_end_m,
                "finger_distance_at_lift_end_m": r.finger_distance_at_lift_end_m,
                "bilateral_contact_at_close_end": r.bilateral_contact_at_close_end,
                "bilateral_contact_at_lift_end": r.bilateral_contact_at_lift_end,
                "bilateral_contact_lift_fraction": r.bilateral_contact_lift_fraction,
                "aborted": r.aborted,
                "abort_reason": r.abort_reason,
                "looks_successful": r.looks_successful,
                "error": r.error,
                "phase_transition_steps": r.phase_transition_steps,
                "phase_timed_out": r.phase_timed_out,
            }
            for r in results
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "edge_pinch_calibration_summary.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"edge_pinch_calibration_summary written to: {out_path}")
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bddl-path", default=DEFAULT_BDDL_PATH)
    p.add_argument("--controller-script", default=DEFAULT_CONTROLLER_SCRIPT)
    p.add_argument("--target-object", default=DEFAULT_TARGET_OBJECT)
    p.add_argument("--target-container", default=DEFAULT_TARGET_CONTAINER)
    p.add_argument(
        "--output-dir",
        default="/root/smolvla-eval-prep/results/controller/task1_edge_pinch_controller_calibrate_v1",
    )
    p.add_argument("--seed", type=int, default=900001)

    p.add_argument(
        "--probe-orientation",
        action="store_true",
        help="Run ONLY the live orientation-frame probe (Protection 1's verification "
        "mechanism) and exit; does not run any calibration.",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Run the edge-pinch calibration sweep. Refused unless --orientation-verified "
        "is also passed (Protection 1).",
    )
    p.add_argument(
        "--orientation-verified",
        action="store_true",
        help="Operator confirmation that --probe-orientation's report was reviewed and "
        "the observed/commanded rotations are consistent. Required for --calibrate to "
        "run at all.",
    )

    p.add_argument("--edge-azimuth-candidates", type=float, nargs="+", default=[0.0])
    p.add_argument("--standoff-candidates", type=float, nargs="+", default=[0.05])

    p.add_argument(
        "--probe-axis-index", type=int, default=1, choices=[0, 1, 2],
        help="Which action[3:6] component to probe (0=X-channel, 1=Y-channel, "
        "2=Z-channel per the action-space index). Only used with --probe-orientation.",
    )
    p.add_argument("--probe-delta", type=float, default=0.3)
    p.add_argument("--probe-steps", type=int, default=5)

    p.add_argument(
        "--probe-combined-axes",
        action="store_true",
        help="Run ONLY the low-cost supplementary combined-three-axis-simultaneous "
        "probe (probe_combined_axes_action_response) and exit; does not re-verify "
        "the three single-axis correspondence conclusions and does not run any "
        "calibration.",
    )
    p.add_argument("--combined-probe-delta", type=float, default=0.15)

    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a multi-seed, FULL 8-phase smoke test (run_smoke_test_edge_pinch) for "
        "a single already-calibrated (azimuth, standoff) candidate. Unlike --calibrate, "
        "runs the full APPROACH..RETREAT sequence so env.check_success() is meaningful. "
        "Refused unless --orientation-verified is also passed (Protection 1).",
    )
    p.add_argument("--smoke-test-edge-azimuth", type=float, default=90.0)
    p.add_argument("--smoke-test-standoff-candidates", type=float, nargs="+", default=[0.03, 0.05])
    p.add_argument(
        "--smoke-test-seeds", type=int, nargs="+",
        default=[950001, 950002, 950003, 950004, 950005],
        help="NEW seeds, distinct from the calibration --seed (900001), so this smoke "
        "test is not just re-running the same episode already seen during calibration.",
    )
    return p


def _build_make_env_fn(ctrl_module, bddl_path: str):
    """Builds the env-construction closure used by --probe-orientation and
    --calibrate.

    Deliberately delegates ENTIRELY to the already-confirmed
    task1_privileged_pickplace_controller_v1.make_env(bddl_path) -- i.e. the
    exact same OffScreenRenderEnv(bddl_file_name=bddl_path,
    camera_heights=128, camera_widths=128, horizon=1000, use_camera_obs=True)
    construction used by every prior calibration round in this project (v1
    through v10, and the official-demo audit's controller-module reuse) --
    rather than reimplementing env construction here. This guarantees the
    edge-pinch controller starts from an identically-configured environment,
    with no risk of a subtly different camera/horizon/render setting causing
    an apples-to-oranges comparison against the v1-v10 calibration history.

    Import deferral (LIBERO/robosuite) already lives inside
    ctrl_module.make_env() itself (see the original controller's own
    docstring: "Import is deferred into this function ... so that --help and
    static/py_compile checks work even without libero installed"), so this
    wrapper adds no new import-time dependency either -- calling the
    returned closure is what triggers the deferred import, not defining it.
    """
    return lambda: ctrl_module.make_env(bddl_path)


def main() -> int:
    args = build_arg_parser().parse_args()
    ctrl_module = _load_controller_module(args.controller_script)
    make_env_fn = _build_make_env_fn(ctrl_module, args.bddl_path)

    if args.probe_orientation:
        report = probe_orientation_action_response(
            make_env_fn,
            ctrl_module,
            probe_axis_index=args.probe_axis_index,
            probe_delta=args.probe_delta,
            probe_steps=args.probe_steps,
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "orientation_probe_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"orientation_probe_report written to: {out_path}")
        return 0

    if args.probe_combined_axes:
        report = probe_combined_axes_action_response(
            make_env_fn,
            ctrl_module,
            probe_delta=args.combined_probe_delta,
            probe_steps=args.probe_steps,
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "orientation_probe_combined_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"orientation_probe_combined_report written to: {out_path}")
        return 0

    if args.smoke_test:
        if not args.orientation_verified:
            print(
                "REFUSING TO RUN --smoke-test: --orientation-verified was not passed. "
                "(Protection 1)",
                file=sys.stderr,
            )
            return 2
        results = run_smoke_test_edge_pinch(
            make_env_fn,
            ctrl_module,
            seeds=args.smoke_test_seeds,
            edge_azimuth_deg=args.smoke_test_edge_azimuth,
            standoff_candidates=args.smoke_test_standoff_candidates,
            target_ee_ori=EdgePinchConfig().target_ee_ori,
            orientation_control_verified=True,
        )
        write_smoke_test_outputs_edge_pinch(
            Path(args.output_dir), results, args.bddl_path, orientation_control_verified=True,
        )
        return 0

    if args.calibrate:
        if not args.orientation_verified:
            print(
                "REFUSING TO RUN --calibrate: --orientation-verified was not passed. "
                "Run --probe-orientation first, review its report, then re-run with "
                "--orientation-verified once the observed and commanded rotations are "
                "confirmed consistent. (Protection 1)",
                file=sys.stderr,
            )
            return 2
        results = calibrate_edge_pinch(
            make_env_fn,
            ctrl_module,
            seed=args.seed,
            edge_azimuth_candidates=args.edge_azimuth_candidates,
            target_ori_candidates=[EdgePinchConfig().target_ee_ori],
            standoff_candidates=args.standoff_candidates,
            orientation_control_verified=True,
        )
        write_calibration_outputs_edge_pinch(
            Path(args.output_dir), results, args.seed, args.bddl_path,
            orientation_control_verified=True,
        )
        return 0

    print("No action requested. Pass --probe-orientation or --calibrate.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
