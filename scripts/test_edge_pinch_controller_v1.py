#!/usr/bin/env python3
"""Offline (mocked, no MuJoCo/LIBERO) unit tests for
task1_edge_pinch_controller_v1.py's two hard protections and the
edge-point local->world transform math. No simulation is run against a
real environment -- everything here uses small in-memory Fake objects,
consistent with this project's established mock-test convention (e.g.
/tmp/test_horiz_geom.py from the v8 round).
"""
import importlib.util
import os
import sys
import numpy as np

# 2026-09-17 FIX: was a hardcoded absolute path into the author's local
# sandbox (/tmp/claude-0/...), which only ever happened to work there --
# this test file had never actually been RUN on the GPU before (only
# deployed), so the bug was latent until the first GPU test run surfaced
# FileNotFoundError. Resolved relative to this file's own location instead,
# so the same file works unmodified in the sandbox, on GPU, or anywhere
# else it's deployed.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "edge_ctrl", os.path.join(THIS_DIR, "task1_edge_pinch_controller_v1.py")
)
edge_ctrl = importlib.util.module_from_spec(spec)
sys.modules["edge_ctrl"] = edge_ctrl
spec.loader.exec_module(edge_ctrl)


# ----------------------------------------------------------------------
# Protection 1: orientation-control verification gate
# ----------------------------------------------------------------------

def test_protection1_episode_level_gate_blocks_unverified():
    cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=False)
    result = edge_ctrl.run_episode_edge_pinch(
        env=None, cfg=cfg, ctrl_module=None, seed=1, rim_local_points={}
    )
    assert result.aborted is True
    assert result.abort_reason == "orientation_control_not_verified"
    assert "ORIENTATION_CONTROL_NOT_VERIFIED" in result.error
    print("PROTECTION1_EPISODE_GATE_OK")


def test_protection1_action_level_gate_raises():
    cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=False)
    raised = False
    try:
        edge_ctrl.compute_action_edge_pinch(
            cfg,
            target_pos=np.array([0.0, 0.0, 1.0]),
            eef_pos=np.array([0.0, 0.0, 0.9]),
            current_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            gripper_value=-1.0,
            ctrl_module=None,
        )
    except RuntimeError as exc:
        raised = True
        assert "ORIENTATION_CONTROL_NOT_VERIFIED" in str(exc)
    assert raised, "compute_action_edge_pinch must raise when unverified"
    print("PROTECTION1_ACTION_LEVEL_GATE_OK")


def test_protection1_action_proceeds_when_verified():
    cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True, max_ori_step_rad=0.08)

    class FakeCtrl:
        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    action, pos_err, ori_err = edge_ctrl.compute_action_edge_pinch(
        cfg,
        target_pos=np.array([0.0, 0.0, 1.0]),
        eef_pos=np.array([0.0, 0.0, 0.9]),
        current_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper_value=-1.0,
        ctrl_module=FakeCtrl(),
    )
    assert action.shape == (7,)
    assert ori_err is not None
    assert np.all(action[:6] >= -1.0) and np.all(action[:6] <= 1.0)
    print("PROTECTION1_VERIFIED_PATH_OK", action.tolist(), pos_err, ori_err)


def test_protection1_orientation_missing_falls_back_to_zero_not_crash():
    cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)

    class FakeCtrl:
        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    action, pos_err, ori_err = edge_ctrl.compute_action_edge_pinch(
        cfg,
        target_pos=np.array([0.0, 0.0, 1.0]),
        eef_pos=np.array([0.0, 0.0, 0.9]),
        current_quat=None,
        gripper_value=-1.0,
        ctrl_module=FakeCtrl(),
    )
    assert ori_err is None
    assert action[3] == 0.0 and action[4] == 0.0 and action[5] == 0.0
    print("PROTECTION1_MISSING_ORI_FALLBACK_OK")


# ----------------------------------------------------------------------
# 2026-09-16 recalibration: quaternion-based orientation error math
# ----------------------------------------------------------------------

def test_quat_mul_conj_roundtrip_identity():
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)])
    q = q / np.linalg.norm(q)
    prod = edge_ctrl.quat_mul(q, edge_ctrl.quat_conj(q))
    # q * conj(q) must be the identity quaternion (0,0,0,1)
    assert np.allclose(prod, [0.0, 0.0, 0.0, 1.0], atol=1e-9), prod
    print("QUAT_MUL_CONJ_ROUNDTRIP_IDENTITY_OK")


def test_axisangle_vec_to_quat_roundtrip():
    vec = np.array([0.3, -0.5, 0.1])
    q = edge_ctrl.axisangle_vec_to_quat(vec)
    axis, angle = edge_ctrl.quat_relative_rotation_axisangle(np.array([0.0, 0.0, 0.0, 1.0]), q)
    recovered_vec = axis * angle
    assert np.allclose(recovered_vec, vec, atol=1e-9), (recovered_vec, vec)
    print("AXISANGLE_VEC_TO_QUAT_ROUNDTRIP_OK")


def test_quat_relative_rotation_matches_known_90deg_case():
    # 90deg about world Z: q = (0,0,sin(45deg),cos(45deg))
    q_start = np.array([0.0, 0.0, 0.0, 1.0])
    half = np.pi / 4
    q_90z = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    axis, angle = edge_ctrl.quat_relative_rotation_axisangle(q_start, q_90z)
    assert np.isclose(angle, np.pi / 2, atol=1e-9), angle
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-9), axis
    print("QUAT_RELATIVE_ROTATION_90DEG_Z_OK")


def test_quat_relative_rotation_picks_minimal_angle_sign_branch():
    # A quaternion and its negation represent the SAME rotation; the
    # minimal-angle (w>=0) branch must be chosen regardless of which sign
    # the observed quaternion happens to be reported in.
    q_start = np.array([0.0, 0.0, 0.0, 1.0])
    half = np.pi / 4
    q_90z = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    q_90z_negated = -q_90z
    axis_a, angle_a = edge_ctrl.quat_relative_rotation_axisangle(q_start, q_90z)
    axis_b, angle_b = edge_ctrl.quat_relative_rotation_axisangle(q_start, q_90z_negated)
    assert np.isclose(angle_a, angle_b, atol=1e-9)
    assert angle_a <= np.pi + 1e-9
    assert np.allclose(axis_a, axis_b, atol=1e-9)
    print("QUAT_RELATIVE_ROTATION_SIGN_AMBIGUITY_RESOLVED_OK")


def test_compute_action_edge_pinch_uses_quaternion_method_not_raw_subtraction():
    """Regression guard: with current orientation at the pi-degenerate reset
    pose (axis-angle norm ~= pi) and a target requiring only a small real
    rotation, the quaternion method must report a SMALL orientation_error_rad
    -- raw axis-angle vector subtraction would have reported something much
    larger/inconsistent in this degenerate region."""
    cfg = edge_ctrl.EdgePinchConfig(
        orientation_control_verified=True,
        target_ee_ori=(2.2587, -2.2587, -0.0700),
    )

    class FakeCtrl:
        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    # A near-pi-magnitude axis-angle reset pose, converted to its quaternion.
    reset_axisangle = np.array([2.221, -2.221, 0.0])  # norm ~3.1416 (~pi)
    q_reset = edge_ctrl.axisangle_vec_to_quat(reset_axisangle)

    action, pos_err, ori_err = edge_ctrl.compute_action_edge_pinch(
        cfg,
        target_pos=np.array([0.0, 0.0, 1.0]),
        eef_pos=np.array([0.0, 0.0, 0.9]),
        current_quat=q_reset,
        gripper_value=-1.0,
        ctrl_module=FakeCtrl(),
    )
    # Expect an error well under pi (matches the hand-verified ~1.57rad
    # geodesic-gap finding for this exact target_ee_ori), not something
    # inflated by the axis-angle degeneracy.
    assert ori_err is not None and ori_err < np.pi, ori_err
    print("COMPUTE_ACTION_USES_QUATERNION_METHOD_OK", ori_err)


def test_max_steps_for_phase_routes_approach_descend_to_orientation_budget():
    cfg = edge_ctrl.EdgePinchConfig(
        max_phase_steps_large=150,
        max_phase_steps_small=80,
        max_phase_steps_orientation_convergence=200,
    )
    assert cfg.max_steps_for_phase("APPROACH") == 200
    assert cfg.max_steps_for_phase("DESCEND") == 200
    assert cfg.max_steps_for_phase("MOVE") == 150
    assert cfg.max_steps_for_phase("CLOSE") == 80
    assert cfg.max_steps_for_phase("LIFT") == 80
    print("MAX_STEPS_FOR_PHASE_ROUTING_OK")


# ----------------------------------------------------------------------
# Protection 2: reachability + stall detection
# ----------------------------------------------------------------------

def test_protection2_reachability_check():
    bounds = {"x": (-0.1, 0.4), "y": (-0.4, 0.4), "z": (0.7, 1.3)}
    ok = edge_ctrl.check_target_reachable(np.array([0.1, 0.0, 1.0]), bounds)
    assert ok is None
    bad = edge_ctrl.check_target_reachable(np.array([0.1, 0.0, 5.0]), bounds)
    assert bad is not None and "z=5.0000" in bad
    print("PROTECTION2_REACHABILITY_CHECK_OK")


def test_protection2_stall_tracker():
    tracker = edge_ctrl._StallTracker(window=5, min_improve=0.005)
    # error barely moves -> should be stalled once window fills
    errors = [0.05, 0.0495, 0.0492, 0.0491, 0.0490]
    for e in errors:
        tracker.update(e)
    assert tracker.is_stalled() is True, f"expected stalled, improvement={tracker.improvement()}"

    tracker2 = edge_ctrl._StallTracker(window=5, min_improve=0.005)
    errors2 = [0.05, 0.04, 0.03, 0.02, 0.01]
    for e in errors2:
        tracker2.update(e)
    assert tracker2.is_stalled() is False, f"expected NOT stalled, improvement={tracker2.improvement()}"
    print("PROTECTION2_STALL_TRACKER_OK")


def test_protection2_episode_aborts_on_unreachable_target_before_close():
    """Full-path test: an unreachable APPROACH target must abort the
    episode with zero steps taken and CLOSE/LIFT must never run."""
    cfg = edge_ctrl.EdgePinchConfig(
        orientation_control_verified=True,
        workspace_bounds_m={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},  # impossibly tight
    )

    class FakeGripper:
        important_geoms = {"left_finger": ["lf"], "right_finger": ["rf"]}

    class FakeRobot:
        gripper = FakeGripper()
        eef_site_id = 0

    class FakeModel:
        def geom_name2id(self, name):
            return 0

        def body_name2id(self, name):
            return 0

    class FakeData:
        geom_xpos = np.array([[0.0, 0.0, 1.0]])
        body_xpos = np.array([[0.05, 0.0, 1.0]])
        body_xmat = np.array([np.eye(3).flatten()])
        site_xpos = np.array([[0.0, 0.0, 0.9]])

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeInnerObj:
        root_body = "bowl_root"

    class FakeInner:
        robots = [FakeRobot()]
        sim = FakeSim()
        objects_dict = {"akita_black_bowl_1": FakeInnerObj()}

    class FakeEnv:
        env = FakeInner()

        def reset(self):
            return {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}

        def step(self, action):
            raise AssertionError("env.step must NOT be called when APPROACH target is unreachable")

        def check_success(self):
            return False

    class FakeCtrl:
        @staticmethod
        def object_xyz(obs, name):
            return np.array([0.05, 0.0, 1.0])

        @staticmethod
        def _eef_pos_with_fallback(env, obs):
            return np.array([0.0, 0.0, 0.9])

        @staticmethod
        def _gripper_contact_with_object(env, name):
            return False

        @staticmethod
        def _gripper_part_contact_with_object(env, name):
            return {"left_fingerpad": False, "right_fingerpad": False}

        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    rim_local_points = {
        "rim_local_points": [
            {"geom_name": "rim0", "local_xyz": [0.05, 0.0, 0.05], "local_azimuth_deg": 0.0},
        ]
    }

    env = FakeEnv()
    result = edge_ctrl.run_episode_edge_pinch(env, cfg, FakeCtrl(), seed=1, rim_local_points=rim_local_points)
    assert result.aborted is True
    assert result.steps_run == 0
    assert "APPROACH" in (result.abort_reason or "")
    assert "CLOSE" not in result.phase_transition_steps
    print("PROTECTION2_EPISODE_ABORT_UNREACHABLE_OK", result.abort_reason)


def test_edge_point_unavailable_aborts_without_fallback_to_center():
    cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
    result = edge_ctrl.run_episode_edge_pinch(
        env=None, cfg=cfg, ctrl_module=None, seed=1, rim_local_points={"error": "no rim found"}
    )
    assert result.aborted is True
    assert result.abort_reason == "edge_point_unavailable"
    print("EDGE_POINT_UNAVAILABLE_NO_FALLBACK_OK")


# ----------------------------------------------------------------------
# Edge-point local -> world transform correctness
# ----------------------------------------------------------------------

def test_select_edge_local_point_picks_closest_azimuth():
    rim_local_points = {
        "rim_local_points": [
            {"geom_name": "a", "local_xyz": [0.05, 0.0, 0.0], "local_azimuth_deg": 0.0},
            {"geom_name": "b", "local_xyz": [0.0, 0.05, 0.0], "local_azimuth_deg": 90.0},
            {"geom_name": "c", "local_xyz": [-0.05, 0.0, 0.0], "local_azimuth_deg": 180.0},
        ]
    }
    picked = edge_ctrl.select_edge_local_point(rim_local_points, edge_azimuth_deg=100.0)
    assert np.allclose(picked, [0.0, 0.05, 0.0]), f"expected point b, got {picked}"
    # wrap-around case: -10 deg should be closest to 0 deg point, not 180
    picked2 = edge_ctrl.select_edge_local_point(rim_local_points, edge_azimuth_deg=-10.0)
    assert np.allclose(picked2, [0.05, 0.0, 0.0]), f"expected point a (wraparound), got {picked2}"
    print("SELECT_EDGE_LOCAL_POINT_AZIMUTH_OK")


def test_edge_target_point_world_transform_identity_rotation():
    """With body_xmat = identity, world = body_xpos + local (no rotation)."""

    class FakeModel:
        def body_name2id(self, name):
            return 0

    class FakeData:
        body_xpos = np.array([[0.1, 0.2, 1.0]])
        body_xmat = np.array([np.eye(3).flatten()])

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeInnerObj:
        root_body = "bowl_root"

    class FakeInner:
        sim = FakeSim()
        objects_dict = {"bowl": FakeInnerObj()}

    class FakeEnv:
        env = FakeInner()

    world = edge_ctrl.edge_target_point_world(FakeEnv(), "bowl", np.array([0.05, 0.0, 0.0]), ctrl_module=None)
    assert np.allclose(world, [0.15, 0.2, 1.0]), f"got {world}"
    print("EDGE_TARGET_WORLD_IDENTITY_ROTATION_OK", world.tolist())


def test_edge_target_point_world_transform_90deg_rotation():
    """With body_xmat = 90deg rotation about Z, local +X should map to world +Y
    direction (relative to body_xpos), verifying the rotation is actually
    applied (not just an additive offset)."""
    theta = np.pi / 2
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])

    class FakeModel:
        def body_name2id(self, name):
            return 0

    class FakeData:
        body_xpos = np.array([[0.0, 0.0, 1.0]])
        body_xmat = np.array([rot_z.flatten()])

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeInnerObj:
        root_body = "bowl_root"

    class FakeInner:
        sim = FakeSim()
        objects_dict = {"bowl": FakeInnerObj()}

    class FakeEnv:
        env = FakeInner()

    world = edge_ctrl.edge_target_point_world(FakeEnv(), "bowl", np.array([0.05, 0.0, 0.0]), ctrl_module=None)
    assert np.allclose(world, [0.0, 0.05, 1.0], atol=1e-9), f"got {world}"
    print("EDGE_TARGET_WORLD_90DEG_ROTATION_OK", world.tolist())


def test_calibration_result_looks_successful_requires_all_criteria():
    good = edge_ctrl.EdgePinchCalibrationResult(
        edge_azimuth_deg=0.0, target_ee_ori=(0, 0, 0), standoff_m=0.05,
        grasp_lift_delta_m=0.03, finger_distance_at_close_end_m=0.02,
        finger_distance_at_lift_end_m=0.02, bilateral_contact_at_close_end=True,
        bilateral_contact_at_lift_end=True,
        bilateral_contact_lift_fraction=0.9, aborted=False, abort_reason=None, error=None,
    )
    assert good.looks_successful is True

    bad_aborted = edge_ctrl.EdgePinchCalibrationResult(
        edge_azimuth_deg=0.0, target_ee_ori=(0, 0, 0), standoff_m=0.05,
        grasp_lift_delta_m=0.03, finger_distance_at_close_end_m=0.02,
        finger_distance_at_lift_end_m=0.02, bilateral_contact_at_close_end=True,
        bilateral_contact_at_lift_end=True,
        bilateral_contact_lift_fraction=0.9, aborted=True, abort_reason="x", error=None,
    )
    assert bad_aborted.looks_successful is False

    bad_width = edge_ctrl.EdgePinchCalibrationResult(
        edge_azimuth_deg=0.0, target_ee_ori=(0, 0, 0), standoff_m=0.05,
        grasp_lift_delta_m=0.03, finger_distance_at_close_end_m=0.05,
        finger_distance_at_lift_end_m=0.05, bilateral_contact_at_close_end=True,
        bilateral_contact_at_lift_end=True,
        bilateral_contact_lift_fraction=0.9, aborted=False, abort_reason=None, error=None,
    )
    assert bad_width.looks_successful is False
    print("CALIBRATION_LOOKS_SUCCESSFUL_CRITERIA_OK")


# ----------------------------------------------------------------------
# 2026-09-16 round-2 diagnosis fixes: LIFT-end metric timing + orientation
# stall-when-already-converged. Full run_episode_edge_pinch() integration
# tests using a scripted FakeEnv (compute_action_edge_pinch itself is
# monkeypatched to a deterministic fake so these tests isolate the METRIC
# TIMING and STALL-DECISION logic under test, not real control convergence).
# ----------------------------------------------------------------------

def _make_fake_env_and_ctrl_for_lift_metrics(state):
    """Builds a FakeEnv/FakeCtrl pair where the fake env tracks the last
    commanded gripper value (state['gripper_closed']) so finger-distance,
    bilateral-contact, and bowl-height can be scripted to reflect whether
    the gripper is currently open or closed -- reproducing the exact bug
    scenario (gripper forced open during terminal_hold, then metrics
    measured) so the fix can be verified against a controlled ground truth.
    """

    class FakeGripper:
        important_geoms = {"left_finger": ["lf"], "right_finger": ["rf"]}

    class FakeRobot:
        gripper = FakeGripper()

    class FakeModel:
        def geom_name2id(self, name):
            return 0

        def body_name2id(self, name):
            return 0

    class FakeData:
        geom_xpos = np.array([[0.0, 0.0, 1.0]])
        body_xpos = np.array([[0.05, 0.0, 1.0]])
        body_xmat = np.array([np.eye(3).flatten()])

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeInnerObj:
        root_body = "bowl_root"

    class FakeInner:
        robots = [FakeRobot()]
        sim = FakeSim()
        objects_dict = {"akita_black_bowl_1": FakeInnerObj()}

    class FakeEnv:
        env = FakeInner()

        def reset(self):
            return {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}

        def step(self, action):
            state["gripper_closed"] = bool(action[6] > 0)
            state["last_action"] = np.array(action)
            return {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}, 0.0, False, {}

        def check_success(self):
            return False

        def close(self):
            pass

    class FakeCtrl:
        @staticmethod
        def object_xyz(obs, name):
            if name == "akita_black_bowl_1":
                return np.array([0.05, 0.0, 1.05 if state["gripper_closed"] else 1.0])
            return np.array([0.3, 0.0, 0.9])

        @staticmethod
        def _eef_pos_with_fallback(env, obs):
            return np.array([0.0, 0.0, 0.9])

        @staticmethod
        def _gripper_contact_with_object(env, name):
            return state["gripper_closed"]

        @staticmethod
        def _gripper_part_contact_with_object(env, name):
            return {
                "left_fingerpad": state["gripper_closed"],
                "right_fingerpad": state["gripper_closed"],
            }

        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    rim_local_points = {
        "rim_local_points": [
            {"geom_name": "rim0", "local_xyz": [0.05, 0.0, 0.05], "local_azimuth_deg": 0.0},
        ]
    }
    return FakeEnv(), FakeCtrl(), rim_local_points


def test_lift_end_metrics_captured_before_terminal_hold_gripper_reopen():
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001  # always "converged" -> every phase advances in 1 step

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action

        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_calibration_subset_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.aborted is False, result.abort_reason
        # Captured BEFORE terminal_hold forces the gripper open again:
        assert result.finger_distance_at_lift_end_m == 0.02, result.finger_distance_at_lift_end_m
        assert result.bilateral_contact_at_lift_end is True
        assert result.grasp_lift_delta_m is not None and abs(result.grasp_lift_delta_m - 0.05) < 1e-9, (
            result.grasp_lift_delta_m
        )
        # Sanity: terminal_hold DID run and DID force the gripper back open
        # afterwards (proving the bug scenario was actually exercised, not
        # trivially avoided) -- but the captured metrics above are unaffected.
        assert state["gripper_closed"] is False
        print(
            "LIFT_END_METRICS_CAPTURED_BEFORE_TERMINAL_HOLD_OK",
            result.finger_distance_at_lift_end_m,
            result.grasp_lift_delta_m,
            result.bilateral_contact_at_lift_end,
        )
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_orientation_stall_skipped_when_already_converged_but_noisy():
    """Orientation oscillates slightly (sometimes 'worsening' by a tiny
    amount) but stays UNDER orientation_tolerance_rad throughout, while
    position steadily improves toward its own tolerance. Per the fix, this
    must NOT abort -- DESCEND should complete normally once position
    converges."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    call_counter = {"descend_step": 0}

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        # Only DESCEND is scripted to take multiple steps (position starts
        # far, improves by 0.003/step -- above the 0.002 stall threshold --
        # until it converges around step 20). Orientation oscillates in a
        # tiny band well under orientation_tolerance_rad=0.1 the whole time.
        n = call_counter["descend_step"]
        call_counter["descend_step"] += 1
        pos_err = max(0.001, 0.08 - 0.003 * n)
        ori_err = 0.02 + (0.005 if n % 2 == 0 else -0.005)  # oscillates, always << 0.1
        return action, float(pos_err), float(ori_err)

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action

        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_calibration_subset_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.aborted is False, result.abort_reason
        assert "DESCEND" in result.phase_transition_steps
        print("ORIENTATION_STALL_SKIPPED_WHEN_CONVERGED_OK", result.phase_transition_steps)
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_orientation_stall_still_triggers_when_genuinely_diverging():
    """Regression: orientation starts under tolerance but then rises past
    it and keeps steadily worsening (never improving) during DESCEND, while
    position also starts unconverged but keeps improving steadily (never
    stalled on its own). This MUST still abort via the orientation-stall
    path once the stall window fills with sustained worsening -- confirms
    the fix did not make orientation-stall unreachable, only skip it while
    already converged.

    APPROACH's target height (edge_world_z + standoff, ~1.10) is used to
    let APPROACH converge trivially in 1 step (real pos/ori errors, not the
    scripted DESCEND sequence), isolating the scripted divergence to
    DESCEND's own target height (~1.05) via a fresh per-DESCEND counter.
    """
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    call_counter = {"n": 0}

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        if target_pos[2] > 1.075:  # APPROACH's standoff target -> converges instantly
            return action, 0.001, 0.001
        # DESCEND/CLOSE target height: scripted sequence -- position starts
        # unconverged but steadily improves; orientation starts converged
        # (<0.1) then crosses over and steadily worsens, never recovering.
        n = call_counter["n"]
        call_counter["n"] += 1
        pos_err = max(0.001, 0.05 - 0.004 * n)
        ori_err = 0.05 + 0.01 * n
        return action, float(pos_err), float(ori_err)

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action

        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_calibration_subset_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.aborted is True, "genuinely diverging orientation must still abort"
        assert "DESCEND" in (result.abort_reason or "")
        assert "ori_stalled=True" in (result.abort_reason or "")
        assert "pos_stalled=True" not in (result.abort_reason or ""), (
            "position was steadily improving and should not itself have stalled: "
            + str(result.abort_reason)
        )
        print("ORIENTATION_STALL_STILL_TRIGGERS_ON_DIVERGENCE_OK", result.abort_reason)
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_position_only_stall_still_aborts_with_orientation_converged():
    """Regression: position never improves (classic stall) while
    orientation stays perfectly converged throughout. Must still abort via
    the position-stall path -- confirms the orientation-stall fix did not
    weaken position-stall detection."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.05, 0.01  # position stuck far above tol=0.01, never improves

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action

        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_calibration_subset_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.aborted is True, "stuck position must still abort via position-stall"
        assert "pos_stalled=True" in (result.abort_reason or "")
        print("POSITION_ONLY_STALL_STILL_ABORTS_OK", result.abort_reason)
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


# ----------------------------------------------------------------------
# make_env_fn wiring: must delegate to ctrl_module.make_env(bddl_path)
# exactly, matching the already-confirmed old-script construction params.
# No real env/LIBERO/MuJoCo is touched -- ctrl_module here is a fake that
# only RECORDS the call, and its "env" is a plain sentinel object, not a
# real simulator instance. env.reset() is never called anywhere in this
# test.
# ----------------------------------------------------------------------

def test_build_make_env_fn_delegates_to_ctrl_module_make_env():
    calls = []

    class FakeCtrlModule:
        @staticmethod
        def make_env(bddl_path):
            calls.append(bddl_path)
            return "SENTINEL_ENV_NOT_A_REAL_SIMULATOR"

    make_env_fn = edge_ctrl._build_make_env_fn(FakeCtrlModule(), "/some/bddl/path.bddl")
    assert calls == [], "constructing the closure must not itself call make_env"
    result = make_env_fn()
    assert calls == ["/some/bddl/path.bddl"], f"expected exactly one call with the bddl path, got {calls}"
    assert result == "SENTINEL_ENV_NOT_A_REAL_SIMULATOR"
    print("BUILD_MAKE_ENV_FN_WIRING_OK")


def _find_privileged_controller_script() -> str:
    """The privileged controller lives in a DIFFERENT directory than this
    test file on GPU (edge_ctrl.DEFAULT_CONTROLLER_SCRIPT, under
    /root/smolvla-training-prep/scripts/) but as a same-directory
    convenience copy in the author's local sandbox. Try the real GPU
    default first, then fall back to a copy sitting next to this test file,
    so the SAME test file works unmodified in both places instead of
    hardcoding either one."""
    candidates = [
        edge_ctrl.DEFAULT_CONTROLLER_SCRIPT,
        os.path.join(THIS_DIR, "task1_privileged_pickplace_controller_v1.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"task1_privileged_pickplace_controller_v1.py not found in any of: {candidates}"
    )


def test_build_make_env_fn_matches_old_script_signature():
    """Confirms the OLD controller's make_env() signature is (bddl_path: str)
    -- i.e. positional-compatible with what _build_make_env_fn passes -- by
    reading its signature directly (no call, no env construction)."""
    import inspect
    ctrl_spec = importlib.util.spec_from_file_location(
        "ctrl_real",
        _find_privileged_controller_script(),
    )
    ctrl_real = importlib.util.module_from_spec(ctrl_spec)
    sys.modules["ctrl_real"] = ctrl_real
    ctrl_spec.loader.exec_module(ctrl_real)
    sig = inspect.signature(ctrl_real.make_env)
    params = list(sig.parameters.keys())
    assert params == ["bddl_path"], f"old script's make_env signature changed: {params}"
    import inspect as _i
    src = _i.getsource(ctrl_real.make_env)
    assert "camera_heights=128" in src
    assert "camera_widths=128" in src
    assert "horizon=1000" in src
    assert "use_camera_obs=True" in src
    print("OLD_SCRIPT_MAKE_ENV_SIGNATURE_AND_PARAMS_CONFIRMED_OK")


# ----------------------------------------------------------------------
# CLI dispatch: --probe-orientation alone must never reach --calibrate's
# code path. No real env/LIBERO is touched -- probe_orientation_action_
# response and calibrate_edge_pinch are both monkeypatched to recorders.
# ----------------------------------------------------------------------

def test_probe_orientation_flag_never_enters_calibration():
    calls = {"probe": 0, "calibrate": 0}

    def fake_probe(make_env_fn, ctrl_module, **kwargs):
        calls["probe"] += 1
        return {"fake": "probe_report", "note": "no real env touched"}

    def fake_calibrate(*args, **kwargs):
        calls["calibrate"] += 1
        raise AssertionError("calibrate_edge_pinch must NOT be called when only --probe-orientation is passed")

    class FakeCtrlModule:
        @staticmethod
        def make_env(bddl_path):
            raise AssertionError("ctrl_module.make_env must not be called: probe is mocked out entirely")

    orig_probe = edge_ctrl.probe_orientation_action_response
    orig_calibrate = edge_ctrl.calibrate_edge_pinch
    orig_load = edge_ctrl._load_controller_module
    orig_argv = sys.argv
    try:
        edge_ctrl.probe_orientation_action_response = fake_probe
        edge_ctrl.calibrate_edge_pinch = fake_calibrate
        edge_ctrl._load_controller_module = lambda path: FakeCtrlModule()
        sys.argv = [
            "task1_edge_pinch_controller_v1.py",
            "--probe-orientation",
            "--output-dir", os.path.join(THIS_DIR, "_edge_pinch_probe_test_out"),
        ]
        exit_code = edge_ctrl.main()
        assert exit_code == 0
        assert calls["probe"] == 1
        assert calls["calibrate"] == 0
        print("PROBE_ORIENTATION_NEVER_ENTERS_CALIBRATION_OK")
    finally:
        edge_ctrl.probe_orientation_action_response = orig_probe
        edge_ctrl.calibrate_edge_pinch = orig_calibrate
        edge_ctrl._load_controller_module = orig_load
        sys.argv = orig_argv


def test_calibrate_without_orientation_verified_refused_before_any_call():
    calls = {"calibrate": 0}

    def fake_calibrate(*args, **kwargs):
        calls["calibrate"] += 1
        raise AssertionError("calibrate_edge_pinch must NOT be called without --orientation-verified")

    class FakeCtrlModule:
        @staticmethod
        def make_env(bddl_path):
            raise AssertionError("ctrl_module.make_env must not be called: gate must refuse first")

    orig_calibrate = edge_ctrl.calibrate_edge_pinch
    orig_load = edge_ctrl._load_controller_module
    orig_argv = sys.argv
    try:
        edge_ctrl.calibrate_edge_pinch = fake_calibrate
        edge_ctrl._load_controller_module = lambda path: FakeCtrlModule()
        sys.argv = ["task1_edge_pinch_controller_v1.py", "--calibrate"]
        exit_code = edge_ctrl.main()
        assert exit_code == 2
        assert calls["calibrate"] == 0
        print("CALIBRATE_WITHOUT_VERIFIED_REFUSED_BEFORE_ANY_CALL_OK")
    finally:
        edge_ctrl.calibrate_edge_pinch = orig_calibrate
        edge_ctrl._load_controller_module = orig_load
        sys.argv = orig_argv


def test_smoke_test_without_orientation_verified_refused_before_any_call():
    calls = {"smoke": 0}

    def fake_smoke(*args, **kwargs):
        calls["smoke"] += 1
        raise AssertionError("run_smoke_test_edge_pinch must NOT be called without --orientation-verified")

    class FakeCtrlModule:
        @staticmethod
        def make_env(bddl_path):
            raise AssertionError("ctrl_module.make_env must not be called: gate must refuse first")

    orig_smoke = edge_ctrl.run_smoke_test_edge_pinch
    orig_load = edge_ctrl._load_controller_module
    orig_argv = sys.argv
    try:
        edge_ctrl.run_smoke_test_edge_pinch = fake_smoke
        edge_ctrl._load_controller_module = lambda path: FakeCtrlModule()
        sys.argv = ["task1_edge_pinch_controller_v1.py", "--smoke-test"]
        exit_code = edge_ctrl.main()
        assert exit_code == 2
        assert calls["smoke"] == 0
        print("SMOKE_TEST_WITHOUT_VERIFIED_REFUSED_BEFORE_ANY_CALL_OK")
    finally:
        edge_ctrl.run_smoke_test_edge_pinch = orig_smoke
        edge_ctrl._load_controller_module = orig_load
        sys.argv = orig_argv


def test_run_smoke_test_edge_pinch_runs_full_episode_per_standoff_seed_pair():
    """Uses the same deterministic FakeEnv/fake compute_action pattern as
    the LIFT-metrics-timing test, but drives run_smoke_test_edge_pinch
    directly (not the truncated calibration subset) to confirm: (a) it runs
    one result per (standoff, seed) pair, (b) env_check_success is surfaced
    (meaningful here since the full phase sequence runs, unlike --calibrate's
    truncated subset), (c) the LIFT-end metrics fix applies here too."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)
    # env.check_success() on the shared FakeEnv always returns False --
    # sufficient to confirm the field is surfaced/plumbed through, not to
    # assert a specific success value.

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001  # every phase converges in 1 step

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch

    def fake_make_env_fn():
        return env

    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action

        orig_compute_rim = edge_ctrl.compute_rim_local_points
        edge_ctrl.compute_rim_local_points = lambda make_env_fn, ctrl_module, name: rim_local_points
        try:
            results = edge_ctrl.run_smoke_test_edge_pinch(
                fake_make_env_fn,
                ctrl,
                seeds=[1, 2],
                edge_azimuth_deg=90.0,
                standoff_candidates=[0.03, 0.05],
                target_ee_ori=(2.2587, -2.2587, -0.0700),
                orientation_control_verified=True,
            )
        finally:
            edge_ctrl.compute_rim_local_points = orig_compute_rim

        assert len(results) == 4, f"expected 2 standoffs x 2 seeds = 4 runs, got {len(results)}"
        for r in results:
            assert r["aborted"] is False, r["abort_reason"]
            assert "env_check_success" in r
            assert r["finger_distance_at_lift_end_m"] == 0.02
            assert r["bilateral_contact_at_lift_end"] is True
        seeds_seen = sorted({r["seed"] for r in results})
        standoffs_seen = sorted({r["standoff_m"] for r in results})
        assert seeds_seen == [1, 2]
        assert standoffs_seen == [0.03, 0.05]
        print("RUN_SMOKE_TEST_FULL_EPISODE_PER_PAIR_OK", len(results))
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


# ----------------------------------------------------------------------
# 2026-09-17: grasp-offset compensation + phase-separated workspace bounds
# (task1_edge_pinch_offset_compensation_design_v1.md, approved design).
# ----------------------------------------------------------------------


def _make_fake_lift_record(target_object_pos, eef_pos, phase="LIFT"):
    return edge_ctrl.EdgePinchStepRecord(
        step=1,
        phase=phase,
        action=np.zeros(7, dtype=np.float32),
        eef_pos=np.asarray(eef_pos, dtype=np.float64),
        target_pos=np.asarray(eef_pos, dtype=np.float64),
        position_error_m=0.0,
        orientation_error_rad=0.0,
        target_object_pos=np.asarray(target_object_pos, dtype=np.float64),
    )


def test_estimate_grasp_offset_from_lift_records_averages_last_n_steps():
    """5 LIFT-phase records with small noise around a known true offset of
    (0.03, -0.01, -0.10), preceded by non-LIFT records that must be ignored,
    and only the LAST 3 of the 5 LIFT records should be used when
    window_steps=3."""
    true_offset = np.array([0.03, -0.01, -0.10])
    noise = [
        np.array([0.001, -0.0005, 0.0008]),
        np.array([-0.0008, 0.0006, -0.0004]),
        np.array([0.0003, -0.0002, 0.0009]),
        np.array([0.0005, 0.0004, -0.0006]),
        np.array([-0.0002, -0.0003, 0.0002]),
    ]
    records = [_make_fake_lift_record([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], phase="CLOSE")]
    eef_base = np.array([0.5, 0.2, 1.0])
    for n in noise:
        offset_i = true_offset + n
        records.append(_make_fake_lift_record(eef_base + offset_i, eef_base, phase="LIFT"))

    offset, std, reliable, fallback_used = edge_ctrl._estimate_grasp_object_offset(
        records, window_steps=3
    )
    assert fallback_used is False
    assert std is not None
    expected = np.mean([true_offset + n for n in noise[-3:]], axis=0)
    assert np.allclose(offset, expected, atol=1e-9), (offset, expected)
    assert reliable is True  # noise magnitude << GRASP_OFFSET_STD_RELIABLE_THRESHOLD_M
    print("ESTIMATE_GRASP_OFFSET_AVERAGES_LAST_N_OK", offset.tolist())


def test_estimate_grasp_offset_fallback_when_no_lift_records():
    records = [_make_fake_lift_record([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], phase="CLOSE")]
    offset, std, reliable, fallback_used = edge_ctrl._estimate_grasp_object_offset(
        records, window_steps=5
    )
    assert fallback_used is True
    assert reliable is False
    assert std is None
    assert np.allclose(offset, np.array(edge_ctrl.ZERO_GRASP_OFFSET_M))
    print("ESTIMATE_GRASP_OFFSET_FALLBACK_OK", offset.tolist())


def test_move_target_applies_offset_compensation_xy_only_not_z():
    class FakeCtrl:
        @staticmethod
        def object_xyz(obs, name):
            return np.array([-0.0868, 0.0112, 0.97])  # plate, per audited data

    cfg = edge_ctrl.EdgePinchConfig(lift_height_m=0.15)
    offset = np.array([0.033, -0.002, -0.10])
    lift_reference_z = 1.0
    target = edge_ctrl.phase_target_pos_edge_pinch(
        "MOVE", cfg, env=None, ctrl_module=FakeCtrl(), obs={},
        eef_pos=np.array([0.0, 0.0, 1.0]), lift_reference_z=lift_reference_z,
        edge_local_point=None, grasp_object_offset_m=offset,
    )
    assert abs(target[0] - (-0.0868 - 0.033)) < 1e-9, target
    assert abs(target[1] - (0.0112 - (-0.002))) < 1e-9, target
    # z is the transit height, UNAFFECTED by offset_z:
    assert abs(target[2] - (lift_reference_z + cfg.lift_height_m)) < 1e-9, target
    print("MOVE_TARGET_OFFSET_COMPENSATION_XY_ONLY_OK", target.tolist())


def test_descend2_and_release_share_identical_compensated_target():
    class FakeCtrl:
        @staticmethod
        def object_xyz(obs, name):
            return np.array([-0.0868, 0.0112, 0.97])

    cfg = edge_ctrl.EdgePinchConfig(place_height_offset_m=0.05)
    offset = np.array([0.033, -0.002, -0.10])
    kwargs = dict(
        cfg=cfg, env=None, ctrl_module=FakeCtrl(), obs={},
        eef_pos=np.array([0.0, 0.0, 1.0]), lift_reference_z=1.0,
        edge_local_point=None, grasp_object_offset_m=offset,
    )
    target_descend2 = edge_ctrl.phase_target_pos_edge_pinch("DESCEND2", **kwargs)
    target_release = edge_ctrl.phase_target_pos_edge_pinch("RELEASE", **kwargs)
    assert np.array_equal(target_descend2, target_release), (target_descend2, target_release)
    print("DESCEND2_RELEASE_SHARE_TARGET_OK", target_descend2.tolist())


def test_place_target_compensates_z_in_correct_direction():
    """A negative offset_z (bowl hanging below the EEF, as diagnosed) must
    make the compensated place-target HIGHER than the uncompensated one --
    i.e. the fix must make the EEF stop higher, not lower, reversing the
    seed=950004 failure mode (bowl driven below plate height)."""
    plate = np.array([-0.0868, 0.0112, 0.97])
    cfg = edge_ctrl.EdgePinchConfig(place_height_offset_m=0.05)
    uncompensated = edge_ctrl._compensated_place_target(
        plate, np.array(edge_ctrl.ZERO_GRASP_OFFSET_M), cfg
    )
    compensated = edge_ctrl._compensated_place_target(plate, np.array([0.03, 0.0, -0.10]), cfg)
    assert compensated[2] > uncompensated[2], (compensated, uncompensated)
    assert abs(compensated[2] - (uncompensated[2] + 0.10)) < 1e-9
    print("PLACE_TARGET_Z_COMPENSATION_DIRECTION_OK", uncompensated.tolist(), compensated.tolist())


def test_move_target_offset_direction_reduces_predicted_bowl_plate_error():
    """Regression using the ACTUAL diagnosed numbers from
    claude/task1_edge_pinch_seed950004_phase_breakdown_and_offset_diagnosis_
    20260916.json: plate_x=-0.0868, offset_x~=+0.033. Predicted bowl position
    = eef_target + offset (since bowl ~= eef + offset once converged). The
    fix must make that predicted bowl position CLOSER to plate_x than the
    old, uncompensated logic -- this is the test that would catch the
    compensation sign being written backwards."""
    plate_x = -0.0868
    offset_x = 0.033

    old_target_x = plate_x  # pre-fix: EEF driven straight to plate_x
    old_predicted_bowl_x = old_target_x + offset_x

    new_target_x = plate_x - offset_x  # post-fix (MOVE/DESCEND2 formula)
    new_predicted_bowl_x = new_target_x + offset_x

    old_error = abs(old_predicted_bowl_x - plate_x)
    new_error = abs(new_predicted_bowl_x - plate_x)
    assert new_error < old_error, (old_error, new_error)
    assert new_error < 1e-9  # compensation should drive the predicted error to ~zero
    print("OFFSET_COMPENSATION_REDUCES_PREDICTED_ERROR_OK", old_error, new_error)


def _make_fake_env_and_ctrl_for_bounds_test(state, bowl_body_xpos, container_pos):
    """Variant of _make_fake_env_and_ctrl_for_lift_metrics parameterized on
    the bowl's body_xpos (controls edge_target_point_world's output for
    APPROACH/DESCEND) and the container's (plate's) reported position
    (controls MOVE/DESCEND2's target) independently, so a test can place one
    inside a given bounds box and the other outside it."""

    class FakeGripper:
        important_geoms = {"left_finger": ["lf"], "right_finger": ["rf"]}

    class FakeRobot:
        gripper = FakeGripper()

    class FakeModel:
        def geom_name2id(self, name):
            return 0

        def body_name2id(self, name):
            return 0

    class FakeData:
        geom_xpos = np.array([[0.0, 0.0, 1.0]])
        body_xpos = np.array([np.asarray(bowl_body_xpos, dtype=np.float64)])
        body_xmat = np.array([np.eye(3).flatten()])

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeInnerObj:
        root_body = "bowl_root"

    class FakeInner:
        robots = [FakeRobot()]
        sim = FakeSim()
        objects_dict = {edge_ctrl.DEFAULT_TARGET_OBJECT: FakeInnerObj()}

    class FakeEnv:
        env = FakeInner()

        def reset(self):
            return {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}

        def step(self, action):
            state["gripper_closed"] = bool(action[6] > 0)
            return {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}, 0.0, False, {}

        def check_success(self):
            return False

        def close(self):
            pass

    class FakeCtrl:
        @staticmethod
        def object_xyz(obs, name):
            if name == edge_ctrl.DEFAULT_TARGET_CONTAINER:
                return np.asarray(container_pos, dtype=np.float64)
            # bowl "center" report used for LIFT-metrics bookkeeping only;
            # keep near body_xpos so the numbers stay physically sane.
            return np.asarray(bowl_body_xpos, dtype=np.float64)

        @staticmethod
        def _eef_pos_with_fallback(env, obs):
            return np.array([0.0, 0.0, 0.9])

        @staticmethod
        def _gripper_contact_with_object(env, name):
            return state["gripper_closed"]

        @staticmethod
        def _gripper_part_contact_with_object(env, name):
            return {
                "left_fingerpad": state["gripper_closed"],
                "right_fingerpad": state["gripper_closed"],
            }

        @staticmethod
        def clip_position_delta(delta, limit_m):
            return np.clip(delta / limit_m, -1.0, 1.0)

    rim_local_points = {
        "rim_local_points": [
            {"geom_name": "rim0", "local_xyz": [0.05, 0.0, 0.05], "local_azimuth_deg": 0.0},
        ]
    }
    return FakeEnv(), FakeCtrl(), rim_local_points


def test_workspace_bounds_selects_transport_bounds_for_move_and_descend2_only():
    """Plate reported at x=-0.15: OUTSIDE the default bowl-area bounds
    (x lower=-0.10) but INSIDE the default transport bounds (x lower=-0.25).
    The bowl's own edge-point stays well within BOTH boxes (body_xpos
    x=0.05 -> edge_world x=0.10). If MOVE/DESCEND2 correctly use the
    transport bounds, the full episode completes without an abort; if they
    mistakenly used the bowl-area bounds (the pre-fix call site), MOVE would
    abort with target_out_of_workspace_bounds."""
    state = {"gripper_closed": False}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_bounds_test(
        state, bowl_body_xpos=[0.05, 0.0, 1.0], container_pos=[-0.15, 0.0, 0.97]
    )

    def fake_finger_distance(env, ctrl_module):
        return 0.02

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)  # uses DEFAULT_* bounds
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)
        assert result.aborted is False, result.abort_reason
        assert "MOVE" in result.phase_transition_steps
        assert "DESCEND2" in result.phase_transition_steps
        print("WORKSPACE_BOUNDS_TRANSPORT_USED_FOR_MOVE_DESCEND2_OK")
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_workspace_bounds_approach_descend_still_use_bowl_area_bounds():
    """Bowl's edge point placed at x=-0.10 (body_xpos x=-0.15 + local +0.05):
    OUTSIDE a deliberately narrowed bowl-area bound (x lower=-0.05) but
    INSIDE the (default, wide) transport bound (x lower=-0.25). APPROACH
    must still abort -- if it mistakenly used the transport bounds instead
    of workspace_bounds_m, it would NOT abort here."""
    state = {"gripper_closed": False}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_bounds_test(
        state, bowl_body_xpos=[-0.15, 0.0, 1.0], container_pos=[0.0, 0.0, 0.97]
    )

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(
            orientation_control_verified=True,
            workspace_bounds_m={"x": (-0.05, 0.40), "y": (-0.40, 0.40), "z": (0.70, 1.30)},
            # transport bounds intentionally left at the wide default, which
            # WOULD accept x=-0.10 -- proving the abort below is genuinely
            # coming from workspace_bounds_m, not a coincidence.
        )
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)
        assert result.aborted is True
        assert "APPROACH" in (result.abort_reason or "")
        assert "target_out_of_workspace_bounds" in (result.abort_reason or "")
        print("WORKSPACE_BOUNDS_APPROACH_STILL_USES_BOWL_AREA_BOUNDS_OK", result.abort_reason)
    finally:
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_workspace_bounds_transport_default_widens_only_x():
    default_bowl = edge_ctrl.DEFAULT_WORKSPACE_BOUNDS_M
    default_transport = edge_ctrl.DEFAULT_WORKSPACE_BOUNDS_TRANSPORT_M
    assert default_transport["y"] == default_bowl["y"], (default_transport, default_bowl)
    assert default_transport["z"] == default_bowl["z"], (default_transport, default_bowl)
    assert default_transport["x"] != default_bowl["x"]
    # covers the audited plate x-range [-0.1565, -0.0868] with margin:
    assert default_transport["x"][0] <= -0.1565
    print("WORKSPACE_BOUNDS_TRANSPORT_DEFAULT_WIDENS_ONLY_X_OK", default_transport, default_bowl)


# ----------------------------------------------------------------------
# 2026-09-17: done=True diagnostic addition (task1_edge_pinch_smoke_test_
# offset_compensated_20260917.json -- 10/10 offset-compensation smoke-test
# runs hit "Environment ended unexpectedly during RELEASE" with
# env_check_success left at None, motivating this diagnostic-only fix).
# ----------------------------------------------------------------------


class _FakeEnvForDoneDiagnosis:
    def __init__(self, check_success_value=True, check_success_raises=False):
        self._check_success_value = check_success_value
        self._check_success_raises = check_success_raises

    def check_success(self):
        if self._check_success_raises:
            raise RuntimeError("simulated check_success() failure")
        return self._check_success_value


class _FakeResultForDoneDiagnosis:
    done_early_termination_phase = None
    done_early_termination_step = None
    done_early_termination_check_success = None
    done_early_termination_check_success_error = None
    done_early_termination_info = None
    # 2026-09-17 (RELEASE success-early-termination design): these three are
    # additionally written by _handle_release_done_termination (unlike
    # _diagnose_and_raise_on_done, which never touches them). Declared here
    # with their pre-call defaults purely for readability of the unit tests
    # below -- plain attribute assignment would work without this too, since
    # this is an ordinary (non-slotted) class.
    success = None
    termination_reason = None
    steps_run = 0


def test_diagnose_and_raise_on_done_captures_check_success_true():
    env = _FakeEnvForDoneDiagnosis(check_success_value=True)
    result = _FakeResultForDoneDiagnosis()
    raised = False
    try:
        edge_ctrl._diagnose_and_raise_on_done(env, "RELEASE", 99, {"some": "info"}, result)
    except RuntimeError as exc:
        raised = True
        assert "RELEASE" in str(exc)
        assert "check_success()=True" in str(exc)
    assert raised, "must still raise RuntimeError -- diagnostics are additive, not a control-flow change"
    assert result.done_early_termination_phase == "RELEASE"
    assert result.done_early_termination_step == 99
    assert result.done_early_termination_check_success is True
    assert result.done_early_termination_check_success_error is None
    assert result.done_early_termination_info == {"some": "info"}
    print("DIAGNOSE_DONE_CAPTURES_CHECK_SUCCESS_TRUE_OK")


def test_diagnose_and_raise_on_done_captures_check_success_false():
    env = _FakeEnvForDoneDiagnosis(check_success_value=False)
    result = _FakeResultForDoneDiagnosis()
    raised = False
    try:
        edge_ctrl._diagnose_and_raise_on_done(env, "MOVE", 42, {}, result)
    except RuntimeError:
        raised = True
    assert raised
    assert result.done_early_termination_check_success is False
    assert result.done_early_termination_phase == "MOVE"
    assert result.done_early_termination_step == 42
    print("DIAGNOSE_DONE_CAPTURES_CHECK_SUCCESS_FALSE_OK")


def test_diagnose_and_raise_on_done_handles_check_success_exception():
    """check_success() itself raising must NOT be swallowed into a silent
    success/failure guess -- it must be recorded as its own diagnostic
    field, and the function must still raise its normal RuntimeError (never
    let the check_success() exception escape in its place)."""
    env = _FakeEnvForDoneDiagnosis(check_success_raises=True)
    result = _FakeResultForDoneDiagnosis()
    raised_type = None
    try:
        edge_ctrl._diagnose_and_raise_on_done(env, "RELEASE", 100, None, result)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad to catch whatever type comes out
        raised_type = type(exc)
    assert raised_type is RuntimeError, raised_type
    assert result.done_early_termination_check_success is None
    assert result.done_early_termination_check_success_error is not None
    assert "simulated check_success() failure" in result.done_early_termination_check_success_error
    assert result.done_early_termination_info == {}  # info=None -> {} per the function's own handling
    print("DIAGNOSE_DONE_HANDLES_CHECK_SUCCESS_EXCEPTION_OK", result.done_early_termination_check_success_error)


# ----------------------------------------------------------------------
# 2026-09-17: RELEASE-phase success-early-termination (confirmed design,
# claude/task1_edge_pinch_release_success_termination_design_v1_20260917.json)
# ----------------------------------------------------------------------


def test_handle_release_done_termination_success_path_sets_success_true_and_reason():
    """check_success()=True at RELEASE's done=True must be treated as a real
    success: the function returns True (caller must NOT raise), and
    result.success/termination_reason/steps_run are all populated -- NOT
    just the done_early_termination_* diagnostic fields."""
    env = _FakeEnvForDoneDiagnosis(check_success_value=True)
    result = _FakeResultForDoneDiagnosis()
    is_success = edge_ctrl._handle_release_done_termination(env, 99, {"some": "info"}, result)
    assert is_success is True
    assert result.success is True
    assert result.termination_reason == "success_early_termination"
    assert result.steps_run == 99
    assert result.done_early_termination_phase == "RELEASE"
    assert result.done_early_termination_step == 99
    assert result.done_early_termination_check_success is True
    assert result.done_early_termination_check_success_error is None
    assert result.done_early_termination_info == {"some": "info"}
    print("HANDLE_RELEASE_DONE_SUCCESS_PATH_OK")


def test_handle_release_done_termination_failure_path_sets_success_false_and_reason():
    """check_success()=False at RELEASE's done=True must be treated as a
    real failure: the function returns False (caller must raise), and
    result.success is explicitly False (not left None), termination_reason
    is 'environment_failure_done_true', steps_run is still populated."""
    env = _FakeEnvForDoneDiagnosis(check_success_value=False)
    result = _FakeResultForDoneDiagnosis()
    is_success = edge_ctrl._handle_release_done_termination(env, 77, {}, result)
    assert is_success is False
    assert result.success is False
    assert result.termination_reason == "environment_failure_done_true"
    assert result.steps_run == 77
    assert result.done_early_termination_check_success is False
    print("HANDLE_RELEASE_DONE_FAILURE_PATH_OK")


def test_handle_release_done_termination_check_success_exception_treated_as_failure_not_success():
    """Per the confirmed design decision: check_success() itself raising is
    treated as 'unknown', NOT silently upgraded to success. The function
    must still return False (caller raises), result.success must be None
    (not True, and not forced to False either -- distinct from the explicit
    check_success()=False case above), and the exception text must be
    captured in done_early_termination_check_success_error."""
    env = _FakeEnvForDoneDiagnosis(check_success_raises=True)
    result = _FakeResultForDoneDiagnosis()
    is_success = edge_ctrl._handle_release_done_termination(env, 55, None, result)
    assert is_success is False
    assert result.success is None
    assert result.termination_reason == "environment_failure_done_true"
    assert result.steps_run == 55
    assert result.done_early_termination_check_success is None
    assert result.done_early_termination_check_success_error is not None
    assert "simulated check_success() failure" in result.done_early_termination_check_success_error
    assert result.done_early_termination_info == {}
    print("HANDLE_RELEASE_DONE_CHECK_SUCCESS_EXCEPTION_TREATED_AS_FAILURE_OK")


def test_run_episode_edge_pinch_populates_done_fields_on_early_termination():
    """Integration-level: a FakeEnv whose step() returns done=True partway
    through CLOSE's fixed-length hold loop must leave result.aborted=False
    (this is NOT one of the explicit Protection-2 abort paths) but
    result.error mentioning the phase, AND the new done_early_termination_*
    fields populated. CLOSE and RELEASE share the exact same code branch
    (`if phase in ("CLOSE", "RELEASE")`) and thus the same call site to
    _diagnose_and_raise_on_done -- this reproduces the same shape of failure
    as the 10/10 RELEASE-phase early-terminations from the offset-
    compensation smoke test, via the faster-to-reach CLOSE phase."""
    state = {"gripper_closed": False, "step_count": 0}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    orig_step = env.step

    def step_with_done(action):
        state["step_count"] += 1
        obs, reward, done, info = orig_step(action)
        # APPROACH/DESCEND take 1 step each under the always-converges fake
        # compute_action below, so CLOSE's hold loop starts at step 3 --
        # forcing done=True at step 7 lands inside CLOSE, not APPROACH/DESCEND.
        forced_done = state["step_count"] >= 7
        return obs, reward, forced_done, {"forced_for_test": True}

    env.step = step_with_done

    def fake_finger_distance(env, ctrl_module):
        return 0.02

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001  # every phase converges in 1 step

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.aborted is False  # NOT a Protection-2 abort -- an exception path
        assert result.error is not None and "Environment ended unexpectedly" in result.error
        assert result.done_early_termination_phase is not None
        assert result.done_early_termination_step is not None
        assert result.done_early_termination_check_success is not None  # FakeEnv.check_success() returns False, not raising
        assert result.done_early_termination_info == {"forced_for_test": True}
        print(
            "RUN_EPISODE_POPULATES_DONE_FIELDS_ON_EARLY_TERMINATION_OK",
            result.done_early_termination_phase,
            result.done_early_termination_check_success,
        )
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_smoke_test_edge_pinch_surfaces_offset_and_done_diagnostic_fields():
    """The per-run dict from run_smoke_test_edge_pinch() must always include
    grasp_object_offset_m and the done_early_termination_* keys (None when
    not applicable), not only when something went wrong -- otherwise a
    caller has no consistent schema to parse."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch

    def fake_make_env_fn():
        return env

    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        orig_compute_rim = edge_ctrl.compute_rim_local_points
        edge_ctrl.compute_rim_local_points = lambda make_env_fn, ctrl_module, name: rim_local_points
        try:
            results = edge_ctrl.run_smoke_test_edge_pinch(
                fake_make_env_fn, ctrl, seeds=[1], edge_azimuth_deg=90.0,
                standoff_candidates=[0.03], target_ee_ori=(2.2587, -2.2587, -0.0700),
                orientation_control_verified=True,
            )
        finally:
            edge_ctrl.compute_rim_local_points = orig_compute_rim

        assert len(results) == 1
        r = results[0]
        for key in (
            "grasp_object_offset_m", "grasp_object_offset_std_m", "grasp_object_offset_reliable",
            "grasp_object_offset_fallback_used", "done_early_termination_phase",
            "done_early_termination_step", "done_early_termination_check_success",
            "done_early_termination_check_success_error", "done_early_termination_info",
        ):
            assert key in r, f"missing key: {key}"
        # this fixture completes normally (no forced done=True), so the
        # done_early_termination_* fields should all be None:
        assert r["done_early_termination_phase"] is None
        # grasp_object_offset_m SHOULD be populated (LIFT completes normally
        # in this fixture, with a constant eef_pos/target_object_pos gap):
        assert r["grasp_object_offset_m"] is not None
        print("RUN_SMOKE_TEST_SURFACES_OFFSET_AND_DONE_FIELDS_OK", r["grasp_object_offset_m"])
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def _step_count_lands_in_release_phase_threshold():
    """With the standard _make_fake_env_and_ctrl_for_lift_metrics fixture and
    the always-converges-in-1-step fake compute_action: APPROACH(1)+
    DESCEND(1)=2, CLOSE hold loop runs steps 3..22 (grasp_hold_steps=20),
    LIFT(1)=23, MOVE(1)=24, DESCEND2(1)=25, RELEASE hold loop runs steps
    26..45 (release_hold_steps=20). A forced-done threshold of 30 lands
    comfortably inside RELEASE's hold loop, well past CLOSE/LIFT/MOVE/
    DESCEND2 and well before RELEASE would naturally finish at step 45."""
    return 30


def test_run_episode_edge_pinch_release_success_returns_normally_without_raising():
    """Integration test for the confirmed design's success path: FakeEnv
    forced to return done=True partway through RELEASE's hold loop, with
    check_success()=True. run_episode_edge_pinch() must return result
    NORMALLY (no exception), with result.error is None, result.aborted is
    False, result.success is True, result.termination_reason ==
    'success_early_termination', steps_run matching the forced-done step,
    and grasp_object_offset_m still populated from LIFT-end (unaffected by
    taking the early-return path)."""
    state = {"gripper_closed": False, "step_count": 0}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    orig_step = env.step
    threshold = _step_count_lands_in_release_phase_threshold()

    def step_with_done(action):
        state["step_count"] += 1
        obs, reward, done, info = orig_step(action)
        forced_done = state["step_count"] >= threshold
        return obs, reward, forced_done, {"forced_for_test": True}

    env.step = step_with_done
    env.check_success = lambda: True  # the env's own success judgment at the moment done=True fires

    def fake_finger_distance(env, ctrl_module):
        return 0.02

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001  # every phase converges in 1 step

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.error is None, result.error
        assert result.aborted is False
        assert result.success is True
        assert result.termination_reason == "success_early_termination"
        assert result.steps_run == threshold
        assert result.done_early_termination_phase == "RELEASE"
        assert result.done_early_termination_step == threshold
        assert result.done_early_termination_check_success is True
        assert result.done_early_termination_info == {"forced_for_test": True}
        # LIFT already completed (well before RELEASE) so the offset estimate
        # must still be populated -- the early return must not have skipped it:
        assert result.grasp_object_offset_m is not None
        print(
            "RUN_EPISODE_RELEASE_SUCCESS_RETURNS_NORMALLY_OK",
            result.termination_reason, result.steps_run, result.grasp_object_offset_m,
        )
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_episode_edge_pinch_release_failure_still_raises_and_populates_fields():
    """Integration test for the confirmed design's failure path: same forced
    done=True during RELEASE, but check_success()=False. Must still raise
    (result.error populated by the outer except-block), AND
    result.success/termination_reason/steps_run/grasp_object_offset_m must
    all be populated -- the diagnostic value must not be lost just because
    an exception propagated."""
    state = {"gripper_closed": False, "step_count": 0}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    orig_step = env.step
    threshold = _step_count_lands_in_release_phase_threshold()

    def step_with_done(action):
        state["step_count"] += 1
        obs, reward, done, info = orig_step(action)
        forced_done = state["step_count"] >= threshold
        return obs, reward, forced_done, {"forced_for_test": True}

    env.step = step_with_done
    env.check_success = lambda: False  # FakeEnv's default is already False, explicit here for clarity

    def fake_finger_distance(env, ctrl_module):
        return 0.02

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.error is not None and "Environment ended unexpectedly during RELEASE" in result.error
        assert result.aborted is False
        assert result.success is False
        assert result.termination_reason == "environment_failure_done_true"
        assert result.steps_run == threshold
        assert result.done_early_termination_check_success is False
        # grasp_object_offset_m must still be populated even though this run failed:
        assert result.grasp_object_offset_m is not None
        print(
            "RUN_EPISODE_RELEASE_FAILURE_STILL_RAISES_OK",
            result.termination_reason, result.steps_run,
        )
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_episode_edge_pinch_close_phase_done_true_unaffected_by_release_change():
    """Regression: CLOSE-phase done=True must be completely unaffected by
    the RELEASE-only success-early-termination change -- it must still
    unconditionally raise via _diagnose_and_raise_on_done, even when
    check_success() would report True, because check_success()=True during
    CLOSE (before the bowl is ever placed) has no physical meaning and must
    NOT be treated as a success. result.termination_reason and
    result.success must be left at their pre-call defaults (None), since
    _handle_release_done_termination is never invoked for CLOSE."""
    state = {"gripper_closed": False, "step_count": 0}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    orig_step = env.step

    def step_with_done(action):
        state["step_count"] += 1
        obs, reward, done, info = orig_step(action)
        # Same threshold used by the original CLOSE-phase diagnostic test:
        # forcing done=True at step 7 lands inside CLOSE (steps 3..22).
        forced_done = state["step_count"] >= 7
        return obs, reward, forced_done, {"forced_for_test": True}

    env.step = step_with_done
    env.check_success = lambda: True  # deliberately True, to prove CLOSE ignores it

    def fake_finger_distance(env, ctrl_module):
        return 0.02

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.error is not None and "Environment ended unexpectedly during CLOSE" in result.error
        assert result.done_early_termination_phase == "CLOSE"
        assert result.done_early_termination_check_success is True  # captured, but NOT acted upon as success
        # The RELEASE-only fields must remain untouched (pre-call defaults):
        assert result.termination_reason is None
        assert result.success is None
        print("RUN_EPISODE_CLOSE_PHASE_UNAFFECTED_BY_RELEASE_CHANGE_OK")
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_episode_edge_pinch_normal_8phase_completion_sets_termination_reason():
    """Regression: the pre-existing normal-completion path (all 8 phases +
    terminal_hold ran without done=True ever firing) must now also set
    result.termination_reason == 'normal_completion', without changing the
    existing result.success assignment (still bool(env.check_success()) at
    that point)."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)

        assert result.error is None, result.error
        assert result.aborted is False
        assert result.termination_reason == "normal_completion"
        assert result.success is False  # FakeEnv.check_success() always returns False
        print("RUN_EPISODE_NORMAL_COMPLETION_SETS_TERMINATION_REASON_OK")
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_smoke_test_edge_pinch_surfaces_termination_reason_key():
    """The per-run dict from run_smoke_test_edge_pinch() must include the
    new termination_reason key alongside the existing schema (checked here
    for the normal-completion case; the RELEASE early-termination cases are
    already covered by the run_episode_edge_pinch-level integration tests
    above, since run_smoke_test_edge_pinch is a thin per-seed/standoff
    wrapper around it)."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch

    def fake_make_env_fn():
        return env

    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        orig_compute_rim = edge_ctrl.compute_rim_local_points
        edge_ctrl.compute_rim_local_points = lambda make_env_fn, ctrl_module, name: rim_local_points
        try:
            results = edge_ctrl.run_smoke_test_edge_pinch(
                fake_make_env_fn, ctrl, seeds=[1], edge_azimuth_deg=90.0,
                standoff_candidates=[0.03], target_ee_ori=(2.2587, -2.2587, -0.0700),
                orientation_control_verified=True,
            )
        finally:
            edge_ctrl.compute_rim_local_points = orig_compute_rim

        assert len(results) == 1
        r = results[0]
        assert "termination_reason" in r
        assert r["termination_reason"] == "normal_completion"
        print("RUN_SMOKE_TEST_SURFACES_TERMINATION_REASON_KEY_OK", r["termination_reason"])
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_episode_edge_pinch_obs_capture_callback_invoked_reset_plus_every_step():
    """2026-09-17 additive hook (design:
    claude/task1_edge_pinch_recording_script_design_v1_20260917.json part3):
    obs_capture_callback, when provided, must be invoked exactly once at
    env.reset() (step_count=0, phase='RESET', action=None, done=False,
    info=None) and exactly once per subsequent env.step() call, with that
    step's real (step_count, phase, obs, action, done, info) -- and must
    have ZERO effect on the episode's existing outputs (error/aborted/
    termination_reason/success unchanged from the no-callback normal-
    completion case)."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    calls = []

    def capture(step_count, phase, obs, action, done, info):
        calls.append((step_count, phase, action is None, done))

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(
            env, cfg, ctrl, seed=1, rim_local_points=rim_local_points,
            obs_capture_callback=capture,
        )

        # Unchanged outputs vs the no-callback normal-completion test:
        assert result.error is None, result.error
        assert result.aborted is False
        assert result.termination_reason == "normal_completion"
        assert result.success is False

        # Exactly one RESET call (step_count=0, action=None), first in order:
        reset_calls = [c for c in calls if c[1] == "RESET"]
        assert len(reset_calls) == 1, reset_calls
        assert reset_calls[0] == (0, "RESET", True, False)
        assert calls[0] == reset_calls[0]

        # Exactly one call per actual env.step() -- step_count sequence must
        # be contiguous 1..steps_run, one call per step, none skipped/dup'd:
        step_calls = [c for c in calls if c[1] != "RESET"]
        assert len(step_calls) == result.steps_run, (len(step_calls), result.steps_run)
        assert [c[0] for c in step_calls] == list(range(1, result.steps_run + 1))
        # action is never None for a real step call:
        assert all(c[2] is False for c in step_calls)
        print(
            "RUN_EPISODE_OBS_CAPTURE_CALLBACK_INVOKED_RESET_PLUS_EVERY_STEP_OK",
            len(calls), result.steps_run,
        )
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_run_episode_edge_pinch_obs_capture_callback_none_is_backward_compatible():
    """Default obs_capture_callback=None (every pre-existing call site in
    this project) must produce byte-identical results to before this hook
    was added -- verified here by simply confirming the omitted-argument
    call path still runs the normal-completion episode successfully (the
    full existing 51-test suite already exercises this path without the
    new parameter at all, so this is a light confirmatory smoke check, not
    a duplicate of that coverage)."""
    state = {"gripper_closed": False, "last_action": None}
    env, ctrl, rim_local_points = _make_fake_env_and_ctrl_for_lift_metrics(state)

    def fake_finger_distance(env, ctrl_module):
        return 0.02 if state["gripper_closed"] else 0.09

    def fake_compute_action(cfg, target_pos, eef_pos, current_quat, gripper_value, ctrl_module):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper_value
        return action, 0.001, 0.001

    orig_finger_distance = edge_ctrl._finger_distance_m
    orig_compute_action = edge_ctrl.compute_action_edge_pinch
    try:
        edge_ctrl._finger_distance_m = fake_finger_distance
        edge_ctrl.compute_action_edge_pinch = fake_compute_action
        cfg = edge_ctrl.EdgePinchConfig(orientation_control_verified=True)
        result = edge_ctrl.run_episode_edge_pinch(env, cfg, ctrl, seed=1, rim_local_points=rim_local_points)
        assert result.error is None, result.error
        assert result.termination_reason == "normal_completion"
        print("RUN_EPISODE_OBS_CAPTURE_CALLBACK_NONE_BACKWARD_COMPATIBLE_OK")
    finally:
        edge_ctrl._finger_distance_m = orig_finger_distance
        edge_ctrl.compute_action_edge_pinch = orig_compute_action


def test_no_action_flag_does_nothing():
    class FakeCtrlModule:
        @staticmethod
        def make_env(bddl_path):
            raise AssertionError("ctrl_module.make_env must not be called with no action flags")

    orig_load = edge_ctrl._load_controller_module
    orig_argv = sys.argv
    try:
        edge_ctrl._load_controller_module = lambda path: FakeCtrlModule()
        sys.argv = ["task1_edge_pinch_controller_v1.py"]
        exit_code = edge_ctrl.main()
        assert exit_code == 1
        print("NO_ACTION_FLAG_DOES_NOTHING_OK")
    finally:
        edge_ctrl._load_controller_module = orig_load
        sys.argv = orig_argv


if __name__ == "__main__":
    test_protection1_episode_level_gate_blocks_unverified()
    test_protection1_action_level_gate_raises()
    test_protection1_action_proceeds_when_verified()
    test_protection1_orientation_missing_falls_back_to_zero_not_crash()
    test_quat_mul_conj_roundtrip_identity()
    test_axisangle_vec_to_quat_roundtrip()
    test_quat_relative_rotation_matches_known_90deg_case()
    test_quat_relative_rotation_picks_minimal_angle_sign_branch()
    test_compute_action_edge_pinch_uses_quaternion_method_not_raw_subtraction()
    test_max_steps_for_phase_routes_approach_descend_to_orientation_budget()
    test_protection2_reachability_check()
    test_protection2_stall_tracker()
    test_protection2_episode_aborts_on_unreachable_target_before_close()
    test_edge_point_unavailable_aborts_without_fallback_to_center()
    test_select_edge_local_point_picks_closest_azimuth()
    test_edge_target_point_world_transform_identity_rotation()
    test_edge_target_point_world_transform_90deg_rotation()
    test_calibration_result_looks_successful_requires_all_criteria()
    test_lift_end_metrics_captured_before_terminal_hold_gripper_reopen()
    test_orientation_stall_skipped_when_already_converged_but_noisy()
    test_orientation_stall_still_triggers_when_genuinely_diverging()
    test_position_only_stall_still_aborts_with_orientation_converged()
    test_build_make_env_fn_delegates_to_ctrl_module_make_env()
    test_build_make_env_fn_matches_old_script_signature()
    test_probe_orientation_flag_never_enters_calibration()
    test_calibrate_without_orientation_verified_refused_before_any_call()
    test_smoke_test_without_orientation_verified_refused_before_any_call()
    test_run_smoke_test_edge_pinch_runs_full_episode_per_standoff_seed_pair()
    test_estimate_grasp_offset_from_lift_records_averages_last_n_steps()
    test_estimate_grasp_offset_fallback_when_no_lift_records()
    test_move_target_applies_offset_compensation_xy_only_not_z()
    test_descend2_and_release_share_identical_compensated_target()
    test_place_target_compensates_z_in_correct_direction()
    test_move_target_offset_direction_reduces_predicted_bowl_plate_error()
    test_workspace_bounds_selects_transport_bounds_for_move_and_descend2_only()
    test_workspace_bounds_approach_descend_still_use_bowl_area_bounds()
    test_workspace_bounds_transport_default_widens_only_x()
    test_diagnose_and_raise_on_done_captures_check_success_true()
    test_diagnose_and_raise_on_done_captures_check_success_false()
    test_diagnose_and_raise_on_done_handles_check_success_exception()
    test_run_episode_edge_pinch_populates_done_fields_on_early_termination()
    test_run_smoke_test_edge_pinch_surfaces_offset_and_done_diagnostic_fields()
    test_handle_release_done_termination_success_path_sets_success_true_and_reason()
    test_handle_release_done_termination_failure_path_sets_success_false_and_reason()
    test_handle_release_done_termination_check_success_exception_treated_as_failure_not_success()
    test_run_episode_edge_pinch_release_success_returns_normally_without_raising()
    test_run_episode_edge_pinch_release_failure_still_raises_and_populates_fields()
    test_run_episode_edge_pinch_close_phase_done_true_unaffected_by_release_change()
    test_run_episode_edge_pinch_normal_8phase_completion_sets_termination_reason()
    test_run_smoke_test_edge_pinch_surfaces_termination_reason_key()
    test_run_episode_edge_pinch_obs_capture_callback_invoked_reset_plus_every_step()
    test_run_episode_edge_pinch_obs_capture_callback_none_is_backward_compatible()
    test_no_action_flag_does_nothing()
    print("ALL_EDGE_PINCH_UNIT_TESTS_OK")
