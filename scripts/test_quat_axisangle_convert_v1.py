"""test_quat_axisangle_convert_v1.py

Offline/mocked unit tests for quat_axisangle_convert_v1.py. No GPU, no
libero, no simulation env required -- these exercise only pure numpy math
(plus an optional scipy cross-check, skipped automatically if scipy is not
installed). robosuite is very unlikely to be importable in this local
environment, so these tests primarily exercise the manual fallback path;
that is intentional and sufficient to validate the MATH (identity, known
angles, hemisphere continuity, round-trip) independent of which converter
backend supplies it. The recorder's --self-test mode is what validates the
live-environment convention question (fallback vs robosuite output
agreement) once running against the real GPU env -- NOT run by this
delivery.

Run: python3 test_quat_axisangle_convert_v1.py
Expects: ALL_QUAT_AXISANGLE_UNIT_TESTS_OK printed at the end.
"""
from __future__ import annotations

import numpy as np

from quat_axisangle_convert_v1 import (
    HemisphereContinuityConverter,
    axisangle_from_quat_preferring_robosuite,
    axisangle_vec_to_quat,
    quat_to_axisangle_fallback,
)


def _quat_xyzw_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Reference constructor (independent of the module under test) used
    only to build known test-input quaternions."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    xyz = axis * np.sin(half)
    return np.array([xyz[0], xyz[1], xyz[2], np.cos(half)])


# --------------------------------------------------------------------------
# 1. Identity quaternion -> zero vector
# --------------------------------------------------------------------------
def test_identity_quat_gives_zero_vector():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    vec = quat_to_axisangle_fallback(q)
    assert np.allclose(vec, [0.0, 0.0, 0.0], atol=1e-9), vec
    result = axisangle_from_quat_preferring_robosuite(q)
    assert np.allclose(result["axisangle_vec"], [0.0, 0.0, 0.0], atol=1e-9), result
    assert result["angle_rad"] < 1e-9


# --------------------------------------------------------------------------
# 2. 90 degree rotation about a known axis (world Z)
# --------------------------------------------------------------------------
def test_90deg_about_z_axis_known_value():
    angle = np.pi / 2.0
    q = _quat_xyzw_from_axis_angle([0.0, 0.0, 1.0], angle)
    vec = quat_to_axisangle_fallback(q)
    expected = np.array([0.0, 0.0, angle])
    assert np.allclose(vec, expected, atol=1e-6), (vec, expected)


# --------------------------------------------------------------------------
# 3. 180 degree (pi) rotation about a known axis -> magnitude pi
# --------------------------------------------------------------------------
def test_180deg_about_x_axis_magnitude_pi():
    angle = np.pi
    q = _quat_xyzw_from_axis_angle([1.0, 0.0, 0.0], angle)
    vec = quat_to_axisangle_fallback(q)
    norm = np.linalg.norm(vec)
    assert abs(norm - np.pi) < 1e-6, norm
    axis = vec / norm
    assert np.allclose(np.abs(axis), [1.0, 0.0, 0.0], atol=1e-6), axis


# --------------------------------------------------------------------------
# 4. Near-pi perturbed quaternion (angle = pi -/+ 1e-6), constructed via the
#    double-cover (-q) representation for the "+" side -- must remain
#    numerically stable (no NaN/Inf) and both magnitudes close to pi.
# --------------------------------------------------------------------------
def test_near_pi_perturbed_quat_stable():
    axis = np.array([0.267, 0.535, 0.802])  # arbitrary unit-ish axis
    axis = axis / np.linalg.norm(axis)

    q_minus = _quat_xyzw_from_axis_angle(axis, np.pi - 1e-6)
    vec_minus = quat_to_axisangle_fallback(q_minus)
    assert np.all(np.isfinite(vec_minus)), vec_minus
    assert abs(np.linalg.norm(vec_minus) - np.pi) < 1e-4, np.linalg.norm(vec_minus)

    q_plus_raw = _quat_xyzw_from_axis_angle(axis, np.pi + 1e-6)
    q_plus_flipped = -q_plus_raw  # double-cover: represents the identical rotation
    vec_plus = quat_to_axisangle_fallback(q_plus_flipped)
    assert np.all(np.isfinite(vec_plus)), vec_plus
    assert abs(np.linalg.norm(vec_plus) - np.pi) < 1e-4, np.linalg.norm(vec_plus)


# --------------------------------------------------------------------------
# 5. Hemisphere continuity across a synthetic sequence crossing angle=pi,
#    with randomly-injected double-cover sign flips -- WITH the fix applied
#    consecutive frames must stay close; WITHOUT it (raw, unfixed
#    conversion of the same flipped inputs) a large jump must appear, as a
#    negative control proving the test actually exercises the fix.
# --------------------------------------------------------------------------
def test_hemisphere_continuity_across_synthetic_sequence():
    axis = np.array([0.1, 0.2, 0.9])
    axis = axis / np.linalg.norm(axis)
    angles = np.linspace(np.pi - 0.05, np.pi + 0.05, 11)

    raw_quats = [_quat_xyzw_from_axis_angle(axis, a) for a in angles]
    # Inject alternating sign flips on every other frame to simulate the
    # environment returning either representative of the double cover.
    flipped_quats = [q if i % 2 == 0 else -q for i, q in enumerate(raw_quats)]

    # WITH the fix:
    conv = HemisphereContinuityConverter()
    fixed_vecs = []
    for q in flipped_quats:
        r = conv.convert(q)
        fixed_vecs.append(np.asarray(r["axisangle_vec"]))
    fixed_jumps = [
        float(np.linalg.norm(fixed_vecs[i] - fixed_vecs[i - 1])) for i in range(1, len(fixed_vecs))
    ]
    assert max(fixed_jumps) < 0.2, fixed_jumps  # smooth, small per-step change

    # WITHOUT the fix (raw conversion of the same flipped inputs) -- negative
    # control: must show at least one large jump, proving the flips were
    # real and the fix is what suppressed them above.
    raw_vecs = [quat_to_axisangle_fallback(q) for q in flipped_quats]
    raw_jumps = [
        float(np.linalg.norm(raw_vecs[i] - raw_vecs[i - 1])) for i in range(1, len(raw_vecs))
    ]
    assert max(raw_jumps) > 1.0, raw_jumps  # a real, large discontinuity


# --------------------------------------------------------------------------
# 6. Round-trip: axisangle_vec_to_quat(quat_to_axisangle(q)) ~= q (or -q)
# --------------------------------------------------------------------------
def test_roundtrip_axisangle_to_quat_and_back():
    rng = np.random.default_rng(20260917)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = float(rng.uniform(0.0, np.pi))
        q = _quat_xyzw_from_axis_angle(axis, angle)
        vec = quat_to_axisangle_fallback(q)
        q_back = axisangle_vec_to_quat(vec)
        same = np.allclose(q_back, q, atol=1e-6)
        opposite = np.allclose(q_back, -q, atol=1e-6)
        assert same or opposite, (q, q_back)


# --------------------------------------------------------------------------
# 7. Batch shape/dtype sanity + w ~= +-1 edge case (near-zero rotation)
# --------------------------------------------------------------------------
def test_zero_norm_xyz_edge_case_and_shape_dtype():
    q_near_identity = np.array([1e-10, -1e-10, 2e-10, 1.0])
    vec = quat_to_axisangle_fallback(q_near_identity)
    assert vec.shape == (3,)
    assert vec.dtype == np.float64
    assert np.all(np.isfinite(vec))
    assert np.allclose(vec, [0.0, 0.0, 0.0], atol=1e-6)

    q_neg_w = np.array([1e-10, -1e-10, 2e-10, -1.0])  # w ~= -1 edge
    vec2 = quat_to_axisangle_fallback(q_neg_w)
    assert vec2.shape == (3,)
    assert np.all(np.isfinite(vec2))


# --------------------------------------------------------------------------
# Optional cross-check against scipy's Rotation.as_rotvec(), skipped
# automatically if scipy is not installed (never a hard dependency).
# --------------------------------------------------------------------------
def test_cross_check_against_scipy_if_available():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  (skipped: scipy not installed)")
        return

    rng = np.random.default_rng(9)
    for _ in range(10):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = float(rng.uniform(0.01, np.pi - 0.01))
        q_xyzw = _quat_xyzw_from_axis_angle(axis, angle)
        vec = quat_to_axisangle_fallback(q_xyzw)

        # scipy's Rotation.from_quat expects [x, y, z, w] (scalar-last),
        # matching this module's assumed convention.
        scipy_vec = Rotation.from_quat(q_xyzw).as_rotvec()
        assert np.allclose(vec, scipy_vec, atol=1e-6), (vec, scipy_vec)


if __name__ == "__main__":
    tests = [
        test_identity_quat_gives_zero_vector,
        test_90deg_about_z_axis_known_value,
        test_180deg_about_x_axis_magnitude_pi,
        test_near_pi_perturbed_quat_stable,
        test_hemisphere_continuity_across_synthetic_sequence,
        test_roundtrip_axisangle_to_quat_and_back,
        test_zero_norm_xyz_edge_case_and_shape_dtype,
        test_cross_check_against_scipy_if_available,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("ALL_QUAT_AXISANGLE_UNIT_TESTS_OK")
