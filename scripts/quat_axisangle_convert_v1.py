"""quat_axisangle_convert_v1.py

Standalone quaternion <-> axis-angle conversion utilities for the Task1
edge-pinch diverse-collection RECORDER (task1_edge_pinch_diverse_collection_
recorder_v1.py), plus an intra-episode hemisphere-continuity wrapper.

Design reference: claude/task1_edge_pinch_recording_script_design_v1_
20260917.json (part1). User-confirmed decisions (2026-09-17):
  1. Quaternion ordering is NOT assumed a priori. This module prefers
     robosuite's own robosuite.utils.transform_utils.quat2axisangle (the
     authoritative converter for whatever convention robosuite's own
     obs-dict quaternions actually use) when robosuite is importable, and
     falls back to a manual scalar-last [x,y,z,w] implementation only when
     it is not (offline/local unit testing). This mirrors
     task1_edge_pinch_controller_v1.py's own
     axisangle_from_quat_preferring_robosuite / quat_to_axisangle split
     (kept deliberately consistent with that file rather than reinvented).
  2. Hemisphere continuity (avoiding a quaternion double-cover sign flip
     producing a discontinuous axis-angle jump between consecutive frames,
     a real risk in exactly the near-angle=pi region this task's reset
     orientation sits in) is applied ONLY WITHIN a single episode.
     HemisphereContinuityConverter.reset() must be called at the start of
     every new episode -- episodes are never aligned to each other.
  3. This module does not itself decide whether the (x,y,z,w) ordering
     assumption is correct on the live GPU environment -- that is what
     --self-test in the recorder script verifies (known-rotation checks
     offline, plus an optional live single-episode row-norm consistency
     check against the existing pilot HDF5's obs/ee_ori statistics, which
     is NOT executed by this delivery).

STATUS: CODE ONLY. Not yet executed against the live simulator. No GPU
deployment, no data collection, no HDF5 writing performed by this module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def quat_to_axisangle_fallback(quat: np.ndarray) -> np.ndarray:
    """Manual scalar-last [x,y,z,w] -> axis-angle conversion. UNVERIFIED
    convention warning: only meant as an offline fallback when robosuite is
    not importable (this manual path must never be trusted as the final
    word on the live GPU environment without cross-checking against
    robosuite's own converter -- see axisangle_from_quat_preferring_
    robosuite below, and the recorder's --self-test mode).
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    x, y, z, w = q
    w = float(np.clip(w, -1.0, 1.0))
    xyz_norm = float(np.linalg.norm([x, y, z]))
    angle = 2.0 * np.arctan2(xyz_norm, w)
    if xyz_norm < 1e-9:
        axis = np.zeros(3)
    else:
        axis = np.array([x, y, z]) / xyz_norm
    return axis * angle


def axisangle_from_quat_preferring_robosuite(quat: np.ndarray) -> dict:
    """Prefers robosuite.utils.transform_utils.quat2axisangle (authoritative
    for the live environment's actual obs-dict quaternion convention) over
    the manual fallback above. Falls back only if robosuite is not
    importable, and labels the result accordingly so a fallback-derived
    value is never silently mistaken for a robosuite-verified one.

    Deliberately duplicated (not imported) from
    task1_edge_pinch_controller_v1.py's identically-named function, so this
    conversion module stays standalone/importable without loading the full
    controller module (which requires libero/robosuite/MuJoCo at import
    time via its own dependencies) for offline unit testing. Kept
    byte-for-byte logic-equivalent to the controller's version by design --
    any future change to one must be mirrored in the other.
    """
    try:
        from robosuite.utils.transform_utils import quat2axisangle  # noqa: WPS433

        vec = np.asarray(quat2axisangle(np.asarray(quat, dtype=np.float64)), dtype=np.float64)
        return {
            "axisangle_vec": vec,
            "angle_rad": float(np.linalg.norm(vec)),
            "source": "robosuite.utils.transform_utils.quat2axisangle (authoritative)",
        }
    except Exception as exc:  # noqa: BLE001 -- fall back to the manual, unverified path
        vec = quat_to_axisangle_fallback(quat)
        return {
            "axisangle_vec": vec,
            "angle_rad": float(np.linalg.norm(vec)),
            "source": "manual_fallback_scalar_last_xyzw (UNVERIFIED convention, robosuite not "
            f"importable: {type(exc).__name__}: {exc})",
        }


def axisangle_vec_to_quat(vec: np.ndarray) -> np.ndarray:
    """Inverse of axisangle_vec = axis*angle. Returns scalar-last [x,y,z,w].
    Identical formula to task1_edge_pinch_controller_v1.py's
    axisangle_vec_to_quat (duplicated for the same standalone-import reason
    as above)."""
    vec = np.asarray(vec, dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    half = angle / 2.0
    xyz = axis * np.sin(half)
    return np.array([xyz[0], xyz[1], xyz[2], np.cos(half)])


class HemisphereContinuityConverter:
    """Stateful per-episode wrapper around
    axisangle_from_quat_preferring_robosuite that resolves the quaternion
    double-cover sign ambiguity (q and -q represent the identical rotation)
    frame-to-frame, so the OUTPUT axis-angle sequence for a single episode
    does not exhibit a spurious sign-flip discontinuity purely from which
    representative quaternion the environment happened to return.

    MUST be reset() at the start of every new episode -- continuity is
    intentionally NOT carried across episodes (design decision, part1 open
    question, user-confirmed 2026-09-17: each episode resets independently,
    no physical meaning to cross-episode alignment).
    """

    def __init__(self) -> None:
        self._prev_quat: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_quat = None

    def convert(self, quat: np.ndarray) -> dict:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        flipped = False
        if self._prev_quat is not None:
            if float(np.dot(q, self._prev_quat)) < 0.0:
                q = -q
                flipped = True
        self._prev_quat = q.copy()
        result = axisangle_from_quat_preferring_robosuite(q)
        result["hemisphere_flipped_this_frame"] = flipped
        result["quat_used"] = q
        return result
