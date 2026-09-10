"""Reproduce the isolated Gate-5 trajectory for Task 34.

The controller configuration itself is frozen in
``scripts.calibrate_libero_36_push.candidates_for``.  This wrapper keeps the
recording command separate from the evidence directory so reruns cannot
overwrite an accepted artifact.
"""

from __future__ import annotations

import sys

from scripts import calibrate_libero_36_push as calibrator


sys.argv = [
    "task34_gate5_controller",
    "--mode",
    "smoke",
    "--layout-spec",
    "data/libero_36/layout_spec.csv",
    "--reset-manifest",
    "results/libero_36_reset_audit.manifest.json",
    "--goal-manifest",
    "results/libero_36_goal_audit.manifest.json",
    "--task-ids",
    "34",
    "--output",
    "/tmp/libero36_gate5_task34_reproduction/calibration.csv",
    "--manifest",
    "/tmp/libero36_gate5_task34_reproduction/calibration.manifest.json",
    "--attempts-dir",
    "/tmp/libero36_gate5_task34_reproduction/attempts",
    "--overwrite",
]

calibrator.main()
