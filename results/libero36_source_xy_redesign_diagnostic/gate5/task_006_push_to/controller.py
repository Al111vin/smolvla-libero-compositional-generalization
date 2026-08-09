from __future__ import annotations

import math
import sys

import numpy as np

from scripts import calibrate_libero_36_push as calibrator

_ORIGINAL_ATTEMPT_RECORDER_STEP = (
    calibrator.AttemptRecorder.step
)
CONTACT_SEARCH_TRANSLATION_SCALE = (
    0.6
)


def slowed_contact_search_step(
    self,
    obs,
    action,
    phase,
):
    action = np.asarray(
        action,
        dtype=np.float32,
    ).copy()

    if phase == "contact_search":
        action[:3] = (
            action[:3]
            * CONTACT_SEARCH_TRANSLATION_SCALE
        ).astype(np.float32)

    return _ORIGINAL_ATTEMPT_RECORDER_STEP(
        self,
        obs,
        action,
        phase,
    )


calibrator.AttemptRecorder.step = (
    slowed_contact_search_step
)



CONFIG = {
    6: {
        "yaw_degrees": -71.0,
        "controller_radius_m": 0.085,
        "finger": "right",
        "pad_height_offset_m": 0.035,
    },
    7: {
        "yaw_degrees": -60.0,
        "controller_radius_m": 0.075,
        "finger": "right",
        "pad_height_offset_m": 0.035,
    },
    8: {
        "yaw_degrees": -75.0,
        "controller_radius_m": 0.075,
        "finger": "right",
        "pad_height_offset_m": 0.035,
    },
}


original_settle = calibrator.settle_environment
original_candidates_for = calibrator.candidates_for


def selected_candidates_for(row):
    task_id = int(row["task_id"])
    config = CONFIG[task_id]
    candidates = []

    for index, height in enumerate(
        (0.04550,)
    ):
        candidates.append(
            {
                "candidate_index": index,
                "pusher_finger":
                    config["finger"],
                "pad_height_offset_m":
                    height,
                "candidate_order_key": (
                    f"selected_yaw="
                    f"{config['yaw_degrees']:+.1f};"
                    f"finger={config['finger']};"
                    f"controller_radius="
                    f"{config['controller_radius_m']:.3f};"
                    f"pad_height={height:.3f}"
                ),
            }
        )

    return candidates


def configured_settle(env, row, seed):
    task_id = int(row["task_id"])
    config = CONFIG[task_id]

    calibrator.reset_validator.SOURCE_SAMPLER_RADII_M[
        "akita_black_bowl"
    ] = config["controller_radius_m"]

    settled = original_settle(
        env,
        row,
        seed,
    )

    yaw = math.radians(
        config["yaw_degrees"]
    )
    rotation_z = np.array(
        [
            [
                math.cos(yaw),
                -math.sin(yaw),
                0.0,
            ],
            [
                math.sin(yaw),
                math.cos(yaw),
                0.0,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    settled["desired_rotation"] = (
        rotation_z
        @ settled["desired_rotation"]
    )

    return settled


calibrator.candidates_for = selected_candidates_for
calibrator.settle_environment = configured_settle

sys.argv = [
    "run_bowl_push_selected.py",
    "--mode",
    "smoke",
    "--layout-spec",
    "data/libero_36/layout_spec.csv",
    "--reset-manifest",
    "results/libero_36_reset_audit.manifest.json",
    "--goal-manifest",
    "results/libero_36_goal_audit.manifest.json",
    "--task-ids",
    "6",
    "--output",
    "/tmp/libero36_gate5_bowl_push_task6_final/calibration.csv",
    "--manifest",
    "/tmp/libero36_gate5_bowl_push_task6_final/calibration.manifest.json",
    "--attempts-dir",
    "/tmp/libero36_gate5_bowl_push_task6_final/attempts",
    "--overwrite",
]

calibrator.main()
