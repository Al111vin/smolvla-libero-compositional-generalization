"""Probe a detached 90-degree task-25 re-entry from a known-safe state."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c


TRACE = Path("/tmp/libero36_closedloop_task25_reanchor_v1/trajectory.npz")


def main() -> None:
    trace = np.load(TRACE)
    start_index = np.where(trace["phase"] == "live_reanchor_push")[0][0] - 1
    row = next(
        item
        for item in c.reset_validator.read_layout_spec(
            Path("data/libero_36/layout_spec.csv")
        )
        if int(item["task_id"]) == 25 and int(item["layout_id"]) == 1
    )
    env = c.reset_validator.make_environment(Path(row["bddl_path"]))
    try:
        target = c.target_instance(row)
        geometry = c.contact_geometry(env, target)
        env.env.sim.set_state_from_flattened(trace["states"][start_index + 1])
        env.env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        recorder = c.AttemptRecorder(
            env, row, target, geometry, "right", float(c.target_xyz(obs, target)[2])
        )
        initial_rotation = c.eef_rotation(obs)
        angle = math.pi / 2
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0],
             [math.sin(angle), math.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ) @ initial_rotation
        live_target = c.target_xyz(obs, target)
        direction = np.array([-0.55, -0.83], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        pad = c.finger_positions(env.env, geometry)["right"]
        obs, _, _ = c.move_pad_to(
            recorder, obs, geometry, "right", pad - np.r_[direction * 0.055, 0.0],
            initial_rotation, "retreat", 30, 0.25,
        )
        pad = c.finger_positions(env.env, geometry)["right"]
        high = pad.copy()
        high[2] = live_target[2] + 0.12
        obs, _, _ = c.move_pad_to(
            recorder, obs, geometry, "right", high, initial_rotation, "lift", 60, 0.25,
        )
        high = np.r_[live_target[:2] - direction * 0.05, live_target[2] + 0.12]
        obs, _, _ = c.move_pad_to(
            recorder, obs, geometry, "right", high, rotation, "rotate_high", 100, 0.25,
        )
        low = high.copy()
        low[2] = live_target[2] + 0.01
        obs, _, _ = c.move_pad_to(
            recorder, obs, geometry, "right", low, rotation, "descend", 100, 0.15,
        )
        for index in range(25):
            low[:2] += direction * 0.0025
            obs, _, _ = c.move_pad_to(
                recorder, obs, geometry, "right", low, rotation, "search", 3, 0.12,
            )
            left, right, grasp = c.push_contact_record(env, geometry)
            hazards = c.active_unexpected_contacts(env, target, geometry, "right")
            if left or right or grasp or any(hazards):
                print("contact", index, left, right, grasp, hazards)
                break
        arrays = recorder.arrays()
        print("phases", dict(zip(*np.unique(arrays["phase"], return_counts=True))))
        print("target", arrays["target_xyz"][-1].tolist())
        print("contacts", int(arrays["left_contact"].sum()), int(arrays["right_contact"].sum()))
        print("hazards", int(arrays["robot_distractor_contact"].sum()), int(arrays["nonselected_robot_target_contact"].sum()), int(arrays["unexpected_object_contact"].sum()))
    finally:
        env.close()


if __name__ == "__main__":
    main()
