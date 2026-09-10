# Gate 5 method for Task 13

- Protocol: `libero_36_proxy_tabletop_draft_v5`
- Task: `13` (`white_yellow_mug put_inside middle`)
- Layout: `1`
- Seed: `491000`
- Target object: `white_yellow_mug_1`
- Receiver: `basket_1`
- Controller: `../mug_put_inside_controller/record_trajectories.py`
- Replay verifier: `../mug_put_inside_controller/replay_trajectories.py`
- Rim offset: `(+0.044 m X, +0.059 m Z)`
- Gripper close steps: `45`
- Release height offset above basket: `+0.140 m`
- A short upward seating motion was used before transport.
- Strict grasp, release, relation, stability, and contact-safety checks passed.
- Exact replay passed with initial-state and trajectory-state maximum absolute differences of `0.0`.
- The trajectory and replay SHA-256 hashes are identical: `24b00fd3f243edc1bdc03eed8a57221151f0fdc7cc63e012f91b6c99b8b32cf7`.
