# Gate 5 method for Task 12

- Protocol: `libero_36_proxy_tabletop_draft_v5`
- Task: `12` (`white_yellow_mug put_inside left`)
- Layout: `1`
- Seed: `481000`
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
- The trajectory and replay SHA-256 hashes are identical: `d37632c95552a673fcf41795acf1fb03398dac2ba22437bb0ff7a2e3efeaca41`.
