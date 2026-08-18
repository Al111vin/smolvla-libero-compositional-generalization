# Gate 5 method for Task 14

- Protocol: `libero_36_proxy_tabletop_draft_v5`
- Task: `14` (`white_yellow_mug put_inside right`)
- Layout: `1`
- Seed: `501000`
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
- The trajectory and replay SHA-256 hashes are identical: `4781eb50374e1176b5bc6bc1765a238c5104fec21aaa393265b46c19b28cf794`.
