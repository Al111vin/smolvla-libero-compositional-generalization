# Gate 5 method for Task 10

- Protocol: `libero_36_proxy_tabletop_draft_v5`
- Task: `10` (`white_yellow_mug put_on_top middle`)
- Layout: `1`
- Seed: `461000`
- Target object: `white_yellow_mug_1`
- Receiver: `plate_1`
- Controller: `../mug_put_on_top_controller/record_trajectories.py`
- Replay verifier: `../mug_put_on_top_controller/replay_trajectories.py`
- Rim offset: `(+0.044 m X, +0.059 m Z)`
- Gripper close steps: `45`
- A short upward seating motion was used before transport.
- The trajectory passed grasp, relation, terminal stability, and strict contact-safety checks.
- Exact replay passed with initial-state and trajectory-state maximum absolute differences of `0.0`.
- All recorded traces matched exactly.
- The trajectory and replay NPZ SHA-256 hashes are identical: `158c021feecfbca61fe26c3598b5150e7dc8a4cd28916f6cdc15373ae3e0022c`.
