# Task 007 Gate-5 trajectory

This directory contains the isolated successful layout-1 trajectory for
task 7 (`akita_black_bowl push_to middle`) under
`libero_36_proxy_tabletop_draft_v5`.

Controller configuration:

- reset index: 0
- yaw: +72 degrees
- selected finger: left
- controller radius: 0.075 m
- approach standoff: 0.090 m
- pad height offset: 0.047 m
- contact-search XYZ action scale: 0.60
- push XYZ action scale: 0.60
- push waypoint increment: 0.00125 m
- active gripper action: -1.0 (open)

The contact-search action scale applies only during `contact_search`.
The push action scale applies only during `push`. Retreat and terminal
hold retain the formal calibrator behavior.

The isolated attempt passed all per-trajectory checks: selected-pad
contact was acquired, the opposite pad never contacted the target,
there was no nonselected robot-target contact, no robot-distractor or
unexpected object contact, table support held throughout all active
steps, and the terminal relation and XY predicates remained stable.

The formal calibrator replayed the saved actions deterministically.
The maximum absolute trajectory-state replay difference was exactly
zero. `trajectory.npz` retains both the original and embedded replay
arrays; `replay.npz` contains the extracted replay arrays.
