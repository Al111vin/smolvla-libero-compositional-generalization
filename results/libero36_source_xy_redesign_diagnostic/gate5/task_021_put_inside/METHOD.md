# Task 021 Gate-5 trajectory

This directory contains the exact, safely replayable layout-1
trajectory for task 21 (`alphabet_soup put_inside left`)
under `libero_36_proxy_tabletop_draft_v5`.

Controller configuration:

- layout: 1
- object: `alphabet_soup_1`
- receiver: `basket_1_contain_region`
- destination region: `left`
- grasp X offset: 0.000 m
- grasp Z offset: -0.015 m
- close steps: 45
- vertical lift: 0.110 m
- carry height above basket: 0.240 m
- release height above basket: 0.140 m
- direct collision-free approach

The recorded attempt passed all trajectory checks. The object was
grasped before transport, placed inside the basket, released, and the
goal relation held through all terminal actions. The basket remained
inside its requested destination region. There was no robot-distractor,
robot-basket, or unexpected object contact.

Formal open-loop replay passed. Initial-state and complete-trajectory
maximum absolute state differences were exactly zero. All recorded
state, contact, grasp, relation, done, target-position, and
end-effector traces matched exactly. `trajectory.npz` and
`replay.npz` are byte-identical.
