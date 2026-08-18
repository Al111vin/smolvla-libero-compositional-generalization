# Task 008 Gate-5 physical-feasibility evidence

Task 8 pushes the Akita black bowl to the right region. The evidence
uses frozen draft-v5 layout 1 and reset seed 441000.

The controller uses a safe two-leg route. Its second leg uses the left
finger pad, -8 degree yaw, 0.043 m pad-height offset, 0.00225 m push
increments, and two controller steps per waypoint. At waypoint 76 the
contact line receives a +0.001 m X correction.

The trajectory has 912 actions, within the 930-action safety horizon.
It preserves table support at every action and has no opposite-pad,
grasp, nonselected robot-target, distractor, or unexpected object
contacts. The terminal relation and target-region XY predicate hold
through the final 20 actions.

Formal open-loop replay passed. Initial-state and complete-trajectory
maximum absolute state differences are both 0.0, and all recorded
action, state, contact, support, and goal traces match exactly.
