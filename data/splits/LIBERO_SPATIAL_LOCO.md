# LIBERO Spatial Target-Role LOCO Splits

## Experimental scope

These are two independently trained leave-one-combination-out
(LOCO) experiments. They are not a full factorial checkerboard.

The primary eligible task pool is:

`[0, 2, 3, 4, 6, 7, 8, 9]`

Each fold uses seven training tasks and one held-out task.

## Holdout task 6

- Training IDs: `[0, 2, 3, 4, 7, 8, 9]`
- Test ID: `[6]`
- Held-out instructed-target composition:
  `next_to × cookie_box`
- `next_to` remains visible through task 8.
- `cookie_box` remains visible through task 3.
- The held-out instructed-target tuple does not occur in training.

## Holdout task 3

- Training IDs: `[0, 2, 4, 6, 7, 8, 9]`
- Test ID: `[3]`
- Held-out instructed-target composition:
  `on × cookie_box`
- `on` remains visible through tasks 7 and 9.
- `cookie_box` remains visible through task 6.
- The held-out instructed-target tuple does not occur in training.

## Excluded task 1

Task 1 describes bowl 1 next to the ramekin, but its distractor
bowl 2 is placed in `main_table_next_to_box_region`. Its images
therefore expose the same black-bowl-next-to-cookie-box visual
configuration used by task 6.

Task 1 is excluded from the shared primary pool to prevent this
visual leakage and to keep the two folds matched at seven training
tasks each.

## Quarantined task 5

Task 5 was removed after an exact 50-state stability audit.

For benchmark states:

- Initial contact and `On`: 0 / 50
- Contact and `On` after one control step: 0 / 50
- After 10 stabilization steps: contact 50 / 50, `On` 0 / 50
- Relative XY drift: 3.721 cm
- Relative Z change: -6.647 cm
- Target-bowl world displacement: 14.209 cm

For HDF5 demonstration states:

- Initial contact and `On`: 0 / 50
- After one step: contact and `On` 50 / 50
- After 10 steps: contact and `On` 50 / 50
- Relative XY drift: 0.300 cm
- Relative Z change: -1.988 cm

This demonstrates a runtime mismatch between serialized task-5
states and the current simulator. It is consistent with the open
LIBERO report:
https://github.com/Lifelong-Robot-Learning/LIBERO/issues/141

Task 5 must not enter training data, test data, normalization
statistics, or checkpoint selection for the primary LOCO results.

## Stabilization checks

A task-3 benchmark spot check became `On=True` after the standard
10-step stabilization period and showed zero additional movement
from steps 10 to 20. Its HDF5 state was stable from the start.

A task-6 benchmark spot check preserved its target-reference
geometry and showed zero additional movement from steps 10 to 20.
Its HDF5 state was also stable.

The full held-out benchmark state sets should still be audited
before final evaluation.

## Interpretation limits

The experiment tests instructed-target task recombination after
fine-tuning. It does not establish full-factorial, scene-unseen,
purely visual, or purely linguistic generalization.

Both tasks contain two visually identical black bowls, so the
instruction helps identify the target instance. However, distractor
positions also vary between tasks and remain a scene-level
confound.

Final evaluation should include correct, blank, and shuffled
instruction controls.
