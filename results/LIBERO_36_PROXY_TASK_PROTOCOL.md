LIBERO Registered-Object 36-Task Proxy Protocol

Status: `tabletop_draft_v5`

Protocol ID: `libero_36_proxy_tabletop_draft_v5`

Benchmark ID: `libero_registered_object_36_proxy_tabletop`

## Scope

This benchmark is the user-selected option A: a runnable proxy for the
original 4-object × 3-skill × 3-region objective using objects already
registered in LIBERO.

It is **not** an exact implementation of the earlier textual object set
(`red_bowl`, `white_plate`, `black_box`, `yellow_cup`). The proxy object
set changes object identity, geometry, appearance, and affordances.
Results must therefore be reported as a:

> LIBERO registered-object 36-task proxy benchmark

The previously trained V3/V4 models and their evaluations are pilot
baselines. They are not trained models for this new 36-task benchmark.

## Frozen logical task matrix

The logical task ID order is the Cartesian product:

```python
OBJECTS = [
    "akita_black_bowl",
    "white_yellow_mug",
    "alphabet_soup",
    "cream_cheese",
]

SKILLS = [
    "put_on_top",
    "put_inside",
    "push_to",
]

REGIONS = [
    "left",
    "middle",
    "right",
]
```

This produces exactly 36 logical tasks:

- each object is the target in 9 tasks;
- each skill occurs in 12 tasks;
- each region occurs in 12 tasks;
- each object × skill pair occurs in 3 tasks;
- each object × region pair occurs in 3 tasks;
- each skill × region pair occurs in 4 tasks.

All tasks use `LIBERO_Tabletop_Manipulation`, a Panda robot, the
`OSC_POSE` controller, 20 Hz control, and 128 × 128 agent-view and wrist
images.

The earlier `draft_v0` used `LIBERO_Floor_Manipulation`. Its first
dynamic smoke test was rejected: the four small registered objects and
the plate changed almost exclusively in world z by 0.9–3.7 cm while
their world-xy motion stayed below 1 mm and their final-five-step motion
was effectively zero. The floor target-site predicate also changed
across passive settling. Because this was a workspace/site-height
mismatch rather than horizontal placement instability, `draft_v1`
moves the still-untrained benchmark to LIBERO's native tabletop domain.

The first `draft_v1` tabletop smoke test passed the representative
`put_on_top` task, but the representative `put_inside` task failed only
the receiver's official runtime `On(basket, table_region)` predicate.
The basket remained inside the independently checked xy region and was
stable. Official LIBERO basket tasks likewise use `On(basket, region)`
only to place the basket in `:init`; their goal checks only
`In(target, basket_contain_region)`. Therefore `draft_v2` removes the
redundant receiver-region predicate from the plate and basket BDDL
goals. Receiver location remains part of the task language and is
enforced independently at the terminal state by a center-xy check.

The first 36-task, one-layout `draft_v2` smoke audit then exposed a
genuine source-layout collision. For task 21, layout 0, the Akita bowl
and yellow-and-white mug were in contact at raw reset, throughout all
20 zero-action settle steps, and at the stabilized endpoint. Their
stabilized center separation was only about 9.7 cm. This is not a
raw-reset diagnostic artifact and the contact gates remain unchanged.

The registered assets advertise only a 2.5--3.0 cm sampler radius,
while their true collision meshes extend farther. `draft_v3` therefore
spreads adjacent source-slot centers from 15 cm to 24 cm and shifts the
source row away from the plate/basket receiver. A frozen-mesh audit of
the four cyclic layouts, including arbitrary-axis orientation changes
up to 15 degrees and the full 2 cm horizontal-drift allowance for each
object, found:

- minimum possible adjacent sampled-center separation: 22.0 cm;
- worst audited adjacent pair requirement (mug to soup): 15.76 cm;
- residual adjacent-pair margin after two 2 cm closing drifts: 2.24 cm;
- residual basket-to-mug margin after two 2 cm closing drifts: 3.14 cm;
- farthest audited geometry remains inside the 1.0 m by 1.2 m table:
  `x <= 0.353 m` and `abs(y) <= 0.507 m`.

These calculations cover the frozen object rotations and four cyclic
object-to-slot mappings. They do not replace dynamic reset validation,
and a new camera preview is required because the objects move farther
toward the lateral image boundaries.

The representative `draft_v3` physical smoke audit passed all 12
combinations of tasks 0, 3, and 6 with layouts 0--3. Its default
45-degree agent-view images nevertheless placed the outer source
objects at, or beyond, the lateral image boundary. The physical layout
was therefore retained, but the observation contract was rejected.

An agent-view vertical-FOV sweep rendered the same four settled task-0
layouts at 60, 65, 70, 75, and 80 degrees. The estimated worst current
horizontal margin increased from approximately 0 pixels at 60 degrees
to 4--5 at 65, 10--11 at 70, 15 at 75, and 18 at 80. The full slot
sampling offset, allowed 2 cm horizontal drift, and allowed 15-degree
orientation change consume approximately 8--9 pixels of reserve.
`draft_v4` therefore froze the agent-view vertical FOV at 75 degrees:
it retained about 6 pixels of worst-case reserve, while 80 degrees added
little safety and made all manipulable objects smaller. `draft_v5`
retains this frozen 75-degree camera contract while replacing the source
XY geometry.

No result or demonstration from `draft_v0`, `draft_v1`, `draft_v2`,
`draft_v3`, or `draft_v4` may be combined with `draft_v5`. Although
`draft_v4` and `draft_v5` share the 75-degree camera contract, their
source geometry and trajectories are not interchangeable. Images from
the earlier 45-degree contracts and the frozen 75-degree contract must
also never be mixed in one dataset.

## Scene and layout controls

All four manipulable objects are present in every scene. One is the
language-specified target and the remaining three are distractors.

Each logical task has four cyclic, balanced source layouts. Across the
four layouts, every object appears exactly once in every source slot.
Thus the benchmark has:

- 36 logical tasks;
- 4 layouts per task;
- 144 generated BDDL files.

The frozen source slots are:

| Slot | x min | y min | x max | y max |
|---:|---:|---:|---:|---:|
| 0 | 0.04 | -0.244 | 0.11 | -0.182 |
| 1 | 0.04 | -0.102 | 0.11 | -0.040 |
| 2 | 0.04 | 0.040 | 0.11 | 0.102 |
| 3 | 0.04 | 0.182 | 0.11 | 0.244 |

The frozen destination zones are:

| Region | x min | y min | x max | y max |
|---|---:|---:|---:|---:|
| left | -0.20 | -0.28 | -0.04 | -0.14 |
| middle | -0.20 | -0.07 | -0.04 | 0.07 |
| right | -0.20 | 0.14 | -0.04 | 0.28 |

The frozen observation-camera contract is:

- camera name: `agentview`;
- MuJoCo model position: `[0.6586131746834771, 0.0,
  1.6103500240372423]` metres;
- MuJoCo model quaternion, WXYZ: `[0.6380177736282349,
  0.3048497438430786, 0.30484986305236816,
  0.6380177736282349]`;
- vertical field of view: 75 degrees;
- image size: 128 by 128 pixels.

The pose and quaternion are dependency guards: they are checked but
not overwritten. The FOV is deliberately applied after every
successful internal environment reset because a LIBERO hard reset
reconstructs the MuJoCo model and restores its default 45-degree FOV.
After applying 75 degrees, the observation must be force-refreshed
without advancing physics; the observation returned directly by reset
must not be used. Demonstration collectors and evaluators must use the
same shared camera helper as the reset validator.

The confirmed image convention is:

- image left ≈ negative world y;
- image right ≈ positive world y.

The 75-degree task-0 four-layout sweep confirmed this convention and
selected the FOV. `draft_v5` retains this frozen camera contract. A new
representative v5 preview covering the plate, basket, and push scene
types was saved and visually inspected. All required scene elements
were fully visible, completing Gate 2.

## Skill and success definitions

### Put on top

The skill receiver is `plate_1`. The formal BDDL goal is:

```text
On(target, plate_1)
```

Formal benchmark success additionally requires:

```text
terminal_center_xy(plate_1) inside requested_main_table_region
```

This independent terminal check prevents success after moving the plate
outside the language-specified region without relying on a brittle
receiver-to-site height predicate.

### Put inside

The skill receiver is `basket_1`. The formal BDDL goal is:

```text
In(target, basket_1_contain_region)
```

Formal benchmark success additionally requires:

```text
terminal_center_xy(basket_1) inside requested_main_table_region
```

This matches official LIBERO basket tasks: the basket's region is an
initial placement condition, while containment is the BDDL goal. The
independent terminal xy check still prevents success after moving the
basket outside the language-specified region.

### Push to

The BDDL terminal predicate is:

```text
On(target, requested_main_table_region)
```

BDDL alone cannot distinguish pushing from pick-and-place. A formal
push success must additionally require:

```text
no bilateral grasp during the trajectory
AND maximum lift from the post-stabilization height <= frozen threshold
```

The current `0.03 m` lift value is provisional. It must be calibrated
from passive-settle traces and valid push pilot trajectories, then
frozen before demonstrations or formal evaluation. Until then, push
tasks are not protocol-ready.

## Required gates before demonstrations or training

## Current draft-v5 validation status

- Gate 1 static generation: passed; 36 task rows and 144 task-layout
  BDDL files were generated under the v5 source-XY geometry.
- Gate 2 environment smoke: passed; all 12 representative task-layout
  rows passed automated reset and determinism checks, and all initial
  and settled agent-view previews passed visual inspection.
- Gate 3 complete reset audit: passed, 720/720 resets.
- Gate 4 goal-semantics audit: passed, 96/96 positive and isolated
  negative cases.
- Gate 5 physical feasibility: incomplete. Exact, safely replayable
  trajectories have passed for tasks 0 through 7
  (`akita_black_bowl`: three `put_on_top`, three `put_inside`, and two
  `push_to`) and task 26 (`alphabet_soup push_to`), all on layout 1.
  This is 9/36 logical tasks and does not satisfy Gate 5.
- Demonstration collection and SmolVLA training remain blocked until
  all 36 Gate 5 trajectories pass.
- The push lift threshold remains provisional and is not yet promoted
  or frozen.


### Gate 1: static generation

- exactly 36 task-spec rows;
- exactly 144 task-layout rows and BDDL files;
- unique task tuples, languages, file paths, and BDDL hashes;
- registered object types;
- exact registered sampler radii: 0.025 m for bowl, mug, and soup,
  and 0.030 m for cream cheese;
- exact factor balance;
- exact parsed goal predicates and region coordinates;
- an identical canonical frozen-camera JSON record in every task and
  layout row.

### Gate 2: environment smoke

Run tasks 0, 3, and 6 across layouts 0--3 with one reset each (12 rows)
so that plate, basket, and push scenes are all represented. Require:

- BDDL parsing;
- bounded reset attempts;
- 20 all-zero passive-settle steps;
- raw-reset and post-stabilization goal both false;
- raw-reset object-object contacts recorded as diagnostic only because
  the simulator has not yet advanced; every zero-action settle step and
  the stabilized endpoint must have no unexpected object-object
  contact;
- every manipulable source object satisfies both the official LIBERO
  `On` predicate and the independent xy-bounds check after
  stabilization;
- the plate or basket receiver satisfies its independent xy-bounds
  check after stabilization; the receiver's official `On` predicate is
  recorded as a diagnostic but is not a gate;
- the applicable placement checks remain true throughout the final five
  settle steps;
- valid object poses and 128-by-128 camera images rendered with the
  frozen 75-degree agent-view contract;
- raw-to-stabilized horizontal displacement remains below the frozen
  bounds; raw vertical displacement is recorded but is not classified
  as horizontal placement drift;
- motion remains below the frozen bounds over the final five steps;
- saved initial and settled previews;
- independent same-seed determinism replay.

The 12 settled agent-view previews must be inspected to confirm that
all objects, receivers, and destination regions are fully visible.

In `draft_v5`, aggregate CSV fields named
`*_receiver_region_ok` mean the independent receiver xy check. The
separate `*_receiver_official_predicate_ok` fields remain diagnostic and
may be false on an otherwise passing row.

### Gate 3: complete reset audit

Run:

```text
36 tasks × 4 layouts × 5 resets = 720 resets
```

Every reset must pass the frozen checks. A smoke run or a partial CSV
does not satisfy this gate.

The `tabletop_draft_v5` full-reset thresholds are:

| Check | Threshold |
|---|---:|
| maximum manipulable-object horizontal drift | 0.02 m |
| maximum plate/basket horizontal drift | 0.02 m |
| maximum orientation drift | 15 degrees |
| maximum per-step translation over the final 5 settle steps | 0.001 m |
| maximum per-step orientation change over the final 5 steps | 1 degree |
| maximum placement attempts per reset | 100 |

Changing one of these values requires a new protocol version; full mode
does not permit command-line threshold overrides.

### Gate 4: positive and negative goal tests

For every logical task:

- construct an oracle positive terminal state;
- verify the exact success predicate becomes true;
- for `put_on_top` and `put_inside`, independently verify that the
  receiver center remains inside the requested main-table xy region;
- keep the state stable for 20 steps;
- construct at least one near-miss negative state and verify failure.

### Gate 5: physical feasibility

Before bulk data collection, complete at least one valid robot or human
trajectory for every one of the 36 logical tasks. Specifically verify:

- every target object can be placed stably on the plate;
- every target object can be inserted into the basket;
- every target object can be pushed to all three zones without grasping
  or excessive lifting.

Only after all five gates pass may the task version be promoted and the
demonstration/training protocol be frozen.

## Reproducibility records

The generated manifests and validation outputs must record:

- LIBERO, robosuite, MuJoCo, and NumPy versions;
- Git commit and script SHA256;
- BDDL and manifest SHA256 values;
- task ID, layout ID, placement seed, and object-to-slot mapping;
- reset-attempt count and passive-settle settings;
- raw and stabilized poses, horizontal drift, vertical settling
  displacement, contacts, images, and goal status;
- raw-reset contact diagnostics separately from hard-gated contacts at
  every physics settle step and the stabilized endpoint;
- explicit failed-gate names for every rejected reset;
- all thresholds used by the validator;
- frozen camera protocol, position, WXYZ quaternion, vertical FOV,
  image dimensions, shared-helper path and SHA256, and the verified
  runtime camera record for every reset;
- saved preview paths and SHA256 values whenever preview rendering is
  enabled;
- confirmation that the reset-returned 45-degree observation was
  discarded and refreshed at 75 degrees without a physics step;
- independent receiver xy status and the diagnostic-only receiver
  official-`On` status;
- push no-grasp and calibrated no-lift settings.

## Scientific limitation

The three factors are not perfectly orthogonal:

- `put_on_top` reveals a plate receiver;
- `put_inside` reveals a basket receiver;
- `push_to` uses a tabletop target zone;
- destination changes path length and reachability.

The appropriate interpretation is:

> object identity × goal schema / receiver type × destination location

This benchmark measures fixed-template compositional generalization
within one simulator and one asset set. It does not establish
open-vocabulary, language, asset, or domain generalization.

## Primary implementation references

- LIBERO registered objects:
  <https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/objects/google_scanned_objects.py>
- Official LIBERO push-task BDDL example:
  <https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_goal/push_the_plate_to_the_front_of_the_stove.bddl>
- Official LIBERO basket-task BDDL example:
  <https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl>
