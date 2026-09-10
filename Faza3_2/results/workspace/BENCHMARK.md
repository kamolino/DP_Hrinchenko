# Global workspace benchmark definition

The validation and final-test task files were generated independently, before evaluating the new controller. Selection uses forward kinematics and collision checks only; no controller successes or failures influence selection.

| Suite | Seed | Tasks | Goals per 30° azimuth sector | Occupied Cartesian cells |
|---|---:|---:|---:|---:|
| Validation | 731003 | 72 | 6 | 71 |
| Final test | 982451 | 216 | 18 | 206 |

Each suite divides tasks equally among `global_home`, `boundary_home` and `random_start`. Global goals span radial and height strata within each of 12 equal azimuth sectors from −180° to +180°. Random starts are separately stratified across the whole sampled workspace, rather than perturbed slightly around home. Boundary goals sample inner/outer radius, upper/lower height and near joint limits. A finite random pool approximates these boundaries; it does not compute an exact workspace surface.

Candidate pools contain 24,000 attempts each: 70% uniform across the full joint ranges with margin and 30% biased toward a joint boundary. The collision-free candidates occupy 312 of the 360 fixed Cartesian cells. Empty cells are **not certified unreachable**. Selecting among occupied cells helps counter the nonuniform Cartesian density produced by uniform joint sampling.

The final test spans radial distance **0.018–0.855 m** from the vertical base axis and TCP height **0.034–1.186 m**. Twenty-one final-test goals are below 0.15 m. Target joint witnesses collectively span 99.82–99.92% of each allowed joint interval. Cartesian minima/maxima, bin counts and exact joint bounds are recorded in `test_coverage.json`.

## Operational bounds and meaning of reachability

- Joint positions remain inside the MuJoCo model limits with a **0.08 rad margin**.
- TCP height is at least **0.03 m above the floor**. This is an operational clearance, not the arm's physical limit and not the reaching tolerance.
- Start and goal witness configurations are free of penetrating contacts under the model's collision checks; fixed base–floor contact is ignored, as in the original precision environment.
- No radial restriction, preferred front-facing sector or goal-orientation restriction is added.
- Start and goal positions must differ by at least 0.03 m to avoid trivial starting success.

Every task stores `start_q`, Cartesian `goal` and `witness_q`. The witness only certifies that a valid joint configuration produces the goal. **A controller must not receive or read `witness_q`.** Neither a valid witness nor valid start proves that a collision-free connecting path exists. Planning failures, collisions, timeouts and poor final accuracy must all remain in reported results.

This is a finite simulation benchmark for position reaching. It does not establish perfect performance at every point of the continuous workspace, prescribed tool orientation, a loaded gripper, added obstacles, calibration error or transfer to hardware. Use validation tasks for development and checkpoint selection; use the final-test tasks only for the final paired evaluation.

## Reproduce in a new output directory

```bash
Faza3_2/.venv/bin/python Faza3_2/code/workspace_benchmark.py \
  --output Faza3_2/results/workspace/reproduced
Faza3_2/.venv/bin/python -m unittest discover \
  -s Faza3_2/tests -p test_workspace_benchmark.py -v
```

The generator refuses to overwrite existing suites. Repeating the same seeds and settings produces identical task definitions. `WorkspaceSamplingEnv` and `valid_workspace_pose` in the generator module expose the same 3 cm clearance rule for controller environments and evaluators.
