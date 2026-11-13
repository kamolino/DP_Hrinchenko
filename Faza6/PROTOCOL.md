# Phase 6 protocol (fixed before the final runs, 12.11.2026)

- Code base: Phase 5, git commit `278d2cd` (`Faza5/` is not modified; Phase 6 uses its scene and final models).
  Phase 6 code is a copy with new parameters whose defaults reproduce Phase 5.
- Control: Phase 5 code and Phase 6 code on the dev set give identical results in 150/150 scenes
  (success, steps, repairs, disturbances, final errors) → the change is neutral and the evaluation is deterministic.
- Scene sets: dev = 150 scenes, seed 1102000 (only for checks); final = 500 scenes, seed 1202000 (every variant,
  the same list of start poses). Phase 5 test seed 992000 is not used.
- Word success: at the end of the scene all three letters are < 2 cm and < 10° from their goals.
- A scene is disturbed if a letter that was already placed (phase success) moved > 1 cm or > 5° from the pose it
  had when it was placed (checked after every phase).
- Steps: all control steps of the scene (scripted moves, descents, policy, repairs). Global limit 9000.
- Repair phases: repair phases actually executed.
- Failure category (one per scene): letter_outside / letter_fell / letter_tilted / letter_lifted / high_contact /
  top_contact / ... (physical failure, scene stops), total_limit (9000 steps), passes_exhausted (all allowed passes
  done, a letter is not at its goal).
- A1: --repair 0/1/2/4, everything else Phase 5.
- A2: --segment-length 0 (straight)/0.06/0.12/0.24 with one budget of 1600 policy steps per placement attempt for
  all pieces together (intermediate pieces at most 500 each, the last piece gets the rest), --repair 4.
  1600 = the Phase 5 budget of the longest frequent route (3 pieces: 500+500+600); Phase 5 gave 1000/1100/1600/2100
  for 1/2/3/4 pieces. The A2 reference is the new 12 cm run, not Phase 5.
- A3/A4: start = step-1 single-letter checkpoints `models/start/*_single.zip`, the same word states
  (`models/start/word_states_645000.json`), PPO 100k critic warm-up + 1.5M steps, 3 envs, Phase 5 hyperparameters,
  model = the last checkpoint (no selection). Training seeds: offsets 0, 10000, 20000.
  A3-single: single fraction 1.0; A3-mixed: 0.25. A4-bez: disturbance coefficients 0 (termination kept);
  A4-s = A3-mixed.
- Statistics: Wilson 95 % CI; paired differences on the same scenes with Newcombe 95 % CI and exact McNemar;
  trained variants per training seed + seed/scene bootstrap.
- A5 (added 13.11.2026 before any A5 run): shared policy trained from scratch with the Phase 6 DAgger recipe
  (student initialised from the Phase 5 letter-A policy, teachers = Phase 5 letter policies, 4 iterations, the last
  iteration is used, no selection), with the one-hot ID (A5-id) or with the ID inputs always 0 (A5-noid);
  training seeds 0, 10000, 20000. Evaluated on the same 500 final scenes with the unchanged manager and on the
  single-letter sets. The A5-noid models get zero weights on the ID inputs, which does not change their function.
