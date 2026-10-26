# Robot model

The FR3 and Franka Hand geometry comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The models are distributed under Apache-2.0; the license is kept in [franka_fr3/LICENSE](franka_fr3/LICENSE).

Source models: [Franka FR3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) and [Franka Emika Panda hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).

For this scene the arm uses joint position actuators. The hand is attached to the FR3 flange with a -45 degree local Z rotation, its fingers are fixed closed, and the TCP is 0.1034 m from the hand frame. There is no additional pushing paddle. The table, block and goal are in scene.xml. The letter scenes are generated at runtime by code/stage2_letters (letters.py, letter_env.py) into mujoco/generated/. Gravity acts on the robot and the block.

The collision geometry and friction are simulation approximations. They have not been calibrated against the laboratory robot. The robot XML and referenced meshes are unchanged from Phase 4. Phase 5 changes scene.xml to use a rectangular block and non-colliding orientation markers.
