# Robot model

The FR3 and Franka Hand geometry comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The models are distributed under Apache-2.0; the license is kept in [franka_fr3/LICENSE](franka_fr3/LICENSE).

Source models: [Franka FR3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) and [Franka Emika Panda hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).

For this scene the arm uses joint position actuators. The hand is attached to the FR3 flange with a -45 degree local Z rotation, its fingers are fixed closed, and the TCP is 0.1034 m from the hand frame. There is no additional pushing paddle. The table, cube and goal are in scene.xml. Gravity acts on the robot and the cube.

The collision geometry and friction are simulation approximations. They have not been calibrated against the laboratory robot. The XML and every referenced mesh are unchanged from Faza4_15; unused mesh files were omitted from this submission.
