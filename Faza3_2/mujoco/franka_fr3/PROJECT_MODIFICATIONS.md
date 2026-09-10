# Project model modifications

The FR3 model and mesh assets originate from MuJoCo Menagerie / Franka Description. Original attribution and licensing are retained in README.md and LICENSE.

Compared with the model distributed in this repository, `scene.xml` adds a non-colliding mocap target marker for the XYZ reaching task. `fr3.xml` and mesh files are unchanged.

The Python environment configures joint position servos and ideal gravity compensation at runtime. Visualization scripts move and resize the target marker for display. The visible marker size is not the reaching tolerance.
