from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "mujoco"
FRANKA = MUJOCO / "franka_fr3"
SCENE = MUJOCO / "scene.xml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

TIMESTEP = 0.002
SUBSTEPS = 25
ACTION_SCALE = 0.005
ACTION_SCALE_Z = 0.004
ACTION_SCALE_YAW = 0.05
WRIST_YAW_LIMIT = np.pi / 2
XY_FILTER_ALPHA = 0.5

TCP_HEIGHT = 0.207
TCP_START = [0.35, 0.0, 0.30]
TCP_Z_LOW = 0.207
TCP_Z_HIGH = 0.32
# Reachable box checked on a 2 cm grid (joint margin >= 5 deg, no robot-table contact).
TCP_LOW = [0.32, -0.32]
TCP_HIGH = [0.72, 0.32]
HOME = [0.0, -0.4, 0.0, -2.1, 0.0, 1.7, 0.7854]

TABLE_TOP = 0.18
OBJECT_HALF = [0.04, 0.02, 0.025]
WORKSPACE_LOW = [0.24, -0.25]
WORKSPACE_HIGH = [0.76, 0.25]
OBJECT_LOW = [0.40, -0.08]
OBJECT_HIGH = [0.54, 0.08]

SUCCESS_DISTANCE = 0.02
SUCCESS_ANGLE = np.deg2rad(10.0)
SUCCESS_HOLD = 0.3

KEYPOINT_SCALE = 2.0
POSITION_SCALE = 0.1
VELOCITY_SCALE = 0.2
ANGULAR_VELOCITY_SCALE = 2.0
