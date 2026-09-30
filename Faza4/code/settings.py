from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "mujoco" / "scene.xml"

# Scene and control (metres, seconds, radians)
TIMESTEP = 0.002
SUBSTEPS = 25
CONTROL_DT = TIMESTEP * SUBSTEPS
ACTION_SCALE = 0.005
ACTION_SCALE_Z = 0.004
TCP_HEIGHT = 0.207
TCP_START = [0.35, 0.0, 0.30]
TCP_Z_LOW = 0.207
TCP_Z_HIGH = 0.32
TCP_LOW = [0.30, -0.21]
TCP_HIGH = [0.65, 0.21]
OBJECT_LOW = [0.40, -0.06]
OBJECT_HIGH = [0.50, 0.06]
GOAL_LOW = [0.38, -0.12]
GOAL_HIGH = [0.56, 0.12]
OBJECT_START = [0.44, 0.0]
GOAL_START = [0.54, 0.0]
TABLE_TOP = 0.18
CUBE_HALF = 0.025
HOME = [0.0, -0.4, 0.0, -2.1, 0.0, 1.7, 0.7854]

SUCCESS_DISTANCE = 0.02
SUCCESS_HOLD = 0.3
MAX_STEPS = 500
POSITION_SCALE = 0.3
VELOCITY_SCALE = 0.2

REWARD_SCALE = 1.0
PROGRESS_WEIGHT = 100.0
APPROACH_DISTANCE_WEIGHT = 1.0
DISTANCE_WEIGHT = 1.0
APPROACH_WEIGHT = 50.0
APPROACH_OFFSET = 0.055
ACTION_WEIGHT = 0.005
SUCCESS_BONUS = 50.0
FAILURE_PENALTY = 50.0
HIGH_APPROACH_WEIGHT = 1.0
HIGH_APPROACH_RADIUS = 0.08
HIGH_APPROACH_SCALE = 0.02

PPO_SETTINGS = dict(
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
)
