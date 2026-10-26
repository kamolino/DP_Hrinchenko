"""Input of the pushing model (model 32): 24 values, the block's 180 deg symmetry folds the yaw into +/-90 deg."""
import mujoco
import numpy as np

from common.geometry import wrap

POSITION_SCALE = 0.3
VELOCITY_SCALE = 0.2
ANGULAR_VELOCITY_SCALE = 2.0
TCP_HEIGHT = 0.207


def push_observation(sim):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_SITE, sim.controller.site, velocity, 0)
    rotation = sim.data.xmat[sim.object_body].reshape(3, 3)
    yaw, goal_yaw = np.arctan2(rotation[1, 0], rotation[0, 0]), sim.goal_yaw
    if abs(yaw) > np.pi / 2:
        yaw, goal_yaw = wrap(yaw - np.pi), wrap(goal_yaw - np.pi)
    obj = sim.data.qpos[sim.object_qpos:sim.object_qpos + 3]
    tcp = sim.tcp
    obs = np.r_[
        (obj - tcp) / POSITION_SCALE, (sim.goal - obj[:2]) / POSITION_SCALE,
        sim.data.qvel[sim.object_dof:sim.object_dof + 3] / VELOCITY_SCALE, velocity[3:6] / VELOCITY_SCALE,
        (sim.controller.target - tcp) / 0.02,
        (tcp - [0.47, 0.0, TCP_HEIGHT]) / [POSITION_SCALE, POSITION_SCALE, 0.1],
        np.cos(yaw), np.sin(yaw), np.cos(goal_yaw), np.sin(goal_yaw),
        sim.data.qvel[sim.object_dof + 5] / ANGULAR_VELOCITY_SCALE, sim.applied_xy,
    ]
    return obs.astype(np.float32)
