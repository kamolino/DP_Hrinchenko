"""Stage 1 with one policy: push the block to a pose (x, y, yaw). Action x/y/z, 30 observations."""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from common import settings as cfg
from common.block_sim import BlockSim
from common.geometry import wrap
from stage1_cube.tasks import make_tasks

MAX_STEPS = 1000
POSITION_PROGRESS = 100.0
ANGLE_PROGRESS = 10.0
SUCCESS_BONUS = 50.0
FAILURE_PENALTY = 20.0


def observation(sim):
    """30 values for the pushing policies (block and letters alike)."""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_SITE, sim.controller.site, velocity, 0)
    obj = sim.data.qpos[sim.object_qpos:sim.object_qpos + 3]
    tcp, yaw, goal_yaw = sim.tcp, sim.yaw, sim.goal_yaw
    to_object = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    to_goal = np.array([[np.cos(goal_yaw), np.sin(goal_yaw)], [-np.sin(goal_yaw), np.cos(goal_yaw)]])
    turn = wrap(goal_yaw - yaw)
    obs = np.r_[
        (obj - tcp) / 0.1,
        (sim.goal - obj[:2]) / 0.1,
        to_goal @ (obj[:2] - sim.goal) / 0.1,
        to_object @ (tcp[:2] - obj[:2]) / 0.1,
        np.cos(yaw), np.sin(yaw), np.cos(goal_yaw), np.sin(goal_yaw),
        np.cos(turn), np.sin(turn),
        sim.data.qvel[sim.object_dof:sim.object_dof + 3] / 0.2,
        sim.data.qvel[sim.object_dof + 5] / 2.0,
        velocity[3:6] / 0.2,
        (sim.controller.target - tcp) / 0.02,
        (tcp - [0.47, 0.0, cfg.TCP_HEIGHT]) / [0.3, 0.3, 0.1],
        sim.applied_xy,
    ]
    return obs.astype(np.float32)


class CubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task_seed=None, sim=None):
        self.sim = sim or BlockSim()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float32)
        self.task_rng = np.random.default_rng(task_seed)
        self.finished = True

    def new_task(self):
        return make_tasks(1, int(self.task_rng.integers(1 << 30)))[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.task_rng = np.random.default_rng(seed)
        self.task = (options or {}).get("task") or self.new_task()
        self.sim.reset(self.task)
        self.previous = (self.sim.distance(), self.sim.angle_error())
        self.finished = False
        return observation(self.sim), self.info(False, "")

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after an episode ends.")
        sim = self.sim
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        success, reason = sim.act(action[:2], action[2])
        distance, angle = sim.distance(), sim.angle_error()
        reward = (POSITION_PROGRESS * (self.previous[0] - distance) + ANGLE_PROGRESS * (self.previous[1] - angle)
                  + (SUCCESS_BONUS if success else 0.0) - (FAILURE_PENALTY if reason else 0.0))
        self.previous = (distance, angle)
        terminated = success or bool(reason)
        truncated = sim.steps >= MAX_STEPS and not terminated
        if success:
            reason = "success"
        elif truncated:
            reason = "timeout"
        self.finished = terminated or truncated
        return observation(sim), float(reward), terminated, truncated, self.info(success, reason)

    def info(self, success, reason):
        sim = self.sim
        return dict(is_success=bool(success), reason=reason, steps=sim.steps,
                    position_error_cm=100 * sim.distance(), angle_error_deg=float(np.rad2deg(sim.angle_error())))

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
