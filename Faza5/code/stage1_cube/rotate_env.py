"""Rotation primitive: turn the block in place until it is within 30 deg / 3 cm of the goal pose.

Discrete pusher velocity (11 bins per axis) as in Del Aguila Ferrandis et al. (2023), arXiv:2308.02459.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common import settings as cfg
from common.block_sim import BlockSim
from common.geometry import fits_workspace, rot, wrap

ACTION_BINS = 11
MAX_STEPS = 400
PUSHER_START_GAP = (0.02, 0.05)
PUSHER_MAX_DISTANCE = 0.15
DONE_DISTANCE, DONE_ANGLE = 0.03, np.deg2rad(30)
CURRICULUM_TURNS = [(45, 60), (40, 90), (40, 135), (40, 180)]
VALIDATION_TURNS = (40, 180)
KEYPOINT_PROGRESS = 50.0
ANGLE_PROGRESS = 10.0
SUCCESS_BONUS = 50.0
FAILURE_PENALTY = 20.0


def decode(action):
    return (np.asarray(action, dtype=float) - ACTION_BINS // 2) / (ACTION_BINS // 2)


def sample_task(rng, turns=VALIDATION_TURNS):
    """Goal position = block position, requested turn in `turns` (deg), pusher a few cm outside the block."""
    for _ in range(10000):
        obj = rng.uniform(cfg.OBJECT_LOW, cfg.OBJECT_HIGH)
        yaw = rng.uniform(-np.pi, np.pi)
        turn = rng.choice([-1, 1]) * np.deg2rad(rng.uniform(*turns))
        goal_yaw = wrap(yaw + turn)
        direction = rng.uniform(-np.pi, np.pi)
        gap = rng.uniform(*PUSHER_START_GAP)
        local_dir = rot(-yaw) @ np.array([np.cos(direction), np.sin(direction)])
        support = np.dot(np.abs(local_dir), cfg.OBJECT_HALF[:2])
        pusher = obj + (support + gap) * np.array([np.cos(direction), np.sin(direction)])
        if not (np.all(pusher > cfg.TCP_LOW) and np.all(pusher < cfg.TCP_HIGH)):
            continue
        if fits_workspace(obj, yaw) and fits_workspace(obj, goal_yaw):
            return dict(object=obj.tolist(), goal=obj.tolist(), yaw=float(yaw),
                        goal_yaw=float(goal_yaw), pusher=pusher.tolist())
    raise RuntimeError("Could not sample a rotation task.")


def fixed_tasks(seeds):
    return [sample_task(np.random.default_rng(seed)) for seed in seeds]


class RotateEnv(gym.Env):
    def __init__(self):
        self.sim = BlockSim()
        self.action_space = spaces.MultiDiscrete([ACTION_BINS, ACTION_BINS])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float32)
        self.level = 0
        self.finished = True

    def set_level(self, level):
        self.level = int(np.clip(level, 0, len(CURRICULUM_TURNS) - 1))

    def reached(self):
        return self.sim.distance() < DONE_DISTANCE and self.sim.angle_error() < DONE_ANGLE

    def failure_reason(self):
        reason = self.sim.failure_reason()
        if not reason and np.linalg.norm(self.sim.tcp[:2] - self.sim.object_xy) > PUSHER_MAX_DISTANCE:
            reason = "pusher_far"
        return reason

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        task = (options or {}).get("task") or sample_task(self.np_random, CURRICULUM_TURNS[self.level])
        self.task = task
        self.sim.reset(task, start_tcp=[*task["pusher"], cfg.TCP_HEIGHT])
        reason = self.failure_reason()
        if reason:
            raise RuntimeError(f"Invalid reset: {reason}")
        self.previous = (self.sim.keypoint_error(), self.sim.angle_error())
        self.finished = False
        return self.sim.observation(), self.info(False, "")

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after an episode ends.")
        sim = self.sim
        sim.apply(decode(action), -1.0)
        success, contact = sim.simulate(self.reached)
        reason = contact or self.failure_reason()
        success = success and not reason
        keypoint, angle = sim.keypoint_error(), sim.angle_error()
        reward = (KEYPOINT_PROGRESS * (self.previous[0] - keypoint) + ANGLE_PROGRESS * (self.previous[1] - angle)
                  + (SUCCESS_BONUS if success else 0.0) - (FAILURE_PENALTY if reason else 0.0))
        self.previous = (keypoint, angle)
        terminated = success or bool(reason)
        truncated = sim.steps >= MAX_STEPS and not terminated
        if success:
            reason = "success"
        elif truncated:
            if sim.distance() >= DONE_DISTANCE:
                reason = "timeout_position"
            elif sim.angle_error() >= DONE_ANGLE:
                reason = "timeout_orientation"
            else:
                reason = "timeout_hold"
        self.finished = terminated or truncated
        return sim.observation(), float(reward), terminated, truncated, self.info(success, reason)

    def info(self, success, reason):
        return dict(is_success=bool(success), reason=reason, level=self.level, steps=self.sim.steps,
                    position_error_cm=100 * self.sim.distance(),
                    angle_error_deg=float(np.rad2deg(self.sim.angle_error())),
                    turn_deg=float(np.rad2deg(abs(wrap(self.task["goal_yaw"] - self.task["yaw"])))))

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
