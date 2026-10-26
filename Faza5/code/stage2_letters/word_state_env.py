"""Training episodes that start from states the word manager really visits (recorded with --record).

Reward = position progress + angle progress + 50 at success - 20 on a failure of any letter, minus the
displacement and rotation of letters that are already in place; -20 and end if one moves > 1 cm or > 5 deg.
"""
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from common.geometry import wrap
from stage1_cube.cube_env import observation
from stage2_letters.letter_env import TRAIN_SEEDS, LetterEnv
from stage2_letters.word_sim import WordSim

MAX_STEPS = 600
POSITION_PROGRESS, ANGLE_PROGRESS, SUCCESS_BONUS, FAILURE_PENALTY = 100.0, 10.0, 50.0, 20.0
DISTURB_POSITION, DISTURB_ANGLE = 100.0, 10.0
DISTURB_LIMIT_M, DISTURB_LIMIT_RAD = 0.01, np.deg2rad(5)


def load_states(path, letter):
    return [s for s in json.loads(Path(path).read_text()) if s["letter"] == letter]


class WordStateEnv(gym.Env):
    def __init__(self, letter, states_path, task_seed=None):
        self.letter = letter
        self.sim = WordSim()
        self.states = load_states(states_path, letter)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float32)
        self.rng = np.random.default_rng(task_seed)
        self.finished = True

    def restore(self, state):
        sim = self.sim
        mujoco.mj_resetData(sim.model, sim.data)
        sim.data.qpos[:] = state["qpos"]
        sim.data.qvel[:] = 0
        sim.data.ctrl[sim.controller.actuators] = sim.data.qpos[sim.controller.qpos]
        mujoco.mj_forward(sim.model, sim.data)
        sim.controller.target = np.array(state["target"], dtype=float)
        sim.controller.yaw = 0.0
        sim.applied_xy[:] = 0
        sim.last_action[:] = 0
        sim.activate(self.letter)
        sim.goal, sim.goal_yaw = np.array(state["goal"][:2]), float(state["goal"][2])
        sim.hold_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        state = (options or {}).get("state") or self.states[self.rng.integers(len(self.states))]
        self.restore(state)
        self.state = state
        self.placed = [n for n in state["placed"] if n != self.letter]
        self.placed_pose = {n: self.sim.pose(n) for n in self.placed}
        self.previous = (self.sim.distance(), self.sim.angle_error(), 0.0, 0.0)
        self.steps, self.finished = 0, False
        return observation(self.sim), self.info(False, "")

    def disturbance(self):
        moved = turned = 0.0
        for name, (xy, yaw) in self.placed_pose.items():
            now_xy, now_yaw = self.sim.pose(name)
            moved += float(np.linalg.norm(now_xy - xy))
            turned += abs(wrap(now_yaw - yaw))
        return moved, turned

    def disturbed(self):
        for name, (xy, yaw) in self.placed_pose.items():
            now_xy, now_yaw = self.sim.pose(name)
            if np.linalg.norm(now_xy - xy) > DISTURB_LIMIT_M or abs(wrap(now_yaw - yaw)) > DISTURB_LIMIT_RAD:
                return True
        return False

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after an episode ends.")
        sim = self.sim
        success, reason = sim.act(action)
        self.steps += 1
        if not reason and self.disturbed():
            reason = "disturbed"
        success = success and not reason
        distance, angle = sim.distance(), sim.angle_error()
        moved, turned = self.disturbance()
        reward = (POSITION_PROGRESS * (self.previous[0] - distance) + ANGLE_PROGRESS * (self.previous[1] - angle)
                  - DISTURB_POSITION * (moved - self.previous[2]) - DISTURB_ANGLE * (turned - self.previous[3])
                  + (SUCCESS_BONUS if success else 0.0) - (FAILURE_PENALTY if reason else 0.0))
        self.previous = (distance, angle, moved, turned)
        terminated = success or bool(reason)
        truncated = self.steps >= MAX_STEPS and not terminated
        if success:
            reason = "success"
        elif truncated:
            reason = "timeout"
        self.finished = terminated or truncated
        return observation(sim), float(reward), terminated, truncated, self.info(success, reason)

    def info(self, success, reason):
        return dict(is_success=bool(success), reason=reason, steps=self.steps, kind=self.state["kind"],
                    position_error_cm=100 * self.sim.distance(), angle_error_deg=float(np.rad2deg(self.sim.angle_error())))


class MixedEnv(gym.Env):
    """75 % recorded word states, 25 % single-letter tasks (so the policy keeps its single-letter skill)."""
    SINGLE_FRACTION = 0.25

    def __init__(self, letter, states_path, task_seed=None):
        self.word = WordStateEnv(letter, states_path, task_seed)
        self.single = LetterEnv(letter, task_seed=TRAIN_SEEDS[letter] + 5000 + (task_seed or 0))
        self.action_space, self.observation_space = self.word.action_space, self.word.observation_space
        self.rng = np.random.default_rng(task_seed)
        self.active = self.word

    def reset(self, *, seed=None, options=None):
        self.active = self.single if self.rng.random() < self.SINGLE_FRACTION else self.word
        obs, info = self.active.reset(seed=seed, options=options)
        info["kind"] = info.get("kind", "single")
        return obs, info

    def step(self, action):
        return self.active.step(action)
