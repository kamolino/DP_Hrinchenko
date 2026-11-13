"""Stateless version of the stage-1 manager. It labels any state, so it can teach one policy (DAgger)."""
import numpy as np

from common import policy, settings as cfg
from stage1_cube.manager import PUSH_MODEL, approach_action, load_rotation
from stage1_cube.push_model import push_observation
from stage1_cube.rotate_env import decode

ROTATE_ABOVE = np.deg2rad(35)


class Teacher:
    def __init__(self):
        self.rot, self.rot_norm = load_rotation()
        self.push = policy.load(PUSH_MODEL)

    @staticmethod
    def ready_to_rotate(sim):
        local = np.array([[np.cos(sim.yaw), np.sin(sim.yaw)], [-np.sin(sim.yaw), np.cos(sim.yaw)]]) @ (sim.tcp[:2] - sim.object_xy)
        outside = np.linalg.norm(np.maximum(np.abs(local) - np.array(cfg.OBJECT_HALF[:2]), 0.0))
        return sim.tcp[2] < cfg.TCP_HEIGHT + 0.006 and outside < 0.10

    def act(self, env):
        """Action in [-1, 1]^3 and the phase that produced it."""
        sim = env.sim
        if sim.angle_error() > ROTATE_ABOVE:
            if not self.ready_to_rotate(sim):
                xy, near = approach_action(sim)
                return np.r_[np.clip(xy, -1, 1), -1.0 if near else 0.5], "approach"
            action, _ = self.rot.predict(self.rot_norm.normalize_obs(sim.observation()[None])[0], deterministic=True)
            return np.r_[decode(action), -1.0], "rotate"
        action, _ = self.push.predict(push_observation(sim), deterministic=True)
        return np.clip(np.asarray(action, float), -1, 1), "push"
