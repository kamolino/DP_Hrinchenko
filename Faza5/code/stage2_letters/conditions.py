"""Test conditions for the word: physics, sensing noise, letter order, goal layout, action delay.

Friction scales the letters and the table (MuJoCo uses the larger value of a pair). Noise is added to the
active letter's pose as seen by the policy only. The delay applies every action one control step later.
"""
from dataclasses import dataclass, field

import numpy as np

from stage1_cube.cube_env import observation
from stage2_letters import letters as L
from stage2_letters.word_sim import ORDER

NAMES = ["nominal", "friction_0.7", "friction_1.3", "mass_0.5", "mass_2", "noise_pos_2mm", "noise_pos_5mm",
         "noise_ang_1deg", "noise_ang_3deg", "order_RAM", "order_MRA", "layout_y+5", "layout_y-5", "layout_x-5",
         "layout_gap6", "delay_1"]
_BASE = dict(friction=L.FRICTION, masses=dict(L.MASSES), goals=dict(L.GOALS))


@dataclass
class Setup:
    order: tuple = ORDER
    position_noise: float = 0.0
    angle_noise: float = 0.0
    delay: bool = False
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def observe(self, sim):
        if not (self.position_noise or self.angle_noise):
            return observation(sim)
        adr = sim.qpos[sim.active]
        saved = sim.data.qpos[adr:adr + 7].copy()
        xy, yaw = sim.pose(sim.active)
        yaw = yaw + self.rng.normal(0, self.angle_noise)
        sim.data.qpos[adr:adr + 2] = xy + self.rng.normal(0, self.position_noise, 2)
        sim.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        try:
            return observation(sim)
        finally:
            sim.data.qpos[adr:adr + 7] = saved

    def wrap_sim(self, sim):
        if self.delay:
            act, reset = sim.act, sim.reset
            previous = {"action": np.zeros(3)}

            def delayed(action):
                last, previous["action"] = previous["action"], np.asarray(action, dtype=float)
                return act(last)

            def reset_and_clear(starts):
                previous["action"] = np.zeros(3)
                return reset(starts)

            sim.act, sim.reset = delayed, reset_and_clear
        return sim


def _goals(dx=0.0, dy=0.0, spacing=None):
    for n, (x, y, yaw) in _BASE["goals"].items():
        if spacing is not None:
            y = {"A": spacing, "R": 0.0, "M": -spacing}[n]
        L.GOALS[n] = (x + dx, y + dy, yaw)


def apply(name):
    """Set the module-level scene parameters for `name` and return the per-episode settings."""
    if name not in NAMES:
        raise ValueError(f"unknown condition {name}")
    L.FRICTION = _BASE["friction"]
    L.MASSES.update(_BASE["masses"])
    L.GOALS.update(_BASE["goals"])
    setup = Setup()
    if name.startswith("friction_"):
        parts = L.FRICTION.split()
        L.FRICTION = " ".join([f"{float(parts[0]) * float(name.split('_')[1]):.4f}", *parts[1:]])
    elif name.startswith("mass_"):
        for n in L.MASSES:
            L.MASSES[n] *= float(name.split("_")[1])
    elif name.startswith("noise_pos_"):
        setup.position_noise = float(name.split("_")[2][:-2]) / 1000
    elif name.startswith("noise_ang_"):
        setup.angle_noise = np.deg2rad(float(name.split("_")[2][:-3]))
    elif name.startswith("order_"):
        setup.order = tuple(name.split("_")[1])
    elif name == "layout_y+5":
        _goals(dy=0.05)
    elif name == "layout_y-5":
        _goals(dy=-0.05)
    elif name == "layout_x-5":
        _goals(dx=-0.05)
    elif name == "layout_gap6":
        _goals(spacing=0.12)
    elif name == "delay_1":
        setup.delay = True
    return setup
