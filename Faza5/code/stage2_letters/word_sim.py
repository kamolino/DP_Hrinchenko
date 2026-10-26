"""Scene with all three letters. The active letter is the one the policies observe; the others are obstacles."""
import os

import mujoco
import numpy as np

from common import settings as cfg
from common.controller import TCPController
from common.geometry import wrap
from stage2_letters import letters as L

ORDER = ("A", "R", "M")
MOVE_STEPS = 60


class WordSim:
    def __init__(self):
        path = L.word_scene(L.GENERATED / f"word_{os.getpid()}.xml")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        path.unlink()
        self.data = mujoco.MjData(self.model)
        self.controller = TCPController(self.model)
        self.table_geom = self.model.geom("table").id
        name_of = lambda g: self.model.body(self.model.geom_bodyid[g]).name  # noqa: E731
        self.letter_of = np.array([name_of(g) if name_of(g) in L.NAMES else "" for g in range(self.model.ngeom)])
        self.is_letter = self.letter_of != ""
        self.is_finger = np.array([name_of(g) in ("left_finger", "right_finger") and self.model.geom_contype[g] != 0
                                   for g in range(self.model.ngeom)])
        self.qpos = {n: self.model.joint(f"{n}_joint").qposadr[0] for n in L.NAMES}
        self.dof = {n: self.model.joint(f"{n}_joint").dofadr[0] for n in L.NAMES}
        self.body = {n: self.model.body(n).id for n in L.NAMES}
        self.start_q = self.controller.start_configuration(cfg.TCP_START)
        self.applied_xy, self.last_action = np.zeros(2), np.zeros(4)
        self.steps = self.hold_steps = 0
        self.contact_letter = ""
        self.order = ORDER
        self.activate("A")

    def activate(self, name):
        self.active = name
        self.object_qpos, self.object_dof, self.object_body = self.qpos[name], self.dof[name], self.body[name]
        self.goal, self.goal_yaw = np.array(L.GOALS[name][:2]), float(L.GOALS[name][2])

    @property
    def tcp(self):
        return self.data.site_xpos[self.controller.site].copy()

    def pose(self, name):
        q = self.data.qpos[self.qpos[name]:self.qpos[name] + 7]
        w, x, y, z = q[3:]
        return q[:2].copy(), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def object_xy(self):
        return self.pose(self.active)[0]

    @property
    def yaw(self):
        return self.pose(self.active)[1]

    def errors(self, name):
        xy, yaw = self.pose(name)
        return float(np.linalg.norm(xy - L.GOALS[name][:2])), abs(wrap(L.GOALS[name][2] - yaw))

    def distance(self):
        return float(np.linalg.norm(self.object_xy - self.goal))

    def angle_error(self):
        return abs(wrap(self.goal_yaw - self.yaw))

    def reset(self, starts):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.qpos] = self.start_q
        self.data.ctrl[self.controller.actuators] = self.start_q
        for name in self.order:
            x, y, yaw = starts[name]
            self.data.qpos[self.qpos[name]:self.qpos[name] + 7] = [
                x, y, L.TABLE_TOP + L.HALF_HEIGHT, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=150)
        mujoco.mj_forward(self.model, self.data)
        self.controller.target = np.array(cfg.TCP_START, dtype=float)
        self.controller.yaw = 0.0
        self.applied_xy[:] = 0
        self.last_action[:] = 0
        self.steps = 0

    def act(self, action):
        """One control step; returns (active letter held at its goal for 0.3 s, failure reason)."""
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.applied_xy = (1 - cfg.XY_FILTER_ALPHA) * self.applied_xy + cfg.XY_FILTER_ALPHA * action[:2]
        self.controller.command(self.data, np.r_[self.applied_xy, action[2], 0.0])
        self.last_action = np.r_[action, 0.0]
        success = False
        for _ in range(cfg.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self.bad_push_contact():
                break
            ok = self.distance() < cfg.SUCCESS_DISTANCE and self.angle_error() < cfg.SUCCESS_ANGLE
            self.hold_steps = self.hold_steps + 1 if ok else 0
            if self.hold_steps * cfg.TIMESTEP >= cfg.SUCCESS_HOLD:
                success = True
                break
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        reason = self.failure_reason()
        return success and not reason, reason

    def bad_push_contact(self):
        c = self.data.contact
        g1, g2 = c.geom1, c.geom2
        touching = (self.is_letter[g1] & self.is_finger[g2]) | (self.is_letter[g2] & self.is_finger[g1])
        for i in np.flatnonzero(touching & (c.dist < 0)):
            letter = self.letter_of[g1[i]] or self.letter_of[g2[i]]
            if abs(c.frame[i, 2]) > 0.7:
                self.contact_letter = letter
                return "top_contact"
            if self.tcp[2] > cfg.TCP_HEIGHT + 0.006:
                self.contact_letter = letter
                return "high_contact"
        return ""

    def failure_reason(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "invalid_state"
        for name in self.order:
            q = self.data.qpos[self.qpos[name]:self.qpos[name] + 3]
            if q[2] < cfg.TABLE_TOP - 0.01:
                return f"{name}_fell"
            if q[2] > cfg.TABLE_TOP + 0.08:
                return f"{name}_lifted"
            if not L.on_table(name, (q[0], q[1], self.pose(name)[1])):
                return f"{name}_outside"
            if self.data.xmat[self.body[name]].reshape(3, 3)[2, 2] < np.cos(np.deg2rad(20)):
                return f"{name}_tilted"
        reason = self.bad_push_contact()
        if reason:
            return reason
        for c in self.data.contact[:self.data.ncon]:
            if c.dist >= -0.001:
                continue
            a, b = int(c.geom1), int(c.geom2)
            ok = any(self.is_letter[x] and (y == self.table_geom or self.is_finger[y] or self.is_letter[y])
                     for x, y in ((a, b), (b, a)))
            if not ok:
                return "forbidden_contact"
        q = self.data.qpos[self.controller.qpos]
        limits = self.model.jnt_range[self.controller.joints]
        if np.any(q < limits[:, 0] - 0.01) or np.any(q > limits[:, 1] + 0.01):
            return "joint_limit"
        if not cfg.TCP_Z_LOW - 0.025 < self.tcp[2] < cfg.TCP_Z_HIGH + 0.025:
            return "tool_height"
        return ""

    def move_to_start(self):
        """Lift to the start height, then go to the start point above the letters."""
        reason = ""
        for target in ([*self.tcp[:2], cfg.TCP_START[2]], cfg.TCP_START):
            for _ in range(MOVE_STEPS):
                error = np.array(target) - self.controller.target
                _, reason = self.act(np.clip(error / [cfg.ACTION_SCALE, cfg.ACTION_SCALE, cfg.ACTION_SCALE_Z], -1, 1))
                if reason:
                    return reason
        self.applied_xy[:] = 0
        return reason
