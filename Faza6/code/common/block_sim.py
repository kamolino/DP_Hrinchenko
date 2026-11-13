import mujoco
import numpy as np

from common import settings as cfg
from common.controller import TCPController
from common.geometry import block_corners, fits_workspace, rot, wrap


class BlockSim:
    """MuJoCo scene with the robot, the table and one rectangular block."""

    def __init__(self, scene=cfg.SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.controller = TCPController(self.model)
        self.object_body = self.model.body("object").id
        self.object_qpos = self.model.joint("object_joint").qposadr[0]
        self.object_dof = self.model.joint("object_joint").dofadr[0]
        self.goal_mocap = self.model.body("goal").mocapid[0]
        self.block_geom = self.model.geom("block").id
        self.table_geom = self.model.geom("table").id
        self.finger_geoms = {
            i for i in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[i]).name in ("left_finger", "right_finger")
            and self.model.geom_contype[i] != 0
        }
        self.is_finger = np.zeros(self.model.ngeom, dtype=bool)
        self.is_finger[list(self.finger_geoms)] = True
        self.goal, self.goal_yaw = np.array([0.5, 0.0]), 0.0
        self.applied_xy, self.last_action = np.zeros(2), np.zeros(4)
        self.steps = self.hold_steps = 0
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.47, 0.0, 0.22]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.9, 150, -45

    @property
    def tcp(self):
        return self.data.site_xpos[self.controller.site].copy()

    @property
    def object_xy(self):
        return self.data.qpos[self.object_qpos:self.object_qpos + 2].copy()

    @property
    def yaw(self):
        w, x, y, z = self.data.qpos[self.object_qpos + 3:self.object_qpos + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def distance(self):
        return float(np.linalg.norm(self.object_xy - self.goal))

    def angle_error(self):
        return abs(wrap(self.goal_yaw - self.yaw))

    def keypoint_error(self):
        current = block_corners(self.object_xy, self.yaw, cfg.KEYPOINT_SCALE)
        target = block_corners(self.goal, self.goal_yaw, cfg.KEYPOINT_SCALE)
        return float(np.linalg.norm(current - target, axis=1).mean())

    def reset(self, task, start_tcp=cfg.TCP_START):
        start_tcp = np.array(start_tcp, dtype=float)
        start_q = self.controller.start_configuration(start_tcp)
        obj, goal = np.asarray(task["object"], float), np.asarray(task["goal"], float)
        yaw, goal_yaw = float(task["yaw"]), float(task["goal_yaw"])
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.qpos] = start_q
        self.data.ctrl[self.controller.actuators] = start_q
        self.data.qpos[self.object_qpos:self.object_qpos + 7] = [
            *obj, cfg.TABLE_TOP + cfg.OBJECT_HALF[2], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.goal, self.goal_yaw = goal.copy(), goal_yaw
        self.data.mocap_pos[self.goal_mocap] = [*goal, cfg.TABLE_TOP + 0.002]
        self.data.mocap_quat[self.goal_mocap] = [np.cos(goal_yaw / 2), 0, 0, np.sin(goal_yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=150)
        mujoco.mj_forward(self.model, self.data)
        self.controller.target = start_tcp.copy()
        self.controller.yaw = 0.0
        self.data.time = 0.0
        self.applied_xy[:] = 0
        self.last_action[:] = 0
        self.steps = self.hold_steps = 0
        reason = self.reset_failure()
        if reason:
            raise RuntimeError(f"Invalid reset: {reason}")

    def reset_failure(self):
        return self.failure_reason()

    def apply(self, xy, z):
        xy = np.clip(np.asarray(xy, dtype=float), -1, 1)
        self.applied_xy = (1 - cfg.XY_FILTER_ALPHA) * self.applied_xy + cfg.XY_FILTER_ALPHA * xy
        self.controller.command(self.data, np.r_[self.applied_xy, np.clip(z, -1, 1), 0.0])
        self.last_action = np.r_[xy, z, 0.0]

    def simulate(self, reached):
        """Physics for one control step. Returns (held at the goal for 0.3 s, contact failure)."""
        success, contact = False, ""
        for _ in range(cfg.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            contact = self.bad_push_contact()
            if contact:
                break
            self.hold_steps = self.hold_steps + 1 if reached() else 0
            if self.hold_steps * cfg.TIMESTEP >= cfg.SUCCESS_HOLD:
                success = True
                break
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        return success, contact

    def at_goal(self):
        return self.distance() < cfg.SUCCESS_DISTANCE and self.angle_error() < cfg.SUCCESS_ANGLE

    def act(self, xy, z):
        """One control step with the task success test (2 cm and 10 deg held 0.3 s)."""
        self.apply(xy, z)
        success, _ = self.simulate(self.at_goal)
        reason = self.failure_reason()
        return success and not reason, reason

    def observation(self):
        """40 values in the goal frame (input of the rotation policy)."""
        to_goal = rot(self.goal_yaw).T
        to_object = rot(self.yaw).T
        tcp = self.tcp
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.controller.site, velocity, 0)
        current = block_corners(self.object_xy, self.yaw, cfg.KEYPOINT_SCALE)
        target = block_corners(self.goal, self.goal_yaw, cfg.KEYPOINT_SCALE)
        relative_yaw = wrap(self.yaw - self.goal_yaw)
        wrist = self.controller.measured_yaw(self.data)
        object_velocity = self.data.qvel[self.object_dof:self.object_dof + 2]
        obs = np.r_[
            to_goal @ (self.object_xy - self.goal) / cfg.POSITION_SCALE,
            np.cos(relative_yaw), np.sin(relative_yaw),
            np.cos(self.yaw), np.sin(self.yaw), np.cos(self.goal_yaw), np.sin(self.goal_yaw),
            ((current - target) @ to_goal.T).ravel() / cfg.POSITION_SCALE,
            to_object @ (tcp[:2] - self.object_xy) / cfg.POSITION_SCALE,
            (tcp[2] - cfg.TCP_HEIGHT) / 0.05,
            to_goal @ (tcp[:2] - self.goal) / cfg.POSITION_SCALE,
            np.cos(wrist - self.yaw), np.sin(wrist - self.yaw), wrist,
            to_goal @ object_velocity / cfg.VELOCITY_SCALE,
            self.data.qvel[self.object_dof + 5] / cfg.ANGULAR_VELOCITY_SCALE,
            velocity[3:6] / cfg.VELOCITY_SCALE,
            (tcp - [0.47, 0.0, cfg.TCP_HEIGHT]) / [0.3, 0.3, 0.1],
            (self.controller.target - tcp) / 0.02,
            self.applied_xy, self.last_action[2:],
        ]
        return obs.astype(np.float32)

    def bad_push_contact(self):
        contacts = self.data.contact
        first, second = contacts.geom1, contacts.geom2
        touching = ((first == self.block_geom) & self.is_finger[second]) | \
                   ((second == self.block_geom) & self.is_finger[first])
        for index in np.flatnonzero(touching & (contacts.dist < 0)):
            if abs(contacts.frame[index, 2]) > 0.7:
                return "top_contact"
            if self.tcp[2] > cfg.TCP_HEIGHT + 0.006:
                return "high_contact"
        return ""

    def forbidden_contact(self):
        allowed = [{self.block_geom, self.table_geom}]
        allowed.extend({self.block_geom, finger} for finger in self.finger_geoms)
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if contact.dist < -0.001 and pair not in allowed:
                return True
        return False

    def outside(self):
        return not fits_workspace(self.object_xy, self.yaw)

    def failure_reason(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "invalid_state"
        z = self.data.qpos[self.object_qpos + 2]
        if z < cfg.TABLE_TOP - 0.01:
            return "object_fell"
        if self.outside():
            return "object_outside"
        if z > cfg.TABLE_TOP + 0.08:
            return "object_lifted"
        if self.data.xmat[self.object_body].reshape(3, 3)[2, 2] < np.cos(np.deg2rad(20)):
            return "object_tilted"
        reason = self.bad_push_contact()
        if reason:
            return reason
        if self.forbidden_contact():
            return "forbidden_contact"
        q = self.data.qpos[self.controller.qpos]
        limits = self.model.jnt_range[self.controller.joints]
        if np.any(q < limits[:, 0] - 0.01) or np.any(q > limits[:, 1] + 0.01):
            return "joint_limit"
        if not cfg.TCP_Z_LOW - 0.025 < self.tcp[2] < cfg.TCP_Z_HIGH + 0.025:
            return "tool_height"
        return ""

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
