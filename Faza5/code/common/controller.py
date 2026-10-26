import mujoco
import numpy as np

from common import settings as cfg

BASE_ROTATION = np.diag([1.0, -1.0, -1.0])


def wrist_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ BASE_ROTATION


class TCPController:
    """Moves a TCP target and solves damped least-squares IK for the 7 joints."""

    def __init__(self, model):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.joints = np.array([model.joint(f"fr3_joint{i}").id for i in range(1, 8)])
        self.qpos = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        self.actuators = np.array([model.actuator(f"fr3_joint{i}").id for i in range(1, 8)])
        self.site = model.site("tcp").id
        self.low = model.jnt_range[self.joints, 0] + 0.04
        self.high = model.jnt_range[self.joints, 1] - 0.04
        self.target = np.array(cfg.TCP_START, dtype=float)
        self.yaw = 0.0
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))

    def solve(self, target, yaw, q, iterations=5):
        data = self.ik_data
        data.qpos[self.qpos] = q
        desired = wrist_rotation(yaw)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            rotation = data.site_xmat[self.site].reshape(3, 3)
            angle_error = 0.5 * sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3))
            error = np.r_[target - data.site_xpos[self.site], angle_error]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(self.model, data, self.jac_pos, self.jac_rot, self.site)
            jacobian = np.vstack([self.jac_pos[:, self.dofs], self.jac_rot[:, self.dofs]])
            change = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(6), error)
            data.qpos[self.qpos] = np.clip(data.qpos[self.qpos] + np.clip(change, -0.12, 0.12), self.low, self.high)
        return data.qpos[self.qpos].copy()

    def measured_yaw(self, data):
        rotation = data.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def command(self, data, action):
        """action = [x, y, z, wrist yaw] in [-1, 1]."""
        self.target[:2] = np.clip(self.target[:2] + cfg.ACTION_SCALE * action[:2], cfg.TCP_LOW, cfg.TCP_HIGH)
        self.target[2] = np.clip(self.target[2] + cfg.ACTION_SCALE_Z * action[2], cfg.TCP_Z_LOW, cfg.TCP_Z_HIGH)
        self.yaw = float(np.clip(self.yaw + cfg.ACTION_SCALE_YAW * action[3],
                                 -cfg.WRIST_YAW_LIMIT, cfg.WRIST_YAW_LIMIT))
        # keep the target close to the tool when a contact blocks it
        actual = data.site_xpos[self.site]
        self.target = np.clip(self.target, actual - 0.02, actual + 0.02)
        actual_yaw = self.measured_yaw(data)
        self.yaw = float(np.clip(self.yaw, actual_yaw - 0.2, actual_yaw + 0.2))
        data.ctrl[self.actuators] = self.solve(self.target, self.yaw, data.qpos[self.qpos])

    def start_configuration(self, target, yaw=0.0):
        q = self.solve(np.array(target), yaw, np.array(cfg.HOME), iterations=150)
        self.solve(np.array(target), yaw, q)
        position = self.ik_data.site_xpos[self.site]
        rotation = self.ik_data.site_xmat[self.site].reshape(3, 3)
        if np.linalg.norm(position - target) > 0.001 or np.linalg.norm(rotation - wrist_rotation(yaw)) > 0.02:
            raise RuntimeError("The start pose is not reachable.")
        return q
