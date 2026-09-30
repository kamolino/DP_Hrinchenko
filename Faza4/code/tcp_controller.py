import mujoco
import numpy as np

from settings import ACTION_SCALE, ACTION_SCALE_Z, HOME, TCP_START, TCP_LOW, TCP_HIGH, TCP_Z_LOW, TCP_Z_HIGH


class TCPController:
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
        self.target = np.array(TCP_START)
        self.rotation = np.diag([1.0, -1.0, -1.0])
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))

    def solve(self, target, q, iterations=5):
        data = self.ik_data
        data.qpos[self.qpos] = q
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            rotation = data.site_xmat[self.site].reshape(3, 3)
            angle_error = 0.5 * sum(np.cross(rotation[:, i], self.rotation[:, i]) for i in range(3))
            error = np.r_[target - data.site_xpos[self.site], angle_error]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(self.model, data, self.jac_pos, self.jac_rot, self.site)
            jacobian = np.vstack([self.jac_pos[:, self.dofs], self.jac_rot[:, self.dofs]])
            change = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(6), error)
            data.qpos[self.qpos] = np.clip(data.qpos[self.qpos] + np.clip(change, -0.12, 0.12), self.low, self.high)
        return data.qpos[self.qpos].copy()

    def command(self, data, action):
        self.target[:2] = np.clip(self.target[:2] + ACTION_SCALE * action[:2], TCP_LOW, TCP_HIGH)
        self.target[2] = np.clip(self.target[2] + ACTION_SCALE_Z * action[2], TCP_Z_LOW, TCP_Z_HIGH)
        # Limit tracking error if contact blocks the tool.
        actual = data.site_xpos[self.site]
        self.target = np.clip(self.target, actual - 0.02, actual + 0.02)
        data.ctrl[self.actuators] = self.solve(self.target, data.qpos[self.qpos])

    def start_configuration(self, target):
        q = self.solve(np.array(target), np.array(HOME), iterations=150)
        self.solve(np.array(target), q)
        position = self.ik_data.site_xpos[self.site]
        rotation = self.ik_data.site_xmat[self.site].reshape(3, 3)
        if np.linalg.norm(position - target) > 0.001 or np.linalg.norm(rotation - self.rotation) > 0.02:
            raise RuntimeError("The start pose is not reachable with the fixed tool orientation.")
        return q
