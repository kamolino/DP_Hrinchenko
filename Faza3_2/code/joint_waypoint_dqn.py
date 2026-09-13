from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
import torch
from torch import nn
from precision_environment import PrecisionEnv, PrecisionConfig

# ============================================================
# 1. ROBOT ENVIRONMENT
# ============================================================


class WorkspaceEnv(PrecisionEnv):
    def valid_pose(self, q):
        q = np.asarray(q)
        if (
            q.shape != (7,)
            or not np.isfinite(q).all()
            or np.any(q < self.low)
            or np.any(q > self.high)
        ):
            return None
        mujoco.mj_resetDataKeyframe(self.model, self.scratch, self.ids.home_key_id)
        self.scratch.qpos[self.ids.qpos_ids] = q
        mujoco.mj_forward(self.model, self.scratch)
        tcp_position = self.scratch.site_xpos[self.ids.ee_site_id].copy()
        return (
            tcp_position
            if tcp_position[2] >= 0.03 and not self.collision(self.scratch)
            else None
        )


# ============================================================
# 2. DQN NETWORK
# ============================================================


class JointActionDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def action(self, x):
        with torch.no_grad():
            return int(self(torch.as_tensor(x)).argmax())


# ============================================================
# 3. ACTIONS AND STATE
# ============================================================


def joint_commands(env, target_q):
    current_q = env.data.qpos[env.ids.qpos_ids]
    control = env.data.ctrl[env.ids.actuator_ids]
    error = target_q - current_q
    step = np.clip(0.65 * np.abs(error), 0.0003, 0.035)
    commands = np.repeat(control[None, :], 15, axis=0)
    for joint_index in range(7):
        commands[1 + 2 * joint_index, joint_index] += step[joint_index]
        commands[2 + 2 * joint_index, joint_index] -= step[joint_index]
    return np.clip(commands, env.low, env.high)


def joint_observation(env, target_q):
    current_q = env.data.qpos[env.ids.qpos_ids]
    control = env.data.ctrl[env.ids.actuator_ids]
    velocity = env.data.qvel[env.ids.dof_ids]
    error = target_q - current_q
    distance = np.linalg.norm(error)
    scale = max(distance, 0.001)
    lag = control - current_q
    commands = joint_commands(env, target_q)
    features = []
    for action_index in range(15):
        joint_index = (action_index - 1) // 2 if action_index else 0
        delta = (
            commands[action_index, joint_index] - control[joint_index]
            if action_index
            else 0
        )
        margin = min(
            commands[action_index, joint_index] - env.low[joint_index],
            env.high[joint_index] - commands[action_index, joint_index],
        )
        features.append(
            [
                (2 * error[joint_index] * delta - delta * delta) / scale**2,
                delta / scale,
                error[joint_index] / scale,
                lag[joint_index] / scale,
                velocity[joint_index] * env.dt / scale,
                np.dot(error, lag) / scale**2,
                np.linalg.norm(lag) / scale,
                np.linalg.norm(velocity) * env.dt / scale,
                distance,
                np.log10(max(distance, 1e-6)) / 6,
                float(action_index == 0),
                margin,
            ]
        )
    return np.clip(np.array(features, dtype=np.float32), -10, 10)


# ============================================================
# 4. TRAINING EXAMPLES
# ============================================================


class JointTeacher:
    def __init__(self, env):
        self.env = env
        self.data = mujoco.MjData(env.model)

    def scores(self, target):
        env = self.env
        distance_before = np.linalg.norm(target - env.data.qpos[env.ids.qpos_ids])
        action_scores = []
        for command in joint_commands(env, target):
            mujoco.mj_copyData(self.data, env.model, env.data)
            self.data.ctrl[env.ids.actuator_ids] = command
            collision = False
            for step_index in range(75):
                mujoco.mj_step(env.model, self.data)
                if step_index % 25 == 0:
                    collision |= env.collision(self.data)
            distance_after = np.linalg.norm(target - self.data.qpos[env.ids.qpos_ids])
            action_scores.append(
                4 * (distance_before - distance_after) / max(distance_before, 0.001)
                - (5 if collision else 0)
            )
        return np.array(action_scores, dtype=np.float32)


# ============================================================
# 5. APPLYING AN ACTION
# ============================================================


def apply_joint_action(env, target, action):
    commands = joint_commands(env, target)
    # Run the physics and goal checks with the selected joint command.
    env.data.ctrl[env.ids.actuator_ids] = commands[action]
    return env.step(0)


# ============================================================
# 6. SAVING AND LOADING
# ============================================================


def save_joint(path, net, stage, config, **metadata):
    torch.save(
        dict(
            version="joint_waypoint_v1",
            state_dict=net.state_dict(),
            stage=stage,
            algorithm="Offline demonstration-augmented Double DQN",
            env_config=asdict(config),
            metadata=metadata,
        ),
        path,
    )


def load_joint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["version"] != "joint_waypoint_v1":
        raise ValueError("Wrong joint policy checkpoint")
    network = JointActionDQN()
    network.load_state_dict(data["state_dict"])
    network.eval()
    return network, data
