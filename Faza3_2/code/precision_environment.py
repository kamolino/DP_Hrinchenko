from dataclasses import dataclass, asdict
import math
import mujoco
import numpy as np
from fr3_common import load_model_and_data, resolve_indices

# ============================================================
# 1. ENVIRONMENT PARAMETERS
# ============================================================


@dataclass
class PrecisionConfig:
    tolerance: float = 0.001
    hold_seconds: float = 0.30
    max_tcp_speed: float = 0.005
    frame_skip: int = 25
    max_steps: int = 300
    joint_margin: float = 0.08  # radians, not a fraction of the range
    max_action: float = 0.04
    min_action: float = 0.0004
    gravity_compensation: bool = True

    def __post_init__(self):
        if not (
            0 < self.tolerance <= 0.03
            and self.hold_seconds > 0
            and self.max_tcp_speed > 0
        ):
            raise ValueError("Invalid precision/hold configuration")
        if (
            self.frame_skip < 1
            or self.max_steps < 1
            or not (0 < self.min_action <= self.max_action)
        ):
            raise ValueError("Invalid step configuration")
        if self.joint_margin < 0:
            raise ValueError("joint_margin must be nonnegative")


# ============================================================
# 2. ROBOT ENVIRONMENT
# ============================================================


class PrecisionEnv:
    action_dim = 15
    feature_dim = 19

    def __init__(self, config=None, seed=0):
        self.config = config or PrecisionConfig()
        self.rng = np.random.default_rng(seed)
        self.model, self.data = load_model_and_data()
        self.ids = resolve_indices(self.model)
        if self.config.gravity_compensation:
            self.model.body_gravcomp[1:] = 1
        # These limits exclude the separate gravity compensation force.
        self.model.actuator_forcelimited[self.ids.actuator_ids] = True
        self.model.actuator_forcerange[self.ids.actuator_ids] = np.array(
            [[-87, 87]] * 4 + [[-12, 12]] * 3
        )
        self.low = (
            self.model.jnt_range[self.ids.joint_ids, 0] + self.config.joint_margin
        )
        self.high = (
            self.model.jnt_range[self.ids.joint_ids, 1] - self.config.joint_margin
        )
        if np.any(self.low >= self.high):
            raise ValueError("Joint margin removes joint range")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.ids.home_key_id)
        self.home = self.data.qpos[self.ids.qpos_ids].copy()
        self.scratch = mujoco.MjData(self.model)
        self.jac = np.zeros((3, self.model.nv))
        self.dt = self.config.frame_skip * self.model.opt.timestep
        self.hold_steps = math.ceil(self.config.hold_seconds / self.dt)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
        self.target_id = int(self.model.body_mocapid[bid])
        self.reset()

    def ee(self):
        return self.data.site_xpos[self.ids.ee_site_id].copy()

    def kinematics(self):
        mujoco.mj_jacSite(self.model, self.data, self.jac, None, self.ids.ee_site_id)
        j = self.jac[:, self.ids.dof_ids].copy()
        return j, j @ self.data.qvel[self.ids.dof_ids]

    def collision(self, data=None):
        data = self.data if data is None else data
        # The base is fixed against the floor; ignore base-floor contacts only.
        for c in data.contact:
            if c.dist >= -1e-5:
                continue
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(b))
                for b in (b1, b2)
            }
            if names <= {"world", "fr3_link0"}:
                continue
            return True
        return False

    def valid_pose(self, q):
        if np.any(q < self.low) or np.any(q > self.high):
            return None
        mujoco.mj_resetDataKeyframe(self.model, self.scratch, self.ids.home_key_id)
        self.scratch.qpos[self.ids.qpos_ids] = q
        mujoco.mj_forward(self.model, self.scratch)
        p = self.scratch.site_xpos[self.ids.ee_site_id].copy()
        return p if p[2] > 0.15 and not self.collision(self.scratch) else None

    def sample_task(self, spread=0.35, random_start=False):
        for _ in range(3000):
            start = self.home.copy()
            if random_start:
                start = np.clip(
                    start + self.rng.uniform(-0.45, 0.45, 7), self.low, self.high
                )
            p0 = self.valid_pose(start)
            qgoal = np.clip(
                start + self.rng.uniform(-spread, spread, 7), self.low, self.high
            )
            goal = self.valid_pose(qgoal)
            if (
                p0 is not None
                and goal is not None
                and np.linalg.norm(goal - p0) >= 0.015
            ):
                return {
                    "start_q": start.tolist(),
                    "goal": goal.tolist(),
                    "witness_q": qgoal.tolist(),
                }
        raise RuntimeError("No collision-free reachable task sampled")

    def reset(self, task=None, spread=0.35, random_start=False):
        task = task or self.sample_task(spread, random_start)
        start = np.asarray(task["start_q"], dtype=float)
        goal = np.asarray(task["goal"], dtype=float)
        if start.shape != (7,) or goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("Task requires finite start_q[7], goal[3]")
        if self.valid_pose(start) is None:
            raise ValueError("Invalid start configuration")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.ids.home_key_id)
        self.data.qpos[self.ids.qpos_ids] = start
        self.data.ctrl[self.ids.actuator_ids] = start
        self.goal = goal.copy()
        self.data.mocap_pos[self.target_id] = goal
        mujoco.mj_forward(self.model, self.data)
        # Let the joints settle before starting the episode.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.count = self.hold_count = 0
        self.finished = False
        self.distance = float(np.linalg.norm(self.goal - self.ee()))
        return self.observation()

    def commands(self):
        j, vel = self.kinematics()
        # Adjust the step size using the local Jacobian.
        length = np.linalg.norm(j, axis=0)
        step = np.clip(
            0.65 * self.distance / np.maximum(length, 0.05),
            self.config.min_action,
            self.config.max_action,
        )
        ctrl = self.data.ctrl[self.ids.actuator_ids]
        commands = np.repeat(ctrl[None, :], 15, axis=0)
        for i in range(7):
            commands[1 + 2 * i, i] += step[i]
            commands[2 + 2 * i, i] -= step[i]
        commands = np.clip(commands, self.low, self.high)
        return commands, j, vel

    def observation(self):
        commands, j, vel = self.commands()
        q = self.data.qpos[self.ids.qpos_ids]
        ctrl = self.data.ctrl[self.ids.actuator_ids]
        error = self.goal - self.ee()
        scale = max(self.distance, self.config.tolerance)
        displacement = (commands - ctrl) @ j.T
        lag = j @ (ctrl - q)
        features = []
        for a in range(15):
            effect = displacement[a] / scale
            action_joint = (a - 1) // 2 if a else 0
            direction = (1 if a % 2 else -1) if a else 0
            margin = min(
                commands[a, action_joint] - self.low[action_joint],
                self.high[action_joint] - commands[a, action_joint],
            )
            features.append(
                np.r_[
                    error / scale,
                    effect,
                    vel * self.dt / scale,
                    lag / scale,
                    self.distance,
                    np.log10(max(self.distance, 1e-6)) / 6,
                    float(a == 0),
                    margin,
                    direction * (ctrl[action_joint] - q[action_joint]),
                    self.hold_count / self.hold_steps,
                    self.config.tolerance / scale,
                ]
            )
        return np.clip(np.asarray(features, dtype=np.float32), -10, 10)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode finished; reset before stepping")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 15:
            raise ValueError("Action must be an integer in [0,14]")
        commands, _, _ = self.commands()
        before = self.distance
        self.data.ctrl[self.ids.actuator_ids] = commands[action]
        collided = False
        stable_all = True
        max_speed = 0.0
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
            collided |= self.collision()
            d = float(np.linalg.norm(self.goal - self.ee()))
            _, velocity = self.kinematics()
            speed = float(np.linalg.norm(velocity))
            max_speed = max(max_speed, speed)
            stable_all &= (
                d <= self.config.tolerance and speed <= self.config.max_tcp_speed
            )
        self.distance = float(np.linalg.norm(self.goal - self.ee()))
        self.count += 1
        self.hold_count = self.hold_count + 1 if stable_all and not collided else 0
        success = self.hold_count >= self.hold_steps
        terminated = bool(success or collided or not np.isfinite(self.distance))
        truncated = bool(self.count >= self.config.max_steps and not terminated)
        # Reward progress and subtract the cost of one control step.
        reward = (
            4 * (before - self.distance) / max(before, self.config.tolerance) - 0.04
        )
        reward += 3.0 if success else 0.0
        reward -= 5.0 if collided else 0.0
        self.finished = terminated or truncated
        info = dict(
            distance=self.distance,
            tcp_speed=max_speed,
            success=bool(success),
            collision=bool(collided),
            terminated=terminated,
            truncated=truncated,
            hold_count=self.hold_count,
            steps=self.count,
        )
        return self.observation(), float(reward), self.finished, info
