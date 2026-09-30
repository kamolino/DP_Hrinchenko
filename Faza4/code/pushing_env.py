import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import settings as cfg
from tcp_controller import TCPController


class PushingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, stage="C", render_mode=None):
        if stage not in ("A", "B", "C"):
            raise ValueError("Stage must be A, B or C.")
        self.stage = stage
        self.curriculum = 1.0
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(cfg.SCENE))
        self.data = mujoco.MjData(self.model)
        self.controller = TCPController(self.model)
        self.object_body = self.model.body("object").id
        self.object_qpos = self.model.joint("object_joint").qposadr[0]
        self.object_dof = self.model.joint("object_joint").dofadr[0]
        self.goal_mocap = self.model.body("goal").mocapid[0]
        self.cube_geom = self.model.geom("cube").id
        self.finger_geoms = {
            i for i in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[i]).name in ("left_finger", "right_finger")
            and self.model.geom_contype[i] != 0
        }
        self.is_finger = np.zeros(self.model.ngeom, dtype=bool)
        self.is_finger[list(self.finger_geoms)] = True
        self.table_geom = self.model.geom("table").id
        self.floor_geom = self.model.geom("floor").id
        # Position servos support the arm against gravity.
        self.start_q = self.controller.start_configuration(cfg.TCP_START)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.43, 0.0, 0.30]
        self.camera.distance = 1.0
        self.camera.azimuth = 135
        self.camera.elevation = -35
        self.goal = np.array(cfg.GOAL_START)
        self.steps = 0
        self.hold_steps = 0
        self.finished = False

    @property
    def tcp(self):
        return self.data.site_xpos[self.controller.site].copy()

    @property
    def object_position(self):
        return self.data.qpos[self.object_qpos:self.object_qpos + 3].copy()

    def set_curriculum(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Curriculum fraction must be between zero and one.")
        self.curriculum = float(fraction)

    def sample_task(self):
        obj = np.array(cfg.OBJECT_START)
        if self.stage == "C":
            centre = np.array(cfg.OBJECT_START)
            low = centre + self.curriculum * (np.array(cfg.OBJECT_LOW) - centre)
            high = centre + self.curriculum * (np.array(cfg.OBJECT_HIGH) - centre)
            obj = self.np_random.uniform(low, high)
        goal = np.array(cfg.GOAL_START)
        if self.stage != "A":
            for _ in range(1000):
                low = np.array(cfg.GOAL_LOW, dtype=float)
                high = np.array(cfg.GOAL_HIGH, dtype=float)
                if self.stage == "B":
                    low = np.array([0.505, -0.015]) + self.curriculum * (low - [0.505, -0.015])
                    high = np.array([0.555, 0.015]) + self.curriculum * (high - [0.555, 0.015])
                goal = self.np_random.uniform(low, high)
                if np.linalg.norm(goal - obj) >= 0.06:
                    break
            else:
                raise RuntimeError("Could not sample a valid goal.")
        return obj, goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obj, goal = self.sample_task()
        if options is not None:
            obj = np.asarray(options.get("object", obj), dtype=float)
            goal = np.asarray(options.get("goal", goal), dtype=float)
        if obj.shape != (2,) or goal.shape != (2,) or not np.isfinite(np.r_[obj, goal]).all():
            raise ValueError("Object and goal must be finite XY positions.")
        if np.linalg.norm(goal - obj) <= cfg.SUCCESS_DISTANCE:
            raise ValueError("Object cannot start inside the goal.")
        if np.linalg.norm(obj - np.array(cfg.TCP_START[:2])) < 0.045:
            raise ValueError("Object overlaps the tool at reset.")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.qpos] = self.start_q
        self.data.ctrl[self.controller.actuators] = self.start_q
        self.data.qpos[self.object_qpos:self.object_qpos + 7] = [*obj, cfg.TABLE_TOP + cfg.CUBE_HALF, 1, 0, 0, 0]
        self.goal = goal.copy()
        self.data.mocap_pos[self.goal_mocap] = [*goal, cfg.TABLE_TOP + 0.001]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=150)
        mujoco.mj_forward(self.model, self.data)
        self.controller.target = np.array(cfg.TCP_START)
        self.data.time = 0.0
        self.steps = 0
        self.hold_steps = 0
        self.finished = False
        self.previous_distance = self.distance()
        self.previous_approach = self.approach_distance()
        reason = self.failure_reason()
        if reason:
            raise RuntimeError(f"Invalid reset: {reason}")
        return self.observation(), self.info(False, "")

    def distance(self):
        return float(np.linalg.norm(self.object_position[:2] - self.goal))

    def approach_distance(self):
        direction = self.goal - self.object_position[:2]
        direction /= max(np.linalg.norm(direction), 1e-6)
        behind = self.object_position[:2] - cfg.APPROACH_OFFSET * direction
        return float(np.linalg.norm(self.tcp - np.r_[behind, cfg.TCP_HEIGHT]))

    def high_approach_cost(self):
        xy_distance = np.linalg.norm(self.tcp[:2] - self.object_position[:2])
        nearby = max(0.0, 1.0 - xy_distance / cfg.HIGH_APPROACH_RADIUS)
        above = np.clip((self.tcp[2] - cfg.TCP_HEIGHT) / cfg.HIGH_APPROACH_SCALE, 0.0, 1.0)
        return -cfg.HIGH_APPROACH_WEIGHT * nearby * float(above)

    def observation(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.controller.site, velocity, 0)
        rotation = self.data.xmat[self.object_body].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        obs = np.r_[
            (self.object_position - self.tcp) / cfg.POSITION_SCALE,
            (self.goal - self.object_position[:2]) / cfg.POSITION_SCALE,
            self.data.qvel[self.object_dof:self.object_dof + 3] / cfg.VELOCITY_SCALE,
            velocity[3:6] / cfg.VELOCITY_SCALE,
            (self.controller.target - self.tcp) / 0.02,
            (self.tcp - [0.47, 0.0, cfg.TCP_HEIGHT]) / [cfg.POSITION_SCALE, cfg.POSITION_SCALE, 0.1],
            np.cos(4 * yaw), np.sin(4 * yaw),
        ]
        return obs.astype(np.float32)

    def forbidden_contact(self):
        allowed = [{self.cube_geom, self.table_geom}]
        allowed.extend({self.cube_geom, finger} for finger in self.finger_geoms)
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if contact.dist < -0.001 and pair not in allowed:
                return True
        return False

    def bad_push_contact(self):
        contacts = self.data.contact
        first, second = contacts.geom1, contacts.geom2
        touching = ((first == self.cube_geom) & self.is_finger[second]) | \
                   ((second == self.cube_geom) & self.is_finger[first])
        for index in np.flatnonzero(touching & (contacts.dist < 0)):
            if abs(contacts.frame[index, 2]) > 0.7:
                return "top_contact"
            if self.tcp[2] > cfg.TCP_HEIGHT + 0.006:
                return "high_contact"
        return ""

    def failure_reason(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "invalid_state"
        obj = self.object_position
        if obj[2] < cfg.TABLE_TOP - 0.01:
            return "object_fell"
        if not (0.24 < obj[0] < 0.76 and abs(obj[1]) < 0.25):
            return "object_outside"
        if obj[2] > cfg.TABLE_TOP + 0.08:
            return "object_lifted"
        contact_reason = self.bad_push_contact()
        if contact_reason:
            return contact_reason
        if self.forbidden_contact():
            return "forbidden_contact"
        q = self.data.qpos[self.controller.qpos]
        limits = self.model.jnt_range[self.controller.joints]
        if np.any(q < limits[:, 0] - 0.01) or np.any(q > limits[:, 1] + 0.01):
            return "joint_limit"
        if not cfg.TCP_Z_LOW - 0.025 < self.tcp[2] < cfg.TCP_Z_HIGH + 0.025:
            return "tool_height"
        return ""

    def info(self, success, reason):
        return dict(is_success=bool(success), reason=reason, object_error=self.distance(),
                    steps=self.steps, sim_time=float(self.data.time),
                    hold_time=self.hold_steps * cfg.TIMESTEP)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after an episode ends.")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Action must contain three finite numbers.")
        action = np.clip(action, -1, 1)
        self.controller.command(self.data, action)
        success = False
        reason = ""
        for _ in range(cfg.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            reason = self.bad_push_contact()
            if reason:
                break
            # qpos is current after integration, so use it for the hold timer.
            self.hold_steps = self.hold_steps + 1 if self.distance() < cfg.SUCCESS_DISTANCE else 0
            if self.hold_steps * cfg.TIMESTEP >= cfg.SUCCESS_HOLD:
                success = True
                break
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        reason = reason or self.failure_reason()
        if reason:
            success = False
        distance = self.distance()
        progress = cfg.PROGRESS_WEIGHT * (self.previous_distance - distance)
        approach = self.approach_distance()
        position_cost = -cfg.APPROACH_DISTANCE_WEIGHT * approach
        effort = -cfg.ACTION_WEIGHT * np.square(action).sum()
        bonus = cfg.SUCCESS_BONUS if success else 0.0
        failure = -cfg.FAILURE_PENALTY if reason else 0.0
        goal_cost = -cfg.DISTANCE_WEIGHT * distance
        approach_reward = cfg.APPROACH_WEIGHT * (self.previous_approach - approach)
        high_approach = self.high_approach_cost()
        self.previous_approach = approach
        reward = cfg.REWARD_SCALE * float(progress + position_cost + goal_cost + effort + approach_reward + bonus + failure + high_approach)
        self.previous_distance = distance
        terminated = success or bool(reason)
        truncated = self.steps >= cfg.MAX_STEPS and not terminated
        if success:
            reason = "success"
        elif truncated:
            reason = "timeout"
        self.finished = terminated or truncated
        info = self.info(success, reason)
        parts = dict(progress=float(progress), position=float(position_cost), effort=float(effort),
                     goal_cost=float(goal_cost), approach=float(approach_reward), bonus=bonus, failure=failure, high_approach=float(high_approach))
        info["reward_parts"] = {name: cfg.REWARD_SCALE * value for name, value in parts.items()}
        return self.observation(), reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=960)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
