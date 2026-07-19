import time
import random
import csv
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Resolve every path from the project root, not from the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUJOCO_DIR = PROJECT_ROOT / "mujoco"
MODELS_DIR = PROJECT_ROOT / "models"
DQN_RESULTS_DIR = PROJECT_ROOT / "results" / "dqn"

XML_PATH = MUJOCO_DIR / "reset_pos_stable.xml"
BEST_MODEL_SAVE_PATH = MODELS_DIR / "dqn_2dof_best.pt"
LAST_MODEL_SAVE_PATH = MODELS_DIR / "dqn_2dof_last.pt"
TRAIN_HISTORY_PATH = DQN_RESULTS_DIR / "dqn_train_history.csv"
EVAL_HISTORY_PATH = DQN_RESULTS_DIR / "dqn_eval_history.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DQN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not XML_PATH.exists():
    raise FileNotFoundError(f"MuJoCo XML model not found: {XML_PATH}")


# ============================================================
# 2. ENVIRONMENT PARAMETERS
# ============================================================

JOINTS = ("joint1", "joint2")
EE_SITE = "ee_site"
TARGET_SITE = "target"

DT = 0.002
CTRL_HZ = 50
SUBSTEPS = int(1.0 / (DT * CTRL_HZ))

MAX_STEPS = 500
GOAL_EPS = 0.03

# Change in the position-actuator target angle per action.
DELTA_Q_MAX = np.deg2rad(1.0)

ACTION_GRID = (-1, 0, 1)
ACTIONS = [(a1, a2) for a1 in ACTION_GRID for a2 in ACTION_GRID]

STATE_SIZE = 8
ACTION_SIZE = len(ACTIONS)

ACTION_PENALTY = 0.01
DISTANCE_PENALTY = 0.20
PROGRESS_REWARD = 5.0
STEP_PENALTY = 0.002
SUCCESS_BONUS = 10.0

# The target is generated near the current end-effector position.
# This task formulation previously produced approximately 40–50% success.
GOAL_NOISE_STD = 0.04

# Safe generation of guaranteed reachable targets.
GOAL_DISTANCE_MIN = 0.05
GOAL_DISTANCE_MAX = 0.18
MIN_EE_HEIGHT = 0.08

# Allow the position actuators to stabilize the random pose after reset.
RESET_SETTLE_STEPS = 100

# The target is generated from a configuration close to the current one.
# This avoids cases where the end effector is close to the target but the robot
# would need to transition to a completely different IK configuration.
GOAL_JOINT_DELTA = np.deg2rad(25.0)


# ============================================================
# 3. VANILLA DQN PARAMETERS
# ============================================================

NUM_EPISODES = 1500

GAMMA = 0.99
LEARNING_RATE = 5e-4

BUFFER_SIZE = 50_000
BATCH_SIZE = 64
MIN_BUFFER_SIZE = 500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.998

TARGET_UPDATE_STEPS = 1_500

# Reduce the optimizer learning rate after a basic policy has been learned.
LR_REDUCE_EPISODE = 1_000
LEARNING_RATE_AFTER_REDUCE = 1e-4

# Stop training if the fixed evaluation score does not improve for a long period.
EARLY_STOPPING_PATIENCE_EPISODES = 1_000
EARLY_STOPPING_MIN_EPISODE = 1_200
MIN_SUCCESS_IMPROVEMENT = 1e-9

# Use the same target set for every evaluation.
EVAL_SEED_BASE = 10_000

# Gradient clipping reduces abrupt changes in the Q-function.
MAX_GRAD_NORM = 10.0

# Show a separate test episode every 100 training episodes.
# Epsilon is zero in this episode, so the robot follows the current learned policy
# rather than random exploratory behavior.
RENDER_EVERY = 100

# Run the demonstration episode in real time.
REAL_TIME_VISUALIZATION = True

# Evaluate the policy separately without random actions.
EVAL_EVERY = 100
EVAL_EPISODES = 100

SEED = 42

torch.set_num_threads(2)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_joint_id(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' was not found.")

    return joint_id


def get_site_id(model, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise ValueError(f"Site '{site_name}' was not found.")

    return site_id


# ============================================================
# 5. RL ENVIRONMENT
# ============================================================

class PlanarArmEnv:
    """
    State:
        [dx, dz, q1/pi, q2/pi, ctrl1, ctrl2, velocity_1/5, velocity_2/5]

    Action:
        one of nine combinations:
        (-1,-1), (-1,0), ..., (1,1)

    Each action changes the target angles of the position actuators.
    """

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # A separate data buffer is used only to calculate
        # guaranteed reachable target positions.
        self.goal_data = mujoco.MjData(self.model)

        if self.model.nu != 2:
            raise RuntimeError(
                f"Expected 2 actuators, found: {self.model.nu}"
            )

        self.ee_site_id = get_site_id(self.model, EE_SITE)
        self.target_site_id = get_site_id(self.model, TARGET_SITE)

        self.joint_ids = [
            get_joint_id(self.model, JOINTS[0]),
            get_joint_id(self.model, JOINTS[1]),
        ]

        self.qpos_addresses = [
            self.model.jnt_qposadr[self.joint_ids[0]],
            self.model.jnt_qposadr[self.joint_ids[1]],
        ]

        self.dof_addresses = [
            self.model.jnt_dofadr[self.joint_ids[0]],
            self.model.jnt_dofadr[self.joint_ids[1]],
        ]

        self.steps = 0
        self.previous_distance = 0.0

    def _sample_safe_initial_pose(self):
        """
        Sample a random initial configuration inside the joint limits.

        Accept the pose only if the end effector remains above the floor.
        """
        for _ in range(200):
            candidate = []

            for joint_id in self.joint_ids:
                low, high = self.model.jnt_range[joint_id]

                # Do not start directly at the mechanical limits.
                margin = 0.10 * (high - low)
                candidate.append(
                    np.random.uniform(low + margin, high - margin)
                )

            for qpos_address, value in zip(
                self.qpos_addresses,
                candidate,
            ):
                self.data.qpos[qpos_address] = value

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            ee_height = self.data.site_xpos[self.ee_site_id][2]

            if ee_height >= MIN_EE_HEIGHT:
                return

        raise RuntimeError(
            "Could not find a safe initial pose."
        )

    def _sample_reachable_goal(self):
        """
        Generate a reachable target from a configuration
        close to the robot current configuration.

        This guarantees that:
        1. the target lies inside the workspace;
        2. the target is reachable under the joint limits;
        3. reaching the target does not require transitioning to a distant
           alternative IK configuration.
        """
        current_ee = self.data.site_xpos[self.ee_site_id].copy()

        current_q = np.array(
            [
                self.data.qpos[self.qpos_addresses[0]],
                self.data.qpos[self.qpos_addresses[1]],
            ],
            dtype=np.float64,
        )

        for _ in range(300):
            mujoco.mj_resetData(self.model, self.goal_data)

            candidate_q = current_q + np.random.uniform(
                -GOAL_JOINT_DELTA,
                GOAL_JOINT_DELTA,
                size=2,
            )

            for index, (joint_id, qpos_address) in enumerate(
                zip(self.joint_ids, self.qpos_addresses)
            ):
                low, high = self.model.jnt_range[joint_id]
                candidate_q[index] = np.clip(
                    candidate_q[index],
                    low,
                    high,
                )
                self.goal_data.qpos[qpos_address] = candidate_q[index]

            self.goal_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.goal_data)

            goal = self.goal_data.site_xpos[
                self.ee_site_id
            ].copy()

            distance = float(
                np.linalg.norm(
                    np.array(
                        [
                            goal[0] - current_ee[0],
                            goal[2] - current_ee[2],
                        ]
                    )
                )
            )

            if (
                GOAL_DISTANCE_MIN <= distance <= GOAL_DISTANCE_MAX
                and goal[2] >= MIN_EE_HEIGHT
            ):
                goal[1] = 0.0
                return goal

        # If no suitable target was found,
        # use the current end-effector position with a small reachable offset
        # obtained from the first suitable nearby joint configuration.
        mujoco.mj_resetData(self.model, self.goal_data)

        fallback_q = current_q.copy()
        fallback_q[0] += np.deg2rad(8.0)

        for index, (joint_id, qpos_address) in enumerate(
            zip(self.joint_ids, self.qpos_addresses)
        ):
            low, high = self.model.jnt_range[joint_id]
            fallback_q[index] = np.clip(fallback_q[index], low, high)
            self.goal_data.qpos[qpos_address] = fallback_q[index]

        mujoco.mj_forward(self.model, self.goal_data)

        goal = self.goal_data.site_xpos[self.ee_site_id].copy()
        goal[1] = 0.0
        goal[2] = max(goal[2], MIN_EE_HEIGHT)
        return goal

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # 1. Sample a random safe pose.
        self._sample_safe_initial_pose()

        # 2. Make the position actuators hold the current joint angles.
        self.data.ctrl[0] = self.data.qpos[
            self.qpos_addresses[0]
        ]
        self.data.ctrl[1] = self.data.qpos[
            self.qpos_addresses[1]
        ]

        # 3. Apply a short stabilization period so the robot does not move abruptly
        # immediately after the episode begins.
        for _ in range(RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        # After stabilization, use the actual current pose
        # as the new position-actuator command.
        self.data.qvel[:] = 0.0
        self.data.ctrl[0] = self.data.qpos[
            self.qpos_addresses[0]
        ]
        self.data.ctrl[1] = self.data.qpos[
            self.qpos_addresses[1]
        ]
        mujoco.mj_forward(self.model, self.data)

        # 4. Generate the target only from a valid
        # joint configuration.
        goal = self._sample_reachable_goal()
        self.model.site_pos[self.target_site_id] = goal
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        state = self.get_state()
        self.previous_distance = self.get_distance()

        return state

    def get_state(self):
        ee = self.data.site_xpos[self.ee_site_id]
        target = self.data.site_xpos[self.target_site_id]

        dx = target[0] - ee[0]
        dz = target[2] - ee[2]

        q1 = self.data.qpos[self.qpos_addresses[0]]
        q2 = self.data.qpos[self.qpos_addresses[1]]

        v1 = self.data.qvel[self.dof_addresses[0]]
        v2 = self.data.qvel[self.dof_addresses[1]]

        # For a position actuator, the current control command is part of the state:
        # identical joint positions and velocities can produce different dynamics under different controls.
        ctrl1_scale = max(
            abs(float(self.model.actuator_ctrlrange[0, 0])),
            abs(float(self.model.actuator_ctrlrange[0, 1])),
            1e-6,
        )
        ctrl2_scale = max(
            abs(float(self.model.actuator_ctrlrange[1, 0])),
            abs(float(self.model.actuator_ctrlrange[1, 1])),
            1e-6,
        )

        ctrl1 = float(self.data.ctrl[0]) / ctrl1_scale
        ctrl2 = float(self.data.ctrl[1]) / ctrl2_scale

        return np.array(
            [
                dx,
                dz,
                q1 / np.pi,
                q2 / np.pi,
                np.clip(ctrl1, -1.0, 1.0),
                np.clip(ctrl2, -1.0, 1.0),
                np.clip(v1 / 5.0, -1.0, 1.0),
                np.clip(v2 / 5.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def get_distance(self):
        state = self.get_state()
        return float(np.linalg.norm(state[:2]))

    def action_id_to_vector(self, action_id):
        return np.asarray(
            ACTIONS[action_id],
            dtype=np.float64,
        )

    def step(self, action_id):
        action_vector = self.action_id_to_vector(action_id)
        delta_q = action_vector * DELTA_Q_MAX

        # Update the position-actuator target angles.
        for actuator_id, joint_id in enumerate(self.joint_ids):
            low, high = self.model.jnt_range[joint_id]

            target_angle = (
                float(self.data.ctrl[actuator_id])
                + float(delta_q[actuator_id])
            )

            self.data.ctrl[actuator_id] = np.clip(
                target_angle,
                low,
                high,
            )

        reached_goal = False

        # Check the goal condition at every physics substep.
        # Otherwise, the robot could pass through the goal during these substeps
        # and move away before the next control-level check.
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

            if self.get_distance() < GOAL_EPS:
                reached_goal = True
                break

        self.steps += 1

        next_state = self.get_state()
        distance = self.get_distance()

        progress = self.previous_distance - distance

        reward = (
            PROGRESS_REWARD * progress
            - DISTANCE_PENALTY * distance
            - STEP_PENALTY
            - ACTION_PENALTY * float(np.sum(delta_q ** 2))
        )

        self.previous_distance = distance

        timeout = self.steps >= MAX_STEPS
        done = reached_goal or timeout

        if reached_goal:
            reward += SUCCESS_BONUS

            # After reaching the goal, hold the current pose
            # so the robot does not continue moving due to inertia.
            self.data.ctrl[0] = self.data.qpos[
                self.qpos_addresses[0]
            ]
            self.data.ctrl[1] = self.data.qpos[
                self.qpos_addresses[1]
            ]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        info = {
            "distance": distance,
            "reached_goal": reached_goal,
            "timeout": timeout,
        }

        return next_state, reward, done, info


# ============================================================
# 6. REPLAY BUFFER
# ============================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(
                    next_state,
                    dtype=np.float32,
                ).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = np.stack([item.state for item in batch])
        actions = np.asarray(
            [item.action for item in batch],
            dtype=np.int64,
        )
        rewards = np.asarray(
            [item.reward for item in batch],
            dtype=np.float32,
        )
        next_states = np.stack(
            [item.next_state for item in batch]
        )
        dones = np.asarray(
            [item.done for item in batch],
            dtype=np.float32,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 7. SIMPLE DQN NETWORK
# ============================================================

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# 8. EPSILON-GREEDY
# ============================================================

def select_action(network, state, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


# ============================================================
# 9. ONE TRAINING STEP
# ============================================================

def train_dqn_step(
    online_network,
    target_network,
    replay_buffer,
    optimizer,
    loss_function,
    device,
):
    if len(replay_buffer) < MIN_BUFFER_SIZE:
        return None

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(BATCH_SIZE)

    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        actions,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)

    rewards = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.as_tensor(
        next_states,
        dtype=torch.float32,
        device=device,
    )

    dones = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=device,
    )

    current_q = online_network(states).gather(
        1,
        actions,
    ).squeeze(1)

    with torch.no_grad():
        max_next_q = target_network(
            next_states
        ).max(dim=1).values

        target_q = (
            rewards
            + GAMMA * (1.0 - dones) * max_next_q
        )

    loss = loss_function(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        online_network.parameters(),
        MAX_GRAD_NORM,
    )
    optimizer.step()

    return float(loss.item())


# ============================================================
# 10. POLICY EVALUATION WITHOUT EXPLORATION
# ============================================================

def evaluate_policy(
    network,
    env,
    device,
    num_episodes=EVAL_EPISODES,
):
    """
    Evaluate the network on the same fixed task set.

    Save and restore the random-number-generator states so that
    evaluation does not alter the sequence of training episodes.
    """
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()

    was_training = network.training
    network.eval()

    successes = []
    final_distances = []
    episode_returns = []
    action_counts = np.zeros(ACTION_SIZE, dtype=np.int64)

    try:
        for eval_index in range(num_episodes):
            seed = EVAL_SEED_BASE + eval_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            state = env.reset()
            episode_return = 0.0

            while True:
                action = select_action(
                    network,
                    state,
                    epsilon=0.0,
                    device=device,
                )
                action_counts[action] += 1

                state, reward, done, info = env.step(action)
                episode_return += reward

                if done:
                    successes.append(
                        1.0 if info["reached_goal"] else 0.0
                    )
                    final_distances.append(info["distance"])
                    episode_returns.append(episode_return)
                    break
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)

        if was_training:
            network.train()

    total_actions = max(int(action_counts.sum()), 1)
    action_frequencies = action_counts / total_actions

    return {
        "success_rate": float(np.mean(successes)),
        "mean_distance": float(np.mean(final_distances)),
        "mean_return": float(np.mean(episode_returns)),
        "action_counts": action_counts,
        "action_frequencies": action_frequencies,
    }


# ============================================================
# 11. VISUAL TEST WITHOUT RANDOM ACTIONS
# ============================================================

def run_visual_evaluation(network, env, device, episode_number):
    """
    Display one separate episode with epsilon equal to zero.

    In this episode:
    - the network is not trained;
    - the replay buffer is not modified;
    - no random actions are used.
    """
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()

    # Use the same demonstration task for comparable behavior.
    random.seed(EVAL_SEED_BASE)
    np.random.seed(EVAL_SEED_BASE)
    torch.manual_seed(EVAL_SEED_BASE)
    state = env.reset()

    viewer = mujoco.viewer.launch_passive(
        env.model,
        env.data,
    )

    episode_return = 0.0
    final_info = None

    print(
        f"\n=== VISUAL EVALUATION after episode "
        f"{episode_number} | epsilon=0 ==="
    )

    try:
        while viewer.is_running():
            action = select_action(
                network,
                state,
                epsilon=0.0,
                device=device,
            )

            state, reward, done, info = env.step(action)

            episode_return += reward
            final_info = info

            viewer.sync()

            if REAL_TIME_VISUALIZATION:
                time.sleep(1.0 / CTRL_HZ)

            if done:
                # Pause briefly so the final pose remains visible.
                for _ in range(30):
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
                break

    finally:
        viewer.close()
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)

    if final_info is not None:
        print(
            f"Visual result: "
            f"success={final_info['reached_goal']}, "
            f"distance={final_info['distance']:.4f}, "
            f"return={episode_return:.2f}"
        )


# ============================================================
# 12. TRAINING WITH VISUALIZATION
# ============================================================

def train():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Substeps: {SUBSTEPS}")
    print(f"State size: {STATE_SIZE}")
    print(
        "Reward: "
        f"{PROGRESS_REWARD}*progress - "
        f"{DISTANCE_PENALTY}*distance - "
        f"{STEP_PENALTY} step penalty"
    )
    print(f"Action size: {ACTION_SIZE}")
    print(f"Render every: {RENDER_EVERY} episode(s)")
    print(
        f"Episode limit: {MAX_STEPS} steps "
        f"({MAX_STEPS / CTRL_HZ:.1f} s simulated time)"
    )
    print(
        f"Evaluation: every {EVAL_EVERY} episodes, "
        f"{EVAL_EPISODES} episodes, epsilon=0"
    )
    print(f"Replay buffer: {BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Fixed evaluation seeds: {EVAL_SEED_BASE}..{EVAL_SEED_BASE + EVAL_EPISODES - 1}")
    print(f"Best model: {BEST_MODEL_SAVE_PATH}")
    print(f"Last model: {LAST_MODEL_SAVE_PATH}")
    print(f"XML: {XML_PATH}")

    env = PlanarArmEnv(str(XML_PATH))
    eval_env = PlanarArmEnv(str(XML_PATH))

    online_network = DQN(
        STATE_SIZE,
        ACTION_SIZE,
    ).to(device)

    target_network = DQN(
        STATE_SIZE,
        ACTION_SIZE,
    ).to(device)

    target_network.load_state_dict(
        online_network.state_dict()
    )
    target_network.eval()

    optimizer = optim.Adam(
        online_network.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    total_steps = 0

    returns = []
    distances = []
    successes = []
    losses = []
    train_history = []

    eval_history = []
    best_success = -1.0
    best_distance = float("inf")
    best_episode = 0
    last_improvement_episode = 0
    lr_reduced = False
    train_action_counts = np.zeros(ACTION_SIZE, dtype=np.int64)

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state = env.reset()

            episode_return = 0.0
            final_distance = None
            success = False
            episode_losses = []
            episode_steps = 0

            while True:
                action = select_action(
                    online_network,
                    state,
                    epsilon,
                    device,
                )

                train_action_counts[action] += 1

                next_state, reward, done, info = env.step(
                    action
                )

                replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = train_dqn_step(
                    online_network,
                    target_network,
                    replay_buffer,
                    optimizer,
                    loss_function,
                    device,
                )

                if loss is not None:
                    losses.append(loss)
                    episode_losses.append(loss)

                state = next_state
                episode_return += reward
                final_distance = info["distance"]
                success = info["reached_goal"]

                total_steps += 1
                episode_steps += 1

                if total_steps % TARGET_UPDATE_STEPS == 0:
                    target_network.load_state_dict(
                        online_network.state_dict()
                    )

                if done:
                    break

            returns.append(episode_return)
            distances.append(final_distance)
            successes.append(1.0 if success else 0.0)

            epsilon = max(
                EPS_END,
                epsilon * EPS_DECAY,
            )

            train_return_ma50 = float(np.mean(returns[-50:]))
            train_distance_ma50 = float(np.mean(distances[-50:]))
            train_success_ma50 = float(np.mean(successes[-50:]))
            episode_mean_loss = (
                float(np.mean(episode_losses))
                if episode_losses
                else float("nan")
            )
            current_lr = float(optimizer.param_groups[0]["lr"])

            train_history.append(
                {
                    "episode": episode,
                    "episode_return": float(episode_return),
                    "return_ma50": train_return_ma50,
                    "final_distance": float(final_distance),
                    "distance_ma50": train_distance_ma50,
                    "success": 1 if success else 0,
                    "success_ma50": train_success_ma50,
                    "episode_steps": int(episode_steps),
                    "epsilon": float(epsilon),
                    "mean_loss": episode_mean_loss,
                    "learning_rate": current_lr,
                    "buffer_size": len(replay_buffer),
                    "total_steps": int(total_steps),
                }
            )

            if (
                not lr_reduced
                and episode >= LR_REDUCE_EPISODE
            ):
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = LEARNING_RATE_AFTER_REDUCE
                lr_reduced = True
                print(
                    f"  LR reduced: "
                    f"{LEARNING_RATE:.1e} -> "
                    f"{LEARNING_RATE_AFTER_REDUCE:.1e}"
                )

            if episode % 50 == 0:
                mean_return = np.mean(returns[-50:])
                mean_distance = np.mean(distances[-50:])
                success_rate = np.mean(successes[-50:])
                mean_loss = (
                    np.mean(losses[-200:])
                    if losses
                    else float("nan")
                )

                print(
                    f"EP {episode:4d} | "
                    f"epsilon={epsilon:.3f} | "
                    f"return(50)={mean_return:8.2f} | "
                    f"distance(50)={mean_distance:.4f} | "
                    f"success(50)={success_rate:.2f} | "
                    f"loss={mean_loss:.5f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                    f"buffer={len(replay_buffer)}"
                )


            if episode % EVAL_EVERY == 0:
                evaluation = evaluate_policy(
                    online_network,
                    eval_env,
                    device,
                    num_episodes=EVAL_EPISODES,
                )

                eval_history.append(
                    (
                        episode,
                        evaluation["success_rate"],
                        evaluation["mean_distance"],
                        evaluation["mean_return"],
                    )
                )

                dominant_ids = np.argsort(
                    evaluation["action_frequencies"]
                )[-3:][::-1]
                dominant_actions = ", ".join(
                    f"{ACTIONS[action_id]}="
                    f"{evaluation['action_frequencies'][action_id]:.1%}"
                    for action_id in dominant_ids
                )

                print(
                    f"  EVAL | epsilon=0 | "
                    f"success={evaluation['success_rate']:.2f} | "
                    f"distance={evaluation['mean_distance']:.4f} | "
                    f"return={evaluation['mean_return']:.2f}"
                )
                print(f"         top actions: {dominant_actions}")

                success_improved = (
                    evaluation["success_rate"]
                    > best_success + MIN_SUCCESS_IMPROVEMENT
                )
                distance_improved_on_tie = (
                    abs(evaluation["success_rate"] - best_success)
                    <= MIN_SUCCESS_IMPROVEMENT
                    and evaluation["mean_distance"] < best_distance
                )

                if success_improved or distance_improved_on_tie:
                    best_success = evaluation["success_rate"]
                    best_distance = evaluation["mean_distance"]
                    best_episode = episode
                    last_improvement_episode = episode

                    torch.save(
                        online_network.state_dict(),
                        BEST_MODEL_SAVE_PATH,
                    )

                    print(
                        f"  BEST saved: {BEST_MODEL_SAVE_PATH} | "
                        f"episode={best_episode}, "
                        f"success={best_success:.2f}, "
                        f"distance={best_distance:.4f}"
                    )

                if (
                    episode >= EARLY_STOPPING_MIN_EPISODE
                    and episode - last_improvement_episode
                    >= EARLY_STOPPING_PATIENCE_EPISODES
                ):
                    print(
                        f"\nEARLY STOPPING: evaluation did not improve for "
                        f"{EARLY_STOPPING_PATIENCE_EPISODES} episodes."
                    )
                    break


            if RENDER_EVERY > 0 and episode % RENDER_EVERY == 0:
                run_visual_evaluation(
                    online_network,
                    eval_env,
                    device,
                    episode,
                )

    finally:
        pass

    torch.save(
        online_network.state_dict(),
        LAST_MODEL_SAVE_PATH,
    )

    print(f"\nLast model saved: {LAST_MODEL_SAVE_PATH}")
    if best_episode > 0:
        print(
            f"Best model: {BEST_MODEL_SAVE_PATH} | "
            f"episode={best_episode}, "
            f"success={best_success:.2f}, "
            f"distance={best_distance:.4f}"
        )

    if train_history:
        fieldnames = list(train_history[0].keys())
        with open(TRAIN_HISTORY_PATH, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(train_history)
        print(f"Training history saved: {TRAIN_HISTORY_PATH}")

    if eval_history:
        np.savetxt(
            EVAL_HISTORY_PATH,
            np.asarray(eval_history, dtype=np.float64),
            delimiter=",",
            header="episode,success_rate,mean_distance,mean_return",
            comments="",
        )
        print(f"Evaluation history saved: {EVAL_HISTORY_PATH}")


if __name__ == "__main__":
    train()
