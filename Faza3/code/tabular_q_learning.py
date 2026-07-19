import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco


# ============================================================
# 1. DEFAULT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML_PATH = str(PROJECT_ROOT / "mujoco" / "reset_pos_stable.xml")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "results" / "tabular" / "tabular_q_results")

JOINTS = ("joint1", "joint2")
EE_SITE = "ee_site"
TARGET_SITE = "target"

DT = 0.002
CTRL_HZ = 50
SUBSTEPS = int(1.0 / (DT * CTRL_HZ))

MAX_STEPS = 500
GOAL_EPS = 0.03
MIN_EE_HEIGHT = 0.08
RESET_SETTLE_STEPS = 100

GOAL_DISTANCE_MIN = 0.05
GOAL_DISTANCE_MAX = 0.18
GOAL_JOINT_DELTA = np.deg2rad(25.0)

ACTION_GRID = (-1, 0, 1)
ACTIONS = [(a1, a2) for a1 in ACTION_GRID for a2 in ACTION_GRID]
ACTION_SIZE = len(ACTIONS)

# Reward kept consistent with the working DQN baseline.
PROGRESS_REWARD = 5.0
DISTANCE_PENALTY = 0.20
STEP_PENALTY = 0.002
ACTION_PENALTY = 0.01
SUCCESS_BONUS = 10.0

# Tabular state: [dx, dz, v1, v2]
STATE_DIMENSIONS = 4

SEED = 42
EVAL_SEED_BASE = 10_000


@dataclass
class Config:
    xml_path: str
    output_dir: str
    episodes: int
    bins: int
    action_step_deg: float
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    eval_every: int
    eval_episodes: int
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint '{name}' not found.")
    return joint_id


def get_site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"Site '{name}' not found.")
    return site_id


class PlanarArmTabularEnv:
    """
    Continuous observation:
        [dx, dz, angular_velocity_1, angular_velocity_2]

    Discrete action:
        one of 9 combinations (-1, 0, +1) for two position actuators.
    """

    def __init__(self, xml_path: str, action_step_deg: float):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.goal_data = mujoco.MjData(self.model)

        if self.model.nu != 2:
            raise RuntimeError(f"Expected 2 actuators, found {self.model.nu}.")

        self.ee_site_id = get_site_id(self.model, EE_SITE)
        self.target_site_id = get_site_id(self.model, TARGET_SITE)

        self.joint_ids = [get_joint_id(self.model, name) for name in JOINTS]
        self.qpos_addresses = [
            self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids
        ]
        self.dof_addresses = [
            self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids
        ]

        self.action_step = np.deg2rad(float(action_step_deg))
        self.steps = 0
        self.previous_distance = 0.0

    def _sample_safe_initial_pose(self) -> None:
        for _ in range(200):
            candidate = []
            for joint_id in self.joint_ids:
                low, high = self.model.jnt_range[joint_id]
                margin = 0.10 * (high - low)
                candidate.append(np.random.uniform(low + margin, high - margin))

            for address, value in zip(self.qpos_addresses, candidate):
                self.data.qpos[address] = value

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            if self.data.site_xpos[self.ee_site_id][2] >= MIN_EE_HEIGHT:
                return

        raise RuntimeError("Could not sample a safe initial pose.")

    def _sample_reachable_goal(self) -> np.ndarray:
        current_ee = self.data.site_xpos[self.ee_site_id].copy()
        current_q = np.array(
            [self.data.qpos[address] for address in self.qpos_addresses],
            dtype=np.float64,
        )

        for _ in range(300):
            mujoco.mj_resetData(self.model, self.goal_data)

            candidate_q = current_q + np.random.uniform(
                -GOAL_JOINT_DELTA,
                GOAL_JOINT_DELTA,
                size=2,
            )

            for index, (joint_id, address) in enumerate(
                zip(self.joint_ids, self.qpos_addresses)
            ):
                low, high = self.model.jnt_range[joint_id]
                candidate_q[index] = np.clip(candidate_q[index], low, high)
                self.goal_data.qpos[address] = candidate_q[index]

            self.goal_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.goal_data)

            goal = self.goal_data.site_xpos[self.ee_site_id].copy()
            planar_distance = float(
                np.linalg.norm(
                    [
                        goal[0] - current_ee[0],
                        goal[2] - current_ee[2],
                    ]
                )
            )

            if (
                GOAL_DISTANCE_MIN <= planar_distance <= GOAL_DISTANCE_MAX
                and goal[2] >= MIN_EE_HEIGHT
            ):
                goal[1] = 0.0
                return goal

        # Deterministic fallback through a nearby joint configuration.
        mujoco.mj_resetData(self.model, self.goal_data)
        fallback_q = current_q.copy()
        fallback_q[0] += np.deg2rad(8.0)

        for index, (joint_id, address) in enumerate(
            zip(self.joint_ids, self.qpos_addresses)
        ):
            low, high = self.model.jnt_range[joint_id]
            fallback_q[index] = np.clip(fallback_q[index], low, high)
            self.goal_data.qpos[address] = fallback_q[index]

        mujoco.mj_forward(self.model, self.goal_data)
        goal = self.goal_data.site_xpos[self.ee_site_id].copy()
        goal[1] = 0.0
        goal[2] = max(goal[2], MIN_EE_HEIGHT)
        return goal

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self._sample_safe_initial_pose()

        for actuator_id, address in enumerate(self.qpos_addresses):
            self.data.ctrl[actuator_id] = self.data.qpos[address]

        for _ in range(RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = 0.0
        for actuator_id, address in enumerate(self.qpos_addresses):
            self.data.ctrl[actuator_id] = self.data.qpos[address]
        mujoco.mj_forward(self.model, self.data)

        goal = self._sample_reachable_goal()
        self.model.site_pos[self.target_site_id] = goal
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.previous_distance = self.get_distance()
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site_id]
        target = self.data.site_xpos[self.target_site_id]

        dx = target[0] - ee[0]
        dz = target[2] - ee[2]
        v1 = self.data.qvel[self.dof_addresses[0]]
        v2 = self.data.qvel[self.dof_addresses[1]]

        return np.array([dx, dz, v1, v2], dtype=np.float32)

    def get_distance(self) -> float:
        observation = self.get_observation()
        return float(np.linalg.norm(observation[:2]))

    def get_ee_position(self) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site_id]
        return np.array([ee[0], ee[2]], dtype=np.float64)

    def get_target_position(self) -> np.ndarray:
        target = self.data.site_xpos[self.target_site_id]
        return np.array([target[0], target[2]], dtype=np.float64)

    def step(self, action_id: int):
        if not 0 <= action_id < ACTION_SIZE:
            raise ValueError(f"Invalid action id: {action_id}")

        action_vector = np.asarray(ACTIONS[action_id], dtype=np.float64)
        delta_q = action_vector * self.action_step

        for actuator_id, joint_id in enumerate(self.joint_ids):
            low, high = self.model.jnt_range[joint_id]
            self.data.ctrl[actuator_id] = np.clip(
                float(self.data.ctrl[actuator_id]) + float(delta_q[actuator_id]),
                low,
                high,
            )

        reached_goal = False
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self.get_distance() < GOAL_EPS:
                reached_goal = True
                break

        self.steps += 1
        observation = self.get_observation()
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
            for actuator_id, address in enumerate(self.qpos_addresses):
                self.data.ctrl[actuator_id] = self.data.qpos[address]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        info = {
            "distance": distance,
            "reached_goal": reached_goal,
            "timeout": timeout,
            "steps": self.steps,
        }
        return observation, float(reward), done, info


class StateDiscretizer:
    """
    Discretizes [dx, dz, v1, v2].

    Position error ranges are tied to the generated task difficulty.
    Velocity ranges are clipped to avoid an impractically large table.
    """

    def __init__(self, bins_per_dimension: int):
        if bins_per_dimension < 2:
            raise ValueError("bins_per_dimension must be at least 2.")

        self.bins_per_dimension = int(bins_per_dimension)

        # dx and dz are normally within +/- 0.18 m by construction.
        # Slightly wider limits reduce clipping near boundaries.
        self.low = np.array([-0.22, -0.22, -3.0, -3.0], dtype=np.float64)
        self.high = np.array([0.22, 0.22, 3.0, 3.0], dtype=np.float64)

        self.edges = [
            np.linspace(
                self.low[index],
                self.high[index],
                self.bins_per_dimension + 1,
                dtype=np.float64,
            )[1:-1]
            for index in range(STATE_DIMENSIONS)
        ]

    @property
    def state_shape(self):
        return (self.bins_per_dimension,) * STATE_DIMENSIONS

    @property
    def number_of_states(self) -> int:
        return self.bins_per_dimension ** STATE_DIMENSIONS

    def encode(self, observation: np.ndarray):
        observation = np.asarray(observation, dtype=np.float64)
        clipped = np.clip(observation, self.low, self.high)
        return tuple(
            int(np.digitize(clipped[index], self.edges[index]))
            for index in range(STATE_DIMENSIONS)
        )


def epsilon_greedy_action(
    q_table: np.ndarray,
    state_index,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)

    q_values = q_table[state_index]
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(np.isclose(q_values, max_q))
    return int(np.random.choice(best_actions))


def evaluate_policy(
    q_table: np.ndarray,
    discretizer: StateDiscretizer,
    env: PlanarArmTabularEnv,
    num_episodes: int,
):
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    successes = []
    final_distances = []
    returns = []
    lengths = []
    action_counts = np.zeros(ACTION_SIZE, dtype=np.int64)

    try:
        for eval_index in range(num_episodes):
            seed = EVAL_SEED_BASE + eval_index
            random.seed(seed)
            np.random.seed(seed)

            observation = env.reset()
            episode_return = 0.0

            while True:
                state_index = discretizer.encode(observation)
                action = epsilon_greedy_action(
                    q_table,
                    state_index,
                    epsilon=0.0,
                )
                action_counts[action] += 1

                observation, reward, done, info = env.step(action)
                episode_return += reward

                if done:
                    successes.append(float(info["reached_goal"]))
                    final_distances.append(info["distance"])
                    returns.append(episode_return)
                    lengths.append(info["steps"])
                    break
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_distance": float(np.mean(final_distances)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "action_counts": action_counts,
    }


def save_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def train(config: Config) -> None:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PlanarArmTabularEnv(config.xml_path, config.action_step_deg)
    eval_env = PlanarArmTabularEnv(config.xml_path, config.action_step_deg)
    discretizer = StateDiscretizer(config.bins)

    q_shape = discretizer.state_shape + (ACTION_SIZE,)
    q_table = np.zeros(q_shape, dtype=np.float32)

    print("=== Tabular Q-learning ===")
    print(f"XML: {config.xml_path}")
    print(f"Episodes: {config.episodes}")
    print(f"Bins per dimension: {config.bins}")
    print(f"Discrete states: {discretizer.number_of_states:,}")
    print(f"Q-values: {q_table.size:,}")
    print(f"Q-table memory: {q_table.nbytes / (1024 ** 2):.2f} MiB")
    print(f"Action step: {config.action_step_deg:.2f} deg")
    print(f"Alpha: {config.alpha}")
    print(f"Gamma: {config.gamma}")
    print(
        f"Epsilon: {config.epsilon_start} -> {config.epsilon_end}, "
        f"decay={config.epsilon_decay}"
    )

    training_rows = []
    evaluation_rows = []

    epsilon = config.epsilon_start
    best_success = -1.0
    best_distance = float("inf")

    start_time = time.time()

    for episode in range(1, config.episodes + 1):
        observation = env.reset()
        state_index = discretizer.encode(observation)

        episode_return = 0.0
        episode_td_errors = []

        while True:
            action = epsilon_greedy_action(q_table, state_index, epsilon)
            next_observation, reward, done, info = env.step(action)
            next_state_index = discretizer.encode(next_observation)

            old_q = float(q_table[state_index + (action,)])
            bootstrap = 0.0 if done else float(np.max(q_table[next_state_index]))
            td_target = reward + config.gamma * bootstrap
            td_error = td_target - old_q

            q_table[state_index + (action,)] = (
                old_q + config.alpha * td_error
            )

            episode_td_errors.append(abs(td_error))
            episode_return += reward
            state_index = next_state_index

            if done:
                break

        epsilon = max(
            config.epsilon_end,
            epsilon * config.epsilon_decay,
        )

        training_rows.append(
            [
                episode,
                episode_return,
                info["distance"],
                int(info["reached_goal"]),
                info["steps"],
                epsilon,
                float(np.mean(episode_td_errors)),
            ]
        )

        if episode % 50 == 0:
            recent = training_rows[-50:]
            print(
                f"EP {episode:5d} | "
                f"eps={epsilon:.3f} | "
                f"return50={np.mean([row[1] for row in recent]):7.2f} | "
                f"distance50={np.mean([row[2] for row in recent]):.4f} | "
                f"success50={np.mean([row[3] for row in recent]):.2f} | "
                f"steps50={np.mean([row[4] for row in recent]):.1f}"
            )

        if episode % config.eval_every == 0:
            result = evaluate_policy(
                q_table,
                discretizer,
                eval_env,
                config.eval_episodes,
            )

            evaluation_rows.append(
                [
                    episode,
                    result["success_rate"],
                    result["mean_distance"],
                    result["mean_return"],
                    result["mean_length"],
                ]
            )

            print(
                f"  EVAL | epsilon=0 | "
                f"success={result['success_rate']:.2f} | "
                f"distance={result['mean_distance']:.4f} | "
                f"return={result['mean_return']:.2f} | "
                f"length={result['mean_length']:.1f}"
            )

            improved = (
                result["success_rate"] > best_success
                or (
                    np.isclose(result["success_rate"], best_success)
                    and result["mean_distance"] < best_distance
                )
            )
            if improved:
                best_success = result["success_rate"]
                best_distance = result["mean_distance"]
                np.save(output_dir / "q_table_best.npy", q_table)

    np.save(output_dir / "q_table_last.npy", q_table)

    save_csv(
        output_dir / "tabular_train_history.csv",
        [
            "episode",
            "episode_return",
            "final_distance",
            "success",
            "episode_steps",
            "epsilon",
            "mean_abs_td_error",
        ],
        training_rows,
    )

    save_csv(
        output_dir / "tabular_eval_history.csv",
        [
            "episode",
            "success_rate",
            "mean_distance",
            "mean_return",
            "mean_length",
        ],
        evaluation_rows,
    )

    metadata = {
        "algorithm": "tabular_q_learning",
        "state": ["dx", "dz", "joint_velocity_1", "joint_velocity_2"],
        "actions": ACTIONS,
        "bins_per_dimension": config.bins,
        "number_of_discrete_states": discretizer.number_of_states,
        "q_values": int(q_table.size),
        "q_table_memory_mib": q_table.nbytes / (1024 ** 2),
        "action_step_deg": config.action_step_deg,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay": config.epsilon_decay,
        "episodes": config.episodes,
        "eval_every": config.eval_every,
        "eval_episodes": config.eval_episodes,
        "seed": config.seed,
        "elapsed_seconds": time.time() - start_time,
        "best_eval_success": best_success,
        "best_eval_distance": best_distance,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved:")
    print(output_dir / "q_table_best.npy")
    print(output_dir / "q_table_last.npy")
    print(output_dir / "tabular_train_history.csv")
    print(output_dir / "tabular_eval_history.csv")
    print(output_dir / "metadata.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tabular Q-learning for a 2-DOF MuJoCo planar arm."
    )
    parser.add_argument("--xml", default=DEFAULT_XML_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--bins", type=int, default=7)
    parser.add_argument("--action-step-deg", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9985)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        Config(
            xml_path=args.xml,
            output_dir=args.output_dir,
            episodes=args.episodes,
            bins=args.bins,
            action_step_deg=args.action_step_deg,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
    )
