import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from precision_environment import PrecisionConfig
from joint_waypoint_dqn import (
    WorkspaceEnv,
    JointActionDQN,
    JointTeacher,
    joint_observation,
    apply_joint_action,
    save_joint,
)
from workspace_planner import WorkspacePlanner

# ============================================================
# 1. TRAINING PARAMETERS
# ============================================================

NUM_EPISODES = 160
SEED = 411
MAX_STEPS = 600
EXPLORATION_RATE = 0.08
PRETRAIN_UPDATES = 2500
DQN_UPDATES = 2500
BATCH_SIZE = 256
PRETRAIN_LEARNING_RATE = 0.0003
LEARNING_RATE = 0.0001
GAMMA = 0.95
EXPERT_MARGIN = 0.15
TARGET_UPDATE_STEPS = 200


# ============================================================
# 2. TRAINING
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = PrecisionConfig(max_steps=MAX_STEPS)
    env = WorkspaceEnv(config, args.seed)
    planner = WorkspacePlanner(env, args.seed)

    # Collect examples from the simulation.
    teacher = JointTeacher(env)
    states = []
    scores = []
    actions = []
    rewards = []
    next_states = []
    terminals = []
    training_start = time.time()
    for episode_index in range(args.episodes):
        for _ in range(3000):
            start_q = env.rng.uniform(env.low, env.high)
            if env.valid_pose(start_q) is None:
                continue
            target_q = np.clip(
                start_q + env.rng.uniform(-0.18, 0.18, 7), env.low, env.high
            )
            goal = env.valid_pose(target_q)
            if goal is not None and planner.edge(start_q, target_q):
                break
        else:
            raise RuntimeError("No valid local training task")
        env.reset(dict(start_q=start_q, goal=goal))
        state = joint_observation(env, target_q)
        for k in range(100):
            action_scores = teacher.scores(target_q)
            action = (
                int(action_scores.argmax())
                if env.rng.random() > EXPLORATION_RATE
                else int(env.rng.integers(15))
            )
            distance_before = np.linalg.norm(target_q - env.data.qpos[env.ids.qpos_ids])
            _, _, done, info = apply_joint_action(env, target_q, action)
            distance_after = np.linalg.norm(target_q - env.data.qpos[env.ids.qpos_ids])
            next_state = joint_observation(env, target_q)
            reward = (
                4 * (distance_before - distance_after) / max(distance_before, 0.001)
                - 0.04
                + (3 if info["success"] else 0)
                - (5 if info["collision"] else 0)
            )
            states.append(state)
            scores.append(action_scores)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(info["terminated"])
            state = next_state
            if done:
                break
        if (episode_index + 1) % 20 == 0:
            print(
                "episodes",
                episode_index + 1,
                "samples",
                len(states),
                "seconds",
                round(time.time() - training_start),
                flush=True,
            )

    # Save the collected transitions.
    arrays = dict(
        states=np.array(states),
        scores=np.array(scores),
        actions=np.array(actions),
        rewards=np.array(rewards, np.float32),
        nextstates=np.array(next_states),
        terminals=np.array(terminals, np.float32),
    )
    np.savez_compressed(args.output / "replay.npz", **arrays)
    replay = {k: torch.as_tensor(v) for k, v in arrays.items()}
    transition_count = len(states)

    # First train the network on the teacher scores.
    network = JointActionDQN()
    optimizer = torch.optim.Adam(network.parameters(), lr=PRETRAIN_LEARNING_RATE)
    for update_index in range(PRETRAIN_UPDATES):
        batch_indices = torch.randint(transition_count, (BATCH_SIZE,))
        q_values = network(replay["states"][batch_indices])
        teacher_values = replay["scores"][batch_indices]
        loss = nn.functional.smooth_l1_loss(
            q_values, teacher_values
        ) + 0.05 * nn.functional.cross_entropy(q_values / 0.1, teacher_values.argmax(1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
    save_joint(
        args.output / "pretrain.pt",
        network,
        "teacher_pretrain",
        config,
        seed=args.seed,
        transitions=transition_count,
    )

    # Continue with Double DQN updates.
    target_network = JointActionDQN()
    target_network.load_state_dict(network.state_dict())
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE
    for update_index in range(DQN_UPDATES):
        batch_indices = torch.randint(transition_count, (BATCH_SIZE,))
        q_values = network(replay["states"][batch_indices])
        chosen_values = q_values.gather(
            1, replay["actions"][batch_indices, None]
        ).squeeze(1)
        with torch.no_grad():
            batch_next_states = replay["nextstates"][batch_indices]
            best_actions = network(batch_next_states).argmax(1, keepdim=True)
            target_values = replay["rewards"][batch_indices] + GAMMA * (
                1 - replay["terminals"][batch_indices]
            ) * target_network(batch_next_states).gather(1, best_actions).squeeze(1)
        expert_actions = replay["scores"][batch_indices].argmax(1)
        margin = torch.full_like(q_values, EXPERT_MARGIN)
        margin.scatter_(1, expert_actions[:, None], 0)
        loss = (
            nn.functional.smooth_l1_loss(chosen_values, target_values)
            + (
                (q_values + margin).max(1).values
                - q_values.gather(1, expert_actions[:, None]).squeeze(1)
            ).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
        if (update_index + 1) % TARGET_UPDATE_STEPS == 0:
            target_network.load_state_dict(network.state_dict())
    save_joint(
        args.output / "double_dqn.pt",
        network,
        "double_dqn",
        config,
        seed=args.seed,
        transitions=transition_count,
        pretrain_updates=PRETRAIN_UPDATES,
        dqn_updates=DQN_UPDATES,
    )

    # Save the training summary.
    (args.output / "training.json").write_text(
        json.dumps(
            dict(
                seed=args.seed,
                episodes=args.episodes,
                transitions=transition_count,
                seconds=time.time() - training_start,
            ),
            indent=2,
        )
    )
    print(
        "Saved Double DQN",
        args.output,
        "elapsed",
        time.time() - training_start,
        flush=True,
    )


if __name__ == "__main__":
    main()
