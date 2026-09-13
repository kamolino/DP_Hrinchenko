import argparse
import json
from pathlib import Path
import numpy as np
import torch
from precision_dqn import load_model
from workspace_benchmark import WorkspaceSamplingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output")
    torch.set_num_threads(1)
    network, config, metadata = load_model(args.model)
    config.max_steps = args.max_steps
    env = WorkspaceSamplingEnv(config)
    episodes = []
    for episode_index, task in enumerate(json.loads(args.tasks.read_text())):
        state = env.reset(dict(start_q=task["start_q"], goal=task["goal"]))
        positions = [env.ee().tolist()]
        joint_positions = [env.data.qpos[env.ids.qpos_ids].tolist()]
        for step_index in range(args.max_steps):
            state, _, done, info = env.step(network.action(state))
            positions.append(env.ee().tolist())
            joint_positions.append(env.data.qpos[env.ids.qpos_ids].tolist())
            if done:
                break
        episodes.append(
            dict(task=task, **info, positions=positions, qpos=joint_positions)
        )
        if (episode_index + 1) % 24 == 0:
            print(
                episode_index + 1,
                "success",
                sum(episode["success"] for episode in episodes),
                flush=True,
            )
    distances = np.array([episode["distance"] for episode in episodes])
    summary = dict(
        controller="DQN without global planner",
        model=str(args.model.resolve()),
        metadata=metadata,
        episodes=len(episodes),
        success_count=sum(episode["success"] for episode in episodes),
        success_rate=float(np.mean([episode["success"] for episode in episodes])),
        mean_error_mm=float(distances.mean() * 1000),
        p95_error_mm=float(np.percentile(distances, 95) * 1000),
        collisions=sum(episode["collision"] for episode in episodes),
        max_steps=args.max_steps,
        control_dt=env.dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, episodes=episodes)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
