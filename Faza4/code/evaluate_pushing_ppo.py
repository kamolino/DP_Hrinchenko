import argparse
import hashlib
import gzip
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from pushing_env import PushingEnv
from plot_pushing_results import plot_evaluation


def run_episode(env, policy, task):
    obs, _ = env.reset(seed=task["seed"], options=task)
    object_trail = [env.object_position.tolist()]
    tcp_trail = [env.tcp.tolist()]
    robot_start = env.data.qpos[env.controller.qpos].tolist()
    actions, states = [], [env.data.qpos.tolist()]
    rewards = 0.0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, timeout, info = env.step(action)
        actions.append(action.tolist())
        states.append(env.data.qpos.tolist())
        object_trail.append(env.object_position.tolist())
        tcp_trail.append(env.tcp.tolist())
        rewards += reward
        if done or timeout:
            break
    path_length = np.linalg.norm(np.diff(np.array(object_trail)[:, :2], axis=0), axis=1).sum()
    return dict(**task, **{k: v for k, v in info.items() if k != "reward_parts"},
                robot_start=robot_start, reward=rewards, object_path_length=float(path_length),
                time_to_success=info["sim_time"] if info["is_success"] else None,
                object_trajectory=object_trail, tcp_trajectory=tcp_trail, actions=actions, qpos=states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=920617)
    parser.add_argument("--tasks", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    # Copy and hash the checkpoint before creating or running the final tasks.
    frozen = args.output / "frozen_model.zip"
    shutil.copy2(args.model, frozen)
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (args.output / "freeze.json").write_text(json.dumps(dict(model_sha256=digest,
        source=str(args.model.resolve()), scenario_seed=args.seed, stage="C"), indent=2))
    source = args.output / "source"
    source.mkdir()
    project = Path(__file__).resolve().parents[1]
    for name in ["settings.py", "pushing_env.py", "tcp_controller.py"]:
        shutil.copy2(project / "code" / name, source / name)
    shutil.copy2(project / "mujoco" / "scene.xml", source / "scene.xml")
    shutil.copy2(project / "mujoco" / "franka_fr3" / "fr3.xml", source / "fr3.xml")
    if (args.model.parent / "config.json").exists():
        shutil.copy2(args.model.parent / "config.json", source / "training_config.json")
    env = PushingEnv("C")
    if args.tasks:
        tasks = json.loads(args.tasks.read_text())
    else:
        tasks = []
        for index in range(args.episodes):
            env.reset(seed=args.seed + index)
            tasks.append(dict(id=index + 1, seed=args.seed + index,
                              object=env.object_position[:2].tolist(), goal=env.goal.tolist()))
    (args.output / "tasks.json").write_text(json.dumps(tasks, indent=2))
    model = PPO.load(frozen, device="cpu")
    episodes = []
    with gzip.open(args.output / "episodes.jsonl.gz", "wt") as file:
        for task in tasks:
            episode = run_episode(env, model, task)
            episodes.append(episode)
            file.write(json.dumps(episode) + "\n")
            file.flush()
            print(f'{len(episodes)}/{len(tasks)} {episode["reason"]} error={episode["object_error"]*100:.2f}cm', flush=True)
    errors = np.array([e["object_error"] for e in episodes])
    successes = [e for e in episodes if e["is_success"]]
    summary = dict(episodes=len(episodes), successes=len(successes), success_rate=len(successes)/len(episodes),
                   mean_error_m=float(errors.mean()), median_error_m=float(np.median(errors)),
                   std_error_m=float(errors.std()), failure_reasons=dict(Counter(e["reason"] for e in episodes if not e["is_success"])),
                   mean_success_steps=float(np.mean([e["steps"] for e in successes])) if successes else None,
                   mean_time_to_success=float(np.mean([e["time_to_success"] for e in successes])) if successes else None,
                   model_sha256=digest, protocol=("Frozen PPO, replay of saved C tasks, deterministic actions" if args.tasks else "Frozen PPO, independent fixed C tasks, deterministic actions"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    compact = [{k: v for k, v in e.items() if k not in ("qpos", "actions")} for e in episodes]
    (args.output / "evaluation.json").write_text(json.dumps(dict(summary=summary, episodes=compact)))
    with (args.output / "metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "seed", "is_success", "reason", "object_error", "steps", "sim_time", "time_to_success", "object_path_length"])
        writer.writeheader()
        writer.writerows({k: e[k] for k in writer.fieldnames} for e in episodes)
    plot_evaluation(args.output)
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
