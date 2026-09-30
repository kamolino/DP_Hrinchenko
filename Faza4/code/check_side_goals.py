import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from evaluate_pushing_ppo import run_episode
from pushing_env import PushingEnv
from run_pushing_demo import save_replay


def main():
    parser = argparse.ArgumentParser(description="Check small goal changes with frozen weights.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "policy.zip"
    shutil.copy2(args.model, frozen)
    torch.set_num_threads(1)
    policy = PPO.load(frozen, device="cpu")
    env = PushingEnv("B")
    tasks = [dict(id=i+1, seed=42, object=[.44, 0], goal=[.54, offset])
             for i, offset in enumerate([-.06, -.04, -.02, 0, .02, .04, .06])]
    episodes = [run_episode(env, policy, task) for task in tasks]
    env.close()
    rows = []
    for episode in episodes:
        final = episode["object_trajectory"][-1]
        rows.append(dict(id=episode["id"], goal_y_cm=episode["goal"][1]*100,
                         final_y_cm=final[1]*100, final_error_cm=episode["object_error"]*100,
                         success=episode["is_success"], reason=episode["reason"]))
    result = dict(model_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  purpose="Seven development layouts; not an independent benchmark", rows=rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    (args.output / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (args.output / "episodes.json").write_text(json.dumps(episodes))
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for ax, episode in zip(axes.flat, episodes):
        obj = np.array(episode["object_trajectory"])
        tcp = np.array(episode["tcp_trajectory"])
        ax.plot(obj[:, 0]*100, obj[:, 1]*100, color="#df7319", label="Cube")
        ax.plot(tcp[:, 0]*100, tcp[:, 1]*100, color="#2685be", alpha=.7, label="Tool")
        ax.add_patch(plt.Circle(np.array(episode["goal"])*100, 2, color="#30b56d", alpha=.3))
        ax.scatter(44, 0, marker="s", color="#df7319")
        ax.set(title=f'Goal Y: {episode["goal"][1]*100:+.0f} cm\n{episode["reason"]}, error {episode["object_error"]*100:.1f} cm',
               xlim=(28, 76), ylim=(-22, 22), xlabel="X (cm)", ylabel="Y (cm)")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02, .8, "Same frozen policy\nSame cube and robot start\nOnly the goal changes\n\nGreen radius: 2 cm", fontsize=12)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(args.output / "side_goals.png", dpi=140)
    plt.close(fig)
    if args.replay:
        for index in [1, 3, 5]:
            task = tasks[index]
            save_replay(policy, args.output / f'goal_{task["goal"][1]*100:+.0f}cm.gif', "B", 42, task,
                        title=f'Frozen PPO | goal Y {task["goal"][1]*100:+.0f} cm')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
