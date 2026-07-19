from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn_train_visual_v7_1_logging import (
    PlanarArmEnv,
    DQN,
    STATE_SIZE,
    ACTION_SIZE,
    select_action,
    set_seed,
    EVAL_SEED_BASE,
    PROJECT_ROOT,
    XML_PATH,
    BEST_MODEL_SAVE_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN and save the EE trajectory.")
    parser.add_argument("--model", default=str(BEST_MODEL_SAVE_PATH))
    parser.add_argument("--xml", default=str(XML_PATH))
    parser.add_argument("--seed", type=int, default=EVAL_SEED_BASE)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "dqn" / "trajectory"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PlanarArmEnv(args.xml)
    network = DQN(STATE_SIZE, ACTION_SIZE).to(device)
    state_dict = torch.load(args.model, map_location=device)
    network.load_state_dict(state_dict)
    network.eval()

    state = env.reset()
    target = env.data.site_xpos[env.target_site_id].copy()

    rows = []
    episode_return = 0.0
    step_index = 0

    while True:
        ee = env.data.site_xpos[env.ee_site_id].copy()
        distance = env.get_distance()
        rows.append({
            "step": step_index,
            "ee_x": float(ee[0]),
            "ee_z": float(ee[2]),
            "target_x": float(target[0]),
            "target_z": float(target[2]),
            "distance": float(distance),
            "action": "",
            "reward": "",
        })

        action = select_action(network, state, epsilon=0.0, device=device)
        next_state, reward, done, info = env.step(action)
        episode_return += reward

        rows[-1]["action"] = int(action)
        rows[-1]["reward"] = float(reward)
        state = next_state
        step_index += 1

        if done:
            ee = env.data.site_xpos[env.ee_site_id].copy()
            rows.append({
                "step": step_index,
                "ee_x": float(ee[0]),
                "ee_z": float(ee[2]),
                "target_x": float(target[0]),
                "target_z": float(target[2]),
                "distance": float(info["distance"]),
                "action": "",
                "reward": "",
            })
            break

    csv_path = output_dir / f"trajectory_seed_{args.seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["ee_x"] for row in rows], dtype=float)
    z = np.array([row["ee_z"] for row in rows], dtype=float)

    plt.figure(figsize=(7, 7))
    plt.plot(x, z, linewidth=1.8, label="EE trajectory")
    plt.scatter([x[0]], [z[0]], s=70, marker="o", label="Start")
    plt.scatter([target[0]], [target[2]], s=110, marker="*", label="Target")
    plt.scatter([x[-1]], [z[-1]], s=70, marker="x", label="Final")
    goal_circle = plt.Circle((target[0], target[2]), 0.03, fill=False, linestyle="--", label="Goal tolerance")
    plt.gca().add_patch(goal_circle)
    plt.xlabel("EE x, m")
    plt.ylabel("EE z, m")
    plt.title(f"DQN end-effector trajectory (seed={args.seed})")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"trajectory_seed_{args.seed}.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"success={info['reached_goal']}")
    print(f"final_distance={info['distance']:.6f}")
    print(f"episode_return={episode_return:.6f}")
    print(f"steps={step_index}")
    print(f"trajectory_csv={csv_path}")
    print(f"trajectory_png={png_path}")


if __name__ == "__main__":
    main()
