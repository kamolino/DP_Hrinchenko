import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "tabular" / "tabular_q_results"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def moving_average(values, window=50):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.mean(values[start:index + 1])
    return result


def save_plot(x, y, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    train_rows = read_csv(results_dir / "tabular_train_history.csv")
    eval_rows = read_csv(results_dir / "tabular_eval_history.csv")

    episodes = np.array([int(row["episode"]) for row in train_rows])
    rewards = np.array([float(row["episode_return"]) for row in train_rows])
    distances = np.array([float(row["final_distance"]) for row in train_rows])
    successes = np.array([float(row["success"]) for row in train_rows])
    td_errors = np.array([float(row["mean_abs_td_error"]) for row in train_rows])
    epsilon = np.array([float(row["epsilon"]) for row in train_rows])

    save_plot(
        episodes,
        moving_average(rewards),
        "Tabular Q-learning training reward (moving average 50)",
        "Training episode",
        "Mean episode return",
        results_dir / "training_reward_ma50.png",
    )
    save_plot(
        episodes,
        moving_average(distances),
        "Tabular Q-learning final distance (moving average 50)",
        "Training episode",
        "Mean final distance, m",
        results_dir / "training_distance_ma50.png",
    )
    save_plot(
        episodes,
        moving_average(successes),
        "Tabular Q-learning success rate (moving average 50)",
        "Training episode",
        "Success rate",
        results_dir / "training_success_ma50.png",
    )
    save_plot(
        episodes,
        td_errors,
        "Tabular Q-learning mean absolute TD error",
        "Training episode",
        "Mean absolute TD error",
        results_dir / "training_td_error.png",
    )
    save_plot(
        episodes,
        epsilon,
        "Tabular Q-learning epsilon decay",
        "Training episode",
        "Epsilon",
        results_dir / "epsilon_decay.png",
    )

    if eval_rows:
        eval_episodes = np.array([int(row["episode"]) for row in eval_rows])
        eval_success = np.array([float(row["success_rate"]) for row in eval_rows])
        eval_distance = np.array([float(row["mean_distance"]) for row in eval_rows])
        eval_return = np.array([float(row["mean_return"]) for row in eval_rows])

        save_plot(
            eval_episodes,
            eval_success,
            "Tabular Q-learning evaluation success rate (epsilon = 0)",
            "Training episode",
            "Success rate",
            results_dir / "evaluation_success.png",
        )
        save_plot(
            eval_episodes,
            eval_distance,
            "Tabular Q-learning evaluation final distance (epsilon = 0)",
            "Training episode",
            "Mean final distance, m",
            results_dir / "evaluation_distance.png",
        )
        save_plot(
            eval_episodes,
            eval_return,
            "Tabular Q-learning evaluation reward (epsilon = 0)",
            "Training episode",
            "Mean evaluation return",
            results_dir / "evaluation_reward.png",
        )

    print(f"Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
