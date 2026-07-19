from pathlib import Path
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DQN_RESULTS_DIR = PROJECT_ROOT / "results" / "dqn"


def save_line(df, x, y, title, ylabel, output_path, *, ylim=None):
    plt.figure(figsize=(9, 5))
    plt.plot(df[x], df[y])
    plt.xlabel("Training episode")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training and evaluation histories.")
    parser.add_argument("--train-csv", default=str(DQN_RESULTS_DIR / "dqn_train_history.csv"))
    parser.add_argument("--eval-csv", default=str(DQN_RESULTS_DIR / "dqn_eval_history.csv"))
    parser.add_argument("--output-dir", default=str(DQN_RESULTS_DIR / "graphs"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    required_train = {
        "episode", "episode_return", "return_ma50", "final_distance",
        "distance_ma50", "success", "success_ma50", "episode_steps",
        "epsilon", "mean_loss", "learning_rate"
    }
    missing = required_train.difference(train.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    save_line(train, "episode", "return_ma50", "DQN training reward (moving average 50)",
              "Mean episode return", output_dir / "training_reward_ma50.png")
    save_line(train, "episode", "success_ma50", "DQN training success rate (moving average 50)",
              "Success rate", output_dir / "training_success_ma50.png", ylim=(0, 1.05))
    save_line(train, "episode", "distance_ma50", "DQN training final distance (moving average 50)",
              "Mean final distance, m", output_dir / "training_distance_ma50.png")
    save_line(train, "episode", "epsilon", "Epsilon decay", "Epsilon",
              output_dir / "epsilon_decay.png", ylim=(0, 1.05))
    save_line(train.dropna(subset=["mean_loss"]), "episode", "mean_loss", "DQN mean loss per episode",
              "Smooth L1 loss", output_dir / "training_loss.png")
    save_line(train, "episode", "episode_steps", "Episode length", "Control steps",
              output_dir / "episode_length.png")

    eval_path = Path(args.eval_csv)
    if eval_path.exists():
        evaluation = pd.read_csv(eval_path)
        required_eval = {"episode", "success_rate", "mean_distance", "mean_return"}
        missing_eval = required_eval.difference(evaluation.columns)
        if missing_eval:
            raise ValueError(f"Missing evaluation columns: {sorted(missing_eval)}")

        save_line(evaluation, "episode", "mean_return", "DQN evaluation reward (epsilon = 0)",
                  "Mean evaluation return", output_dir / "evaluation_reward.png")
        save_line(evaluation, "episode", "success_rate", "DQN evaluation success rate (epsilon = 0)",
                  "Success rate", output_dir / "evaluation_success.png", ylim=(0, 1.05))
        save_line(evaluation, "episode", "mean_distance", "DQN evaluation final distance (epsilon = 0)",
                  "Mean final distance, m", output_dir / "evaluation_distance.png")

    print(f"Graphs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
