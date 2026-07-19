import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABULAR_RESULTS_DIR = PROJECT_ROOT / "results" / "tabular"
DQN_RESULTS_DIR = PROJECT_ROOT / "results" / "dqn"
COMPARISON_RESULTS_DIR = PROJECT_ROOT / "results" / "comparison"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_tabular_result_directories(root: Path):
    """
    Find every directory below root that contains both required tabular files.
    This supports names such as tabular_q_results, tabular_exp_bins5, and others.
    """
    if not root.exists():
        return []

    directories = []
    for metadata_path in root.rglob("metadata.json"):
        directory = metadata_path.parent
        eval_path = directory / "tabular_eval_history.csv"
        if eval_path.exists():
            directories.append(directory)

    return sorted(set(directories))


def load_tabular_result(directory: Path):
    metadata_path = directory / "metadata.json"
    eval_path = directory / "tabular_eval_history.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation history: {eval_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    rows = read_csv(eval_path)
    if not rows:
        raise RuntimeError(f"No evaluation rows in {eval_path}")

    best_row = max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            -float(row["mean_distance"]),
        ),
    )
    final_row = rows[-1]

    return {
        "method": f"Tabular Q ({metadata['bins_per_dimension']} bins)",
        "source_directory": str(directory),
        "bins": int(metadata["bins_per_dimension"]),
        "states": int(metadata["number_of_discrete_states"]),
        "q_values": int(metadata["q_values"]),
        "memory_mib": float(metadata["q_table_memory_mib"]),
        "elapsed_seconds": float(metadata["elapsed_seconds"]),
        "best_episode": int(float(best_row["episode"])),
        "best_success": float(best_row["success_rate"]),
        "best_distance": float(best_row["mean_distance"]),
        "best_return": float(best_row["mean_return"]),
        "final_success": float(final_row["success_rate"]),
        "final_distance": float(final_row["mean_distance"]),
        "final_return": float(final_row["mean_return"]),
    }


def load_dqn_result(path: Path):
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No evaluation rows in {path}")

    best_row = max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            -float(row["mean_distance"]),
        ),
    )
    final_row = rows[-1]

    return {
        "method": "DQN",
        "source_directory": str(path.parent),
        "bins": "",
        "states": "",
        "q_values": "",
        "memory_mib": "",
        "elapsed_seconds": "",
        "best_episode": int(float(best_row["episode"])),
        "best_success": float(best_row["success_rate"]),
        "best_distance": float(best_row["mean_distance"]),
        "best_return": float(best_row["mean_return"]),
        "final_success": float(final_row["success_rate"]),
        "final_distance": float(final_row["mean_distance"]),
        "final_return": float(final_row["mean_return"]),
    }


def save_summary_csv(results, path: Path):
    fieldnames = [
        "method",
        "source_directory",
        "bins",
        "states",
        "q_values",
        "memory_mib",
        "elapsed_seconds",
        "best_episode",
        "best_success",
        "best_distance",
        "best_return",
        "final_success",
        "final_distance",
        "final_return",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def bar_plot(labels, values, title, ylabel, output_path):
    plt.figure(figsize=(9, 6))
    positions = np.arange(len(labels))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=15)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare all available tabular Q-learning experiments with DQN."
    )
    parser.add_argument(
        "--tabular-root",
        default=str(TABULAR_RESULTS_DIR),
        help="Root directory that contains tabular experiment folders.",
    )
    parser.add_argument(
        "--tabular-dirs",
        nargs="*",
        default=None,
        help="Optional explicit list of tabular result directories.",
    )
    parser.add_argument(
        "--dqn-eval",
        default=str(DQN_RESULTS_DIR / "dqn_eval_history.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(COMPARISON_RESULTS_DIR),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tabular_dirs:
        tabular_directories = [Path(path) for path in args.tabular_dirs]
    else:
        tabular_directories = find_tabular_result_directories(
            Path(args.tabular_root)
        )

    if not tabular_directories:
        raise FileNotFoundError(
            "No tabular result directories were found. Each result directory "
            "must contain metadata.json and tabular_eval_history.csv."
        )

    tabular_results = [
        load_tabular_result(directory)
        for directory in tabular_directories
    ]

    # Keep only one result per bin count. If duplicates exist, keep the one
    # with the highest success rate and then the lowest distance.
    best_by_bins = {}
    for result in tabular_results:
        bins = result["bins"]
        current = best_by_bins.get(bins)
        if current is None or (
            result["best_success"],
            -result["best_distance"],
        ) > (
            current["best_success"],
            -current["best_distance"],
        ):
            best_by_bins[bins] = result

    tabular_results = sorted(
        best_by_bins.values(),
        key=lambda item: item["bins"],
    )

    results = list(tabular_results)

    dqn_path = Path(args.dqn_eval)
    if dqn_path.exists():
        results.append(load_dqn_result(dqn_path))
    else:
        print(f"Warning: DQN evaluation file was not found: {dqn_path}")

    save_summary_csv(results, output_dir / "method_comparison.csv")

    labels = [result["method"] for result in results]

    bar_plot(
        labels,
        [result["best_success"] for result in results],
        "Best evaluation success rate",
        "Success rate",
        output_dir / "comparison_success.png",
    )

    bar_plot(
        labels,
        [result["best_distance"] for result in results],
        "Best evaluation final distance",
        "Mean final distance, m",
        output_dir / "comparison_distance.png",
    )

    bar_plot(
        labels,
        [result["best_return"] for result in results],
        "Best evaluation return",
        "Mean evaluation return",
        output_dir / "comparison_return.png",
    )

    print("\nDetected tabular result directories:")
    for directory in tabular_directories:
        print(f"  {directory}")

    print("\nComparison summary:")
    for result in results:
        print(
            f"{result['method']:<22} | "
            f"success={result['best_success']:.2f} | "
            f"distance={result['best_distance']:.4f} | "
            f"return={result['best_return']:.2f}"
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
