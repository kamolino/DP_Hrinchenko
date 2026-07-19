#!/usr/bin/env bash
set -euo pipefail

# Resolve the project root from the location of this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="$PROJECT_ROOT/code/tabular_q_learning.py"
COMPARE_SCRIPT="$PROJECT_ROOT/code/compare_tabular_and_dqn.py"
TABULAR_ROOT="$PROJECT_ROOT/results/tabular"

run_experiment() {
    local bins="$1"
    local output_dir="$TABULAR_ROOT/tabular_exp_bins${bins}"

    echo
    echo "============================================================"
    echo "Training tabular Q-learning with ${bins} bins"
    echo "Output directory: ${output_dir}"
    echo "============================================================"

    mkdir -p "$output_dir"

    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --bins "$bins" \
        --episodes 3000 \
        --action-step-deg 1.0 \
        --alpha 0.10 \
        --gamma 0.99 \
        --epsilon-start 1.0 \
        --epsilon-end 0.05 \
        --epsilon-decay 0.9985 \
        --eval-every 100 \
        --eval-episodes 100 \
        --seed 42 \
        --output-dir "$output_dir"
}

run_experiment 5
run_experiment 7
run_experiment 9

echo
echo "============================================================"
echo "All three experiments are complete."
echo "Running the comparison script."
echo "============================================================"

"$PYTHON_BIN" "$COMPARE_SCRIPT"

echo
echo "Expected result directories:"
echo "  $TABULAR_ROOT/tabular_exp_bins5"
echo "  $TABULAR_ROOT/tabular_exp_bins7"
echo "  $TABULAR_ROOT/tabular_exp_bins9"
