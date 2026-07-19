# Reinforcement Learning Control of a 2-DOF Manipulator

This project compares two reinforcement learning methods for controlling a planar 2-DOF manipulator in MuJoCo:

- Deep Q-Network (DQN)
- Tabular Q-learning

The agent must move the manipulator end-effector to a target position.

## 1. Project structure

```text
Faza3_Final/
├── code/
│   ├── dqn_train_visual_v7_1_logging.py
│   ├── evaluate_dqn_trajectory.py
│   ├── plot_dqn_training_results.py
│   ├── tabular_q_learning.py
│   ├── plot_tabular_results.py
│   ├── compare_tabular_and_dqn.py
│   └── run_all_tabular_bins_separate.sh
├── models/
├── mujoco/
│   └── reset_pos_stable.xml
├── results/
│   ├── dqn/
│   ├── tabular/
│   └── comparison/
└── README.md
```

## 2. Requirements

Recommended environment:

- Ubuntu 22.04 or newer
- Python 3.10 or newer
- MuJoCo 3.x

## 3. Installation

Open a terminal in the project directory:

```bash
cd Faza3_Final
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python3 -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install numpy matplotlib pandas torch mujoco
```

## 4. Quick verification

The following commands allow the project to be checked without changing the source code.

### 4.1 Train the DQN agent

```bash
python3 code/dqn_train_visual_v7_1_logging.py
```

The script saves trained models to:

```text
models/
```

Training and evaluation logs are saved to:

```text
results/dqn/
```

Expected files include:

```text
models/dqn_best.pt
models/dqn_last.pt
results/dqn/dqn_train_history.csv
results/dqn/dqn_eval_history.csv
```

### 4.2 Generate DQN plots

```bash
python3 code/plot_dqn_training_results.py
```

Generated plots are saved to:

```text
results/dqn/
```

### 4.3 Evaluate the trained DQN trajectory

```bash
python3 code/evaluate_dqn_trajectory.py
```

The script loads the trained DQN model and saves the trajectory data and plot to:

```text
results/dqn/
```

### 4.4 Run all tabular Q-learning experiments

Make the script executable:

```bash
chmod +x code/run_all_tabular_bins_separate.sh
```

Run all experiments:

```bash
./code/run_all_tabular_bins_separate.sh
```

The script trains tabular Q-learning agents with:

- 5 bins
- 7 bins
- 9 bins

Results are saved separately:

```text
results/tabular/tabular_exp_bins5/
results/tabular/tabular_exp_bins7/
results/tabular/tabular_exp_bins9/
```

Each directory contains the corresponding Q-table, metadata, training history, and evaluation history.

### 4.5 Compare tabular Q-learning and DQN

```bash
python3 code/compare_tabular_and_dqn.py
```

Comparison results are saved to:

```text
results/comparison/
```

Expected files include:

```text
results/comparison/method_comparison.csv
results/comparison/comparison_success.png
results/comparison/comparison_distance.png
results/comparison/comparison_return.png
```

## 5. Recommended execution order

For a complete reproduction of the experiments, run:

```bash
cd Faza3_Final
source .venv/bin/activate

python3 code/dqn_train_visual_v7_1_logging.py
python3 code/plot_dqn_training_results.py
python3 code/evaluate_dqn_trajectory.py

chmod +x code/run_all_tabular_bins_separate.sh
./code/run_all_tabular_bins_separate.sh

python3 code/compare_tabular_and_dqn.py
```

## 6. Run one tabular experiment manually

Example for 7 bins:

```bash
python3 code/tabular_q_learning.py \
    --bins 7 \
    --episodes 3000 \
    --output-dir results/tabular/tabular_exp_bins7
```

Example for 5 bins:

```bash
python3 code/tabular_q_learning.py \
    --bins 5 \
    --episodes 3000 \
    --output-dir results/tabular/tabular_exp_bins5
```

Example for 9 bins:

```bash
python3 code/tabular_q_learning.py \
    --bins 9 \
    --episodes 3000 \
    --output-dir results/tabular/tabular_exp_bins9
```

## 7. Notes

- All paths are resolved relative to the project directory.
- The MuJoCo model is loaded from `mujoco/reset_pos_stable.xml`.
- DQN training may take significantly longer than plotting or evaluation.
- Tabular experiments with 9 bins require more memory and training time than experiments with 5 or 7 bins.
- Existing result files may be overwritten when an experiment is run again.
- Run all commands from the root directory `Faza3_Final`.

## 8. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the dependencies:

```bash
source .venv/bin/activate
pip install numpy matplotlib pandas torch mujoco
```

### MuJoCo XML file not found

Make sure this file exists:

```text
mujoco/reset_pos_stable.xml
```

Run the scripts from the project root:

```bash
cd Faza3_Final
```

### DQN model not found during evaluation

Train the DQN model first:

```bash
python3 code/dqn_train_visual_v7_1_logging.py
```

Then run:

```bash
python3 code/evaluate_dqn_trajectory.py
```

### Missing tabular experiment during comparison

Make sure all three directories exist:

```text
results/tabular/tabular_exp_bins5/
results/tabular/tabular_exp_bins7/
results/tabular/tabular_exp_bins9/
```

If one is missing, run:

```bash
./code/run_all_tabular_bins_separate.sh
```
