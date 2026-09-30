#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fr3-pushing-mpl"
output="${1:-runs/test_$(date +%Y%m%d_%H%M%S_%N)}"
exec .venv/bin/python -u code/evaluate_pushing_ppo.py models/ppo_pushing.zip \
  --tasks results/tasks.json --output "$output"
