#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fr3-pushing-mpl"
output="${1:-runs/train_$(date +%Y%m%d_%H%M%S_%N)}"
exec .venv/bin/python -u code/train_pushing_ppo.py \
  --stage C --steps 1500000 --seed 42 --n-envs 4 \
  --resume reference/faza14.zip --output "$output" \
  --eval-every 200000 --eval-episodes 200
