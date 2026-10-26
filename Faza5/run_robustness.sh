#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="../runs/robustness_$(date +%Y%m%d_%H%M%S).json"
exec ../.venv/bin/python robustness.py --out "$out" "$@"
