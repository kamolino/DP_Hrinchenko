#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="../runs/test_$(date +%Y%m%d_%H%M%S)"
exec ../.venv/bin/python evaluate_final.py --out "$out" "$@"
