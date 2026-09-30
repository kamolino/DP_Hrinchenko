#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
model="${1:-models/ppo_pushing.zip}"
if [ "$#" -gt 0 ]; then shift; fi
exec .venv/bin/python code/run_pushing_demo.py "$model" "$@"
