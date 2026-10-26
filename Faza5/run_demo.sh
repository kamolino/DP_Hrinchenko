#!/usr/bin/env bash
# bash run_demo.sh [cube|cube-manager|word|word-shared] [--index N] [--gif file.gif]
set -euo pipefail
cd "$(dirname "$0")/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mode="${1:-word}"
if [ "$#" -gt 0 ]; then shift; fi
exec ../.venv/bin/python demo.py "$mode" "$@"
