#!/usr/bin/env bash
set -euo pipefail
phase_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$phase_dir/.venv/bin/python" "$phase_dir/code/run_workspace_dqn.py" \
  --tasks "$phase_dir/results/workspace/demo_tasks.json" --viewer "$@"
