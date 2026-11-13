"""Evaluate the one-policy block model on fixed tasks.

    python -m stage1_cube.evaluate --tasks 300 --seed 950000
"""
import argparse
import json
from pathlib import Path

from common import rollout, settings as cfg
from stage1_cube.cube_env import CubeEnv
from stage1_cube.tasks import DEV_SEED, make_tasks, turn_groups

POLICY = cfg.MODELS / "stage1_cube_policy.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY))
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(cfg.ROOT / "results" / "stage1_cube.json"))
    args = parser.parse_args()
    tasks = make_tasks(args.tasks, args.seed)
    rows, _, _ = rollout.run(tasks, CubeEnv, student_path=args.policy, workers=args.workers)
    result = dict(seed=args.seed, **rollout.summary(rows), groups=turn_groups(rows, tasks))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=result, rows=rows), indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
