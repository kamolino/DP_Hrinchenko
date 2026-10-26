"""PPO fine-tuning of the DAgger student with a decaying BC term.

    python -m stage1_cube.finetune --init ../runs/stage1_dagger/student_iter4.zip [--steps 6000000]
"""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from common import rollout, settings as cfg
from common.training import BCPPO, FINETUNE, DevEval, vec_env, warm_up_critic
from stage1_cube.cube_env import CubeEnv
from stage1_cube.tasks import DEV_SEED, make_tasks, turn_groups

TRAIN_SEED = 700000


def evaluate(dev, workers, path):
    rows, _, _ = rollout.run(dev, CubeEnv, student_path=path, workers=workers)
    return dict(rollout.summary(rows), groups=turn_groups(rows, dev))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--dataset", default=str(cfg.RUNS / "stage1_dagger" / "dataset.npz"))
    parser.add_argument("--name", default="stage1_ppo")
    parser.add_argument("--steps", type=int, default=6_000_000)
    parser.add_argument("--warmup", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = cfg.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    env = vec_env([partial(CubeEnv, task_seed=TRAIN_SEED + i) for i in range(args.n_envs)])
    model = BCPPO.load(Path(args.init).resolve(), env=env, device="cpu", custom_objects=FINETUNE)
    data = np.load(args.dataset)
    model.bc_data = (torch.as_tensor(data["obs"]), torch.as_tensor(data["act"]))
    model.set_logger(configure(str(out), ["csv"]))
    callback = DevEval(out, args.eval_every, partial(evaluate, make_tasks(300, DEV_SEED), args.workers))
    warm_up_critic(model, args.warmup)
    callback.init_callback(model)
    callback.evaluate()
    model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
    callback.evaluate()
    env.close()


if __name__ == "__main__":
    main()
