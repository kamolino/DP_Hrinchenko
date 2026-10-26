"""DAgger: distil the stage-1 manager (teacher.py) into one policy with 30 inputs and x/y/z actions.

    python -m stage1_cube.dagger [--iterations 5] [--tasks 300]
"""
import argparse
import json
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from common import rollout, settings as cfg
from common.policy import BoundedMeanPolicy
from common.training import DAGGER_BETAS, fit
from stage1_cube.cube_env import CubeEnv
from stage1_cube.tasks import DEV_SEED, make_tasks, turn_groups
from stage1_cube.teacher import Teacher

TRAIN_SEED = 600000


def new_student():
    return PPO(BoundedMeanPolicy, CubeEnv(), device="cpu", verbose=0, seed=0,
               policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                  min_action_std=0.05, max_action_std=0.5, log_std_init=np.log(0.2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--name", default="stage1_dagger")
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = cfg.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = make_tasks(300, DEV_SEED)
    model = new_student()
    data_obs, data_act, log = [], [], []
    for it in range(args.iterations):
        started = time.time()
        beta = DAGGER_BETAS[min(it, len(DAGGER_BETAS) - 1)]
        driver = "teacher" if it == 0 else "student"
        rows, obs, act = rollout.run(make_tasks(args.tasks, TRAIN_SEED + it), CubeEnv, Teacher, driver,
                                     out / "student.zip" if it else None, label=True, beta=beta,
                                     workers=args.workers, seed=it * 100)
        data_obs.append(obs)
        data_act.append(act)
        loss = fit(model, np.concatenate(data_obs), np.concatenate(data_act), args.epochs, lr=1e-3)
        model.save(out / "student.zip")
        model.save(out / f"student_iter{it}.zip")
        dev_rows, _, _ = rollout.run(dev, CubeEnv, student_path=out / "student.zip", workers=args.workers)
        entry = dict(iteration=it, driver=driver, beta=beta, samples=sum(len(o) for o in data_obs), bc_mse=round(loss, 4),
                     dev=dict(rollout.summary(dev_rows), groups=turn_groups(dev_rows, dev)),
                     minutes=round((time.time() - started) / 60, 1))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        (out / "dagger_log.json").write_text(json.dumps(log, indent=1))
    np.savez_compressed(out / "dataset.npz", obs=np.concatenate(data_obs), act=np.concatenate(data_act))


if __name__ == "__main__":
    main()
