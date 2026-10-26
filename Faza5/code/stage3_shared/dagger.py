"""DAgger: distil the three letter policies into one shared policy (33 inputs).

Without --init the student starts from letter A's weights; the three one-hot inputs get small random weights
(0.1 x std of the first layer) instead of zeros, otherwise they barely learn at a small learning rate.

    python -m stage3_shared.dagger --iterations 4 [--init ../models/stage3_shared.zip]
"""
import argparse
import json
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from common import policy, rollout, settings as cfg
from common.policy import BoundedMeanPolicy
from common.training import DAGGER_BETAS, fit
from stage2_letters.word_manager import LETTER_POLICIES
from stage3_shared.shared_env import LETTERS, SharedLetterEnv, Teachers, dagger_tasks, dev_tasks


def new_student(init=LETTER_POLICIES["A"], noise=0.1, seed=0):
    source = policy.load(init)
    model = PPO(BoundedMeanPolicy, SharedLetterEnv(), device="cpu", verbose=0, seed=seed,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   min_action_std=0.05, max_action_std=0.5, log_std_init=np.log(0.2)))
    old, new = source.policy.state_dict(), model.policy.state_dict()
    generator = torch.Generator().manual_seed(seed)
    for key, value in new.items():
        if old[key].shape == value.shape:
            new[key] = old[key].clone()
        else:
            scale = noise * old[key].std()
            new[key] = torch.cat([old[key], scale * torch.randn(value.shape[0], 3, generator=generator)], dim=1)
    model.policy.load_state_dict(new)
    return model


def letter_summary(rows):
    out = rollout.summary(rows)
    out["letters"] = {n: f"{sum(r['success'] for r in rows if r['letter'] == n)}/{sum(r['letter'] == n for r in rows)}"
                      for n in LETTERS}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--tasks-per-letter", type=int, default=100)
    parser.add_argument("--dev-per-letter", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init", help="continue from an existing shared policy")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--name", default="stage3_dagger")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = cfg.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = dev_tasks(args.dev_per_letter)
    model = policy.load(args.init, env=SharedLetterEnv()) if args.init else new_student()
    data_obs, data_act, log = [], [], []
    for it in range(args.iterations):
        started = time.time()
        beta = DAGGER_BETAS[min(it, len(DAGGER_BETAS) - 1)]
        driver = "teacher" if it == 0 else "student"
        rows, obs, act = rollout.run(dagger_tasks(args.tasks_per_letter, it), SharedLetterEnv, Teachers, driver,
                                     out / "student.zip" if it else None, label=True, beta=beta,
                                     workers=args.workers, seed=it * 100)
        data_obs.append(obs)
        data_act.append(act)
        loss = fit(model, np.concatenate(data_obs), np.concatenate(data_act), args.epochs, args.lr)
        model.save(out / "student.zip")
        model.save(out / f"student_iter{it}.zip")
        np.savez_compressed(out / "dataset.npz", obs=np.concatenate(data_obs), act=np.concatenate(data_act))
        dev_rows, _, _ = rollout.run(dev, SharedLetterEnv, student_path=out / "student.zip", workers=args.workers)
        entry = dict(iteration=it, driver=driver, beta=beta, samples=sum(len(o) for o in data_obs),
                     bc_mse=round(loss, 5), dev=letter_summary(dev_rows), minutes=round((time.time() - started) / 60, 1))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        (out / "dagger_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
