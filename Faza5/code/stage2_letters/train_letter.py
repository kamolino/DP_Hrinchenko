"""Step 1 for a letter policy: PPO fine-tuning of the one-policy block model on single-letter tasks.

    python -m stage2_letters.train_letter --letter A [--steps 3000000]
"""
import argparse
from functools import partial

import torch
from stable_baselines3.common.logger import configure

from common import rollout, settings as cfg
from common.training import BCPPO, FINETUNE, DevEval, vec_env, warm_up_critic
from stage1_cube.evaluate import POLICY as CUBE_POLICY
from stage2_letters.letter_env import DEV_SEEDS, TRAIN_SEEDS, LetterEnv, make_tasks


def evaluate(letter, dev, workers, path):
    rows, _, _ = rollout.run(dev, partial(LetterEnv, letter), student_path=path, workers=workers)
    return rollout.summary(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", required=True, choices=["A", "R", "M"])
    parser.add_argument("--init", default=str(CUBE_POLICY))
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--warmup", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = cfg.RUNS / f"stage2_letter_{args.letter}"
    out.mkdir(parents=True, exist_ok=True)
    env = vec_env([partial(LetterEnv, args.letter, TRAIN_SEEDS[args.letter] + i) for i in range(args.n_envs)])
    model = BCPPO.load(args.init, env=env, device="cpu", custom_objects=FINETUNE)
    model.bc_start = 0.0
    model.set_logger(configure(str(out), ["csv"]))
    dev = make_tasks(args.letter, 200, DEV_SEEDS[args.letter])
    callback = DevEval(out, args.eval_every, partial(evaluate, args.letter, dev, args.workers))
    warm_up_critic(model, args.warmup)
    callback.init_callback(model)
    callback.evaluate()
    model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
    callback.evaluate()
    env.close()


if __name__ == "__main__":
    main()
