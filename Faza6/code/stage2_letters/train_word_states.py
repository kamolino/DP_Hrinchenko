"""Step 2 for a letter policy: PPO on states recorded from the word manager (75 %) + single tasks (25 %).
A3: --single-fraction 1.0 (single tasks only), A4: --disturb-position 0 --disturb-angle 0.

    python -m stage2_letters.word_manager --scenes 600 --seed 645000 --record --out ../runs/word_states
    python -m stage2_letters.train_word_states --letter A --init ../runs/stage2_letter_A/best.zip
"""
import argparse
import json
from functools import partial
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from common import policy  # noqa: F401, needed to load the saved letter models
from common import settings as cfg
from common.training import warm_up_critic
from stage2_letters.word_state_env import DISTURB_ANGLE, DISTURB_POSITION, MixedEnv

STATES = cfg.NEW_MODELS / "start" / "word_states_645000.json"
START = cfg.NEW_MODELS / "start" / "stage2_letter_{}_single.zip"
SEEDS = {"A": 660000, "R": 661000, "M": 662000}


class Checkpoints(BaseCallback):
    def __init__(self, out, every):
        super().__init__()
        self.out, self.every, self.next = out, every, every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.model.save(self.out / f"steps_{self.next}.zip")
            self.next += self.every
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", required=True, choices=["A", "R", "M"])
    parser.add_argument("--init", help="default: models/start/stage2_letter_<letter>_single.zip")
    parser.add_argument("--states", default=str(STATES))
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--single-fraction", type=float, default=MixedEnv.SINGLE_FRACTION)
    parser.add_argument("--disturb-position", type=float, default=DISTURB_POSITION)
    parser.add_argument("--disturb-angle", type=float, default=DISTURB_ANGLE)
    parser.add_argument("--seed-offset", type=int, default=0, help="training seed = letter seed + offset")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    init = args.init or str(START).format(args.letter)
    out = Path(args.out) if args.out else cfg.RUNS / f"stage2_word_{args.letter}"
    out.mkdir(parents=True, exist_ok=True)
    seed = SEEDS[args.letter] + args.seed_offset
    reward = dict(disturb_position=args.disturb_position, disturb_angle=args.disturb_angle)
    env = VecNormalize(VecMonitor(SubprocVecEnv([partial(MixedEnv, args.letter, args.states, seed + i,
                                                         args.single_fraction, **reward)
                                                 for i in range(args.n_envs)], start_method="spawn")),
                       norm_obs=False, norm_reward=True, gamma=0.99)
    (out / "config.json").write_text(json.dumps(dict(vars(args), init=init, seed=seed), indent=1))
    model = PPO.load(init, env=env, device="cpu", custom_objects=dict(
        learning_rate=3e-5, n_steps=1024, batch_size=1536, n_epochs=10, clip_range=0.1, ent_coef=0.0,
        target_kl=0.02, gamma=0.99, gae_lambda=0.95))
    model.set_random_seed(seed)
    model.set_logger(configure(str(out), ["csv"]))
    warm_up_critic(model, args.warmup)
    model.learn(total_timesteps=args.steps, callback=Checkpoints(out, args.save_every), reset_num_timesteps=False)
    model.save(out / "last.zip")
    env.close()


if __name__ == "__main__":
    main()
