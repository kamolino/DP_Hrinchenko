"""Step 2 for a letter policy: PPO on states recorded from the word manager (75 %) + single tasks (25 %).

    python -m stage2_letters.word_manager --scenes 600 --seed 645000 --record --out ../runs/word_states
    python -m stage2_letters.train_word_states --letter A --init ../runs/stage2_letter_A/best.zip
"""
import argparse
from functools import partial

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from common import settings as cfg
from common.training import warm_up_critic
from stage2_letters.word_state_env import MixedEnv

STATES = cfg.RUNS / "word_states" / "states.json"
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
    parser.add_argument("--init", required=True)
    parser.add_argument("--states", default=str(STATES))
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = cfg.RUNS / f"stage2_word_{args.letter}"
    out.mkdir(parents=True, exist_ok=True)
    seed = SEEDS[args.letter]
    env = VecNormalize(VecMonitor(SubprocVecEnv([partial(MixedEnv, args.letter, args.states, seed + i)
                                                 for i in range(args.n_envs)], start_method="spawn")),
                       norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO.load(args.init, env=env, device="cpu", custom_objects=dict(
        learning_rate=3e-5, n_steps=1024, batch_size=1536, n_epochs=10, clip_range=0.1, ent_coef=0.0,
        target_kl=0.02, gamma=0.99, gae_lambda=0.95))
    model.set_logger(configure(str(out), ["csv"]))
    warm_up_critic(model, args.warmup)
    model.learn(total_timesteps=args.steps, callback=Checkpoints(out, args.save_every), reset_num_timesteps=False)
    model.save(out / "last.zip")
    env.close()


if __name__ == "__main__":
    main()
