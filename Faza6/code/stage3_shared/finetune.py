"""Optional PPO after DAgger for the shared policy, on recorded word states + single tasks.
It gave no gain over DAgger (word 84 -> 84 / 83 on dev), so the submitted model is the DAgger student.

    python -m stage3_shared.finetune --init ../runs/stage3_dagger/student_iter3.zip
"""
import argparse
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from common import settings as cfg
from common.training import warm_up_critic
from stage2_letters.train_word_states import STATES, Checkpoints
from stage2_letters.word_state_env import MixedEnv
from stage3_shared.shared_env import LETTERS, ONE_HOT


class SharedWordEnv(gym.Env):
    def __init__(self, seed, states=str(STATES)):
        self.envs = {n: MixedEnv(n, states, task_seed=seed + 1000 * i) for i, n in enumerate(LETTERS)}
        self.rng = np.random.default_rng(seed)
        self.letter = "A"
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.letter = LETTERS[int(self.rng.integers(3))]
        obs, info = self.envs[self.letter].reset()
        return np.concatenate([obs, ONE_HOT[self.letter]]).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[self.letter].step(action)
        return np.concatenate([obs, ONE_HOT[self.letter]]).astype(np.float32), reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = cfg.RUNS / "stage3_ppo"
    out.mkdir(parents=True, exist_ok=True)
    env = VecNormalize(VecMonitor(SubprocVecEnv([partial(SharedWordEnv, 670000 + 10 * i) for i in range(args.n_envs)],
                                                start_method="spawn")), norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO.load(args.init, env=env, device="cpu", custom_objects=dict(
        learning_rate=3e-5, n_steps=1024, batch_size=2048, n_epochs=10, clip_range=0.1, ent_coef=0.0,
        target_kl=0.02, gamma=0.99, gae_lambda=0.95))
    model.set_logger(configure(str(out), ["csv"]))
    warm_up_critic(model, args.warmup)
    model.learn(total_timesteps=args.steps, callback=Checkpoints(out, args.save_every), reset_num_timesteps=False)
    model.save(out / "last.zip")
    env.close()


if __name__ == "__main__":
    main()
