"""PPO for the rotation primitive with a curriculum over the requested turn.

    python -m stage1_cube.train_rotate --name rotate [--steps 15000000] [--n-envs 12]
"""
import argparse
import csv
import json
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from common import settings as cfg
from stage1_cube.rotate_env import CURRICULUM_TURNS, RotateEnv, fixed_tasks

VALIDATION_SEEDS = range(10000, 10200)
CURRICULUM_WINDOW = 300
CURRICULUM_PROMOTE = 0.60
PPO_SETTINGS = dict(learning_rate=3e-4, batch_size=1280, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, target_kl=0.02,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])))
TURN_BINS = [(0, 45), (45, 90), (90, 135), (135, 181)]


def evaluate(model, normalize, tasks):
    env, rows = RotateEnv(), []
    for task in tasks:
        obs, _ = env.reset(options={"task": task})
        while True:
            action, _ = model.predict(normalize.normalize_obs(obs[None])[0], deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rows.append(dict(success=info["is_success"], reason=info["reason"], turn_deg=info["turn_deg"],
                         position_error_cm=info["position_error_cm"], angle_error_deg=info["angle_error_deg"]))
    env.close()
    by_turn = {}
    for low, high in TURN_BINS:
        group = [r for r in rows if low <= r["turn_deg"] < high]
        by_turn[f"{low}-{min(high, 180)}"] = f"{sum(r['success'] for r in group)}/{len(group)}"
    return dict(success_rate=float(np.mean([r["success"] for r in rows])), by_turn=by_turn,
                position_cm=float(np.median([r["position_error_cm"] for r in rows])),
                angle_deg=float(np.median([r["angle_error_deg"] for r in rows])))


class Curriculum(BaseCallback):
    def __init__(self, run_dir, eval_every, eval_episodes, level=0):
        super().__init__()
        self.run_dir, self.eval_every, self.level = run_dir, eval_every, level
        self.tasks = fixed_tasks(list(VALIDATION_SEEDS)[:eval_episodes])
        self.window = deque(maxlen=CURRICULUM_WINDOW)
        self.best, self.last_eval, self.started = -1.0, -1, time.time()

    def _on_training_start(self):
        self.training_env.env_method("set_level", self.level)
        self.next_eval = (self.num_timesteps // self.eval_every + 1) * self.eval_every

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and info.get("level") == self.level:
                self.window.append(info["is_success"])
        if (self.level < len(CURRICULUM_TURNS) - 1 and len(self.window) == self.window.maxlen
                and np.mean(self.window) >= CURRICULUM_PROMOTE):
            self.level += 1
            self.window.clear()
            self.training_env.env_method("set_level", self.level)
            print(f"step {self.num_timesteps:,}: turns {CURRICULUM_TURNS[self.level]} deg", flush=True)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self.validate()
        return True

    def save(self, name):
        self.model.save(self.run_dir / f"{name}.zip")
        self.training_env.save(str(self.run_dir / f"{name}_vecnormalize.pkl"))

    def validate(self):
        if self.last_eval == self.num_timesteps:
            return
        self.last_eval = self.num_timesteps
        self.save("last")
        self.training_env.training = False
        result = evaluate(self.model, self.training_env, self.tasks)
        self.training_env.training = True
        row = dict(steps=self.num_timesteps, minutes=round((time.time() - self.started) / 60, 1), level=self.level,
                   val_success=round(result["success_rate"], 3), val_pos_cm=round(result["position_cm"], 2),
                   val_angle_deg=round(result["angle_deg"], 1), **{f"turn_{k}": v for k, v in result["by_turn"].items()})
        path = self.run_dir / "validation.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        if result["success_rate"] > self.best:
            self.best = result["success_rate"]
            self.save("best")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rotate")
    parser.add_argument("--steps", type=int, default=15_000_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_dir = cfg.RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([RotateEnv] * args.n_envs, start_method="spawn"))
    env.seed(args.seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", n_steps=7680 // args.n_envs, **PPO_SETTINGS)
    model.set_logger(configure(str(run_dir), ["csv"]))
    callback = Curriculum(run_dir, args.eval_every, args.eval_episodes)
    model.learn(total_timesteps=args.steps, callback=callback)
    callback.validate()
    env.close()


if __name__ == "__main__":
    main()
