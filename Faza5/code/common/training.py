"""Behaviour cloning (DAgger) and PPO fine-tuning with a decaying BC term, shared by all stages."""
import json
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

FINETUNE = dict(learning_rate=3e-5, n_steps=1024, batch_size=2048, n_epochs=10, clip_range=0.1, ent_coef=0.0,
                target_kl=0.02, gamma=0.99, gae_lambda=0.95)
DAGGER_BETAS = [1.0, 0.5, 0.25, 0.0, 0.0, 0.0]


def fit(model, obs, act, epochs, lr, batch=4096):
    """Regress the policy mean on the teacher actions (MSE); returns the last epoch loss."""
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    x, y = torch.as_tensor(obs), torch.as_tensor(act)
    n, total = len(x), 0.0
    for _ in range(epochs):
        order, total = torch.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            loss = ((policy.get_distribution(x[idx]).distribution.mean - y[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
    return total / n


class BCPPO(PPO):
    """PPO; after every update a few BC steps pull the mean towards the DAgger data (weight decays to 0)."""
    bc_data = None
    bc_start, bc_decay_steps, bc_steps, bc_batch = 1.0, 1_000_000, 20, 2048

    def bc_coef(self):
        return self.bc_start * max(0.0, 1.0 - self.num_timesteps / self.bc_decay_steps)

    def train(self):
        super().train()
        coef = self.bc_coef()
        if coef <= 0 or self.bc_data is None:
            return
        obs, act = self.bc_data
        losses = []
        for _ in range(self.bc_steps):
            idx = torch.randint(len(obs), (self.bc_batch,))
            loss = coef * ((self.policy.get_distribution(obs[idx]).distribution.mean - act[idx]) ** 2).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item() / coef)
        self.logger.record("train/bc_mse", float(np.mean(losses)))
        self.logger.record("train/bc_coef", coef)


def vec_env(factories):
    return VecNormalize(VecMonitor(SubprocVecEnv(factories, start_method="spawn")),
                        norm_obs=False, norm_reward=True, gamma=0.99)


def actor_parameters(policy):
    return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(), policy.log_std]


def warm_up_critic(model, steps):
    """Train only the value network (the actor is frozen) before PPO touches the policy."""
    for p in actor_parameters(model.policy):
        p.requires_grad_(False)
    bc, model.bc_start = getattr(model, "bc_start", 0.0), 0.0
    if steps:
        model.learn(total_timesteps=steps, reset_num_timesteps=True)
    for p in actor_parameters(model.policy):
        p.requires_grad_(True)
    model.bc_start = bc
    model.num_timesteps = 0


class DevEval(BaseCallback):
    """Every `every` steps: save last.zip, evaluate it on the dev tasks, keep the best."""

    def __init__(self, out, every, evaluate, patience=0):
        super().__init__()
        self.out, self.every, self.evaluate_fn, self.patience = out, every, evaluate, patience
        self.next, self.best, self.no_gain, self.rows, self.started = every, -1, 0, [], time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.evaluate()
            if self.patience and self.no_gain >= self.patience:
                return False
        return True

    def evaluate(self):
        self.model.save(self.out / "last.zip")
        result = self.evaluate_fn(self.out / "last.zip")
        entry = dict(steps=self.num_timesteps, minutes=round((time.time() - self.started) / 60, 1),
                     bc_coef=round(getattr(self.model, "bc_coef", lambda: 0.0)(), 3), **result)
        self.rows.append(entry)
        if result["success"] > self.best:
            self.best, self.no_gain = result["success"], 0
            self.model.save(self.out / "best.zip")
        else:
            self.no_gain += 1
        (self.out / "dev_log.json").write_text(json.dumps(self.rows, indent=1))
        print(json.dumps(entry), flush=True)
