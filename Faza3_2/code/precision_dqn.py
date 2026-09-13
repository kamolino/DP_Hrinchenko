import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import mujoco
import torch
from torch import nn
from precision_environment import PrecisionEnv, PrecisionConfig

# ============================================================
# 1. LOCAL DQN NETWORK
# ============================================================


class ActionDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def action(self, state):
        with torch.no_grad():
            return int(self(torch.as_tensor(state)).argmax().item())


# ============================================================
# 2. TRAINING EXAMPLES
# ============================================================


class PhysicsTeacher:
    """Evaluate constant joint commands in private simulator data for 150 ms."""

    def __init__(self, env):
        self.env = env
        self.data = mujoco.MjData(env.model)

    def scores(self):
        env = self.env
        commands, _, _ = env.commands()
        scores = []
        for command in commands:
            mujoco.mj_copyData(self.data, env.model, env.data)
            self.data.ctrl[env.ids.actuator_ids] = command
            collision = False
            for k in range(env.config.frame_skip * 3):
                mujoco.mj_step(env.model, self.data)
                if k % env.config.frame_skip == 0:
                    collision |= env.collision(self.data)
            error = np.linalg.norm(env.goal - self.data.site_xpos[env.ids.ee_site_id])
            # Score each action by the predicted remaining error.
            scores.append(
                4 * (env.distance - error) / max(env.distance, env.config.tolerance)
                - (5 if collision else 0)
            )
        return np.asarray(scores, dtype=np.float32)


def save_model(path, net, config, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "algorithm": "Double DQN with physics-teacher pretraining",
            "state_dict": net.state_dict(),
            "env_config": asdict(config),
            "metadata": metadata,
        },
        path,
    )


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("Not a precision DQN checkpoint")
    net = ActionDQN()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, PrecisionConfig(**checkpoint["env_config"]), checkpoint["metadata"]


def task_suite(seed=90210, count=30):
    env = PrecisionEnv(seed=seed)
    tasks = []
    for name, spread, start in [
        ("near", 0.15, False),
        ("medium", 0.4, False),
        ("wide", 0.8, False),
        ("random_start", 0.4, True),
    ]:
        for _ in range(count):
            task = env.sample_task(spread, start)
            task["group"] = name
            tasks.append(task)
    return tasks


def evaluate(net, config, tasks, trajectory=False):
    env = PrecisionEnv(config)
    rows = []
    paths = []
    for i, task in enumerate(tasks):
        state = env.reset(task)
        positions = [env.ee().tolist()]
        for _ in range(config.max_steps):
            state, _, done, info = env.step(net.action(state))
            positions.append(env.ee().tolist())
            if done:
                break
        rows.append(dict(task=i, group=task["group"], **info))
        if trajectory:
            paths.append(dict(task=task, positions=positions))
    summary = {}
    for group in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == group]
        d = np.array([r["distance"] for r in sub])
        summary[group] = dict(
            episodes=len(sub),
            success_rate=float(np.mean([r["success"] for r in sub])),
            mean_error_mm=float(d.mean() * 1000),
            p95_error_mm=float(np.percentile(d, 95) * 1000),
            collisions=sum(r["collision"] for r in sub),
        )
    return summary, rows, paths
