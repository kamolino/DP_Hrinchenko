"""Stage 1, variant with a task manager: scripted approach + rotation policy, then the pushing model.

    python -m stage1_cube.manager --tasks 300 --seed 950000 --out ../results/stage1_manager
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from common import policy, settings as cfg
from common.block_sim import BlockSim
from stage1_cube.push_model import push_observation
from stage1_cube.rotate_env import decode
from stage1_cube.tasks import DEV_SEED, make_tasks

ROTATE_MODEL = cfg.MODELS / "stage1_rotate.zip"
ROTATE_NORM = cfg.MODELS / "stage1_rotate_vecnormalize.pkl"
PUSH_MODEL = cfg.MODELS / "stage1_push.zip"

SWITCH_TO_ROTATE = np.deg2rad(40)
ROTATE_DONE = np.deg2rad(30)
ROTATE_STEPS = 300
APPROACH_STEPS = 120
PUSH_STEPS = 500
TOTAL_STEPS = 2000


def load_rotation():
    model = policy.load(ROTATE_MODEL)
    with open(ROTATE_NORM, "rb") as f:
        normalize = pickle.load(f)
    normalize.training = False
    return model, normalize


def approach_action(sim):
    """Towards a point 3 cm outside the block on the side facing the tool."""
    direction = sim.tcp[:2] - sim.object_xy
    direction /= max(np.linalg.norm(direction), 1e-6)
    local = np.abs(np.array([[np.cos(sim.yaw), np.sin(sim.yaw)], [-np.sin(sim.yaw), np.cos(sim.yaw)]]) @ direction)
    point = sim.object_xy + (np.dot(local, cfg.OBJECT_HALF[:2]) + 0.03) * direction
    error = point - sim.tcp[:2]
    return error / cfg.ACTION_SCALE / 4, np.linalg.norm(error) < 0.01


class Manager:
    def __init__(self, retries=2):
        self.rot, self.rot_norm = load_rotation()
        self.push = policy.load(PUSH_MODEL)
        self.retries = retries
        self.sim = BlockSim()

    def approach(self, log, limit):
        sim = self.sim
        for _ in range(min(APPROACH_STEPS, limit)):
            xy, near = approach_action(sim)
            success, reason = sim.act(xy, -1.0 if near else 0.5)
            log.append("approach")
            if reason or success or (near and sim.tcp[2] < cfg.TCP_HEIGHT + 0.005):
                return success, reason
        return False, ""

    def rotate(self, log, limit):
        sim = self.sim
        for _ in range(min(ROTATE_STEPS, limit)):
            action, _ = self.rot.predict(self.rot_norm.normalize_obs(sim.observation()[None])[0], deterministic=True)
            success, reason = sim.act(decode(action), -1.0)
            log.append("rotate")
            if reason or success or sim.angle_error() <= ROTATE_DONE:
                return success, reason
        return False, ""

    def push_phase(self, log, limit):
        sim = self.sim
        for _ in range(min(PUSH_STEPS, limit)):
            action, _ = self.push.predict(push_observation(sim), deterministic=True)
            action = np.clip(action, -1, 1)
            success, reason = sim.act(action[:2], action[2])
            log.append("push")
            if reason or success:
                return success, reason
        return False, "timeout"

    def run(self, task):
        sim = self.sim
        sim.reset(task)
        log, success, reason, cycles = [], False, "", 0
        start_angle, rotated_angle = float(np.rad2deg(sim.angle_error())), None
        while True:
            if sim.angle_error() > (SWITCH_TO_ROTATE if cycles == 0 else ROTATE_DONE):
                success, reason = self.approach(log, TOTAL_STEPS - len(log))
                if not (success or reason):
                    success, reason = self.rotate(log, TOTAL_STEPS - len(log))
            if rotated_angle is None:
                rotated_angle = float(np.rad2deg(sim.angle_error()))
            if not (success or reason):
                success, reason = self.push_phase(log, TOTAL_STEPS - len(log))
            cycles += 1
            if success or (reason and reason != "timeout"):
                break
            if len(log) >= TOTAL_STEPS:
                reason = "total_limit"
                break
            if cycles > self.retries or sim.angle_error() <= ROTATE_DONE:
                break
            reason = ""
        return dict(id=task["id"], success=bool(success), reason="success" if success else reason,
                    start_angle_deg=round(start_angle, 1), after_rotate_deg=round(rotated_angle, 1),
                    angle_error_deg=round(float(np.rad2deg(sim.angle_error())), 1),
                    position_error_cm=round(100 * sim.distance(), 2), cycles=cycles,
                    rotate_steps=log.count("rotate"), push_steps=log.count("push"), steps=len(log))


def summary(rows):
    failures = {}
    for r in rows:
        if not r["success"]:
            failures[r["reason"]] = failures.get(r["reason"], 0) + 1
    return dict(success=sum(r["success"] for r in rows), episodes=len(rows), failures=failures,
                median_cm=float(np.median([r["position_error_cm"] for r in rows])),
                median_deg=float(np.median([r["angle_error_deg"] for r in rows])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out", default=str(cfg.ROOT / "results" / "stage1_manager"))
    args = parser.parse_args()
    manager = Manager(args.retries)
    rows = []
    for task in make_tasks(args.tasks, args.seed):
        rows.append(manager.run(task))
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(seed=args.seed, retries=args.retries, **summary(rows))
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
