import numpy as np

from common import settings as cfg
from common.geometry import fits_workspace, wrap

DEV_SEED = 500000
TEST_SEED = 950000


def make_tasks(count, seed):
    """Block pose tasks: any start yaw, turn 0-180 deg, move 0-15 cm."""
    rng, tasks = np.random.default_rng(seed), []
    while len(tasks) < count:
        obj = rng.uniform([0.40, -0.06], [0.50, 0.06])
        yaw = rng.uniform(-np.pi, np.pi)
        turn = rng.choice([-1, 1]) * rng.uniform(0, np.pi)
        angle, length = rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.15)
        goal = obj + length * np.array([np.cos(angle), np.sin(angle)])
        goal_yaw = wrap(yaw + turn)
        if not (np.all(goal > [0.38, -0.12]) and np.all(goal < [0.56, 0.12])):
            continue
        if length < cfg.SUCCESS_DISTANCE and abs(turn) < cfg.SUCCESS_ANGLE:
            continue
        if fits_workspace(obj, yaw) and fits_workspace(goal, goal_yaw):
            tasks.append(dict(id=len(tasks), object=obj.tolist(), goal=goal.tolist(),
                              yaw=float(yaw), goal_yaw=float(goal_yaw)))
    return tasks


def turn_groups(rows, tasks):
    turn = {t["id"]: abs(np.rad2deg(wrap(t["goal_yaw"] - t["yaw"]))) for t in tasks}
    groups = {}
    for low, high in [(0, 40), (40, 90), (90, 135), (135, 181)]:
        group = [r for r in rows if low <= turn[r["id"]] < high]
        groups[f"{low}-{min(high, 180)}"] = f"{sum(r['success'] for r in group)}/{len(group)}"
    return groups
