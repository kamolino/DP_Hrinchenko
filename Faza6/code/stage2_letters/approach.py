"""Scripted approach: above the nearest free point next to the active letter, then down to pushing height."""
import numpy as np

from common import settings as cfg
from stage2_letters import letters as L

CLEARANCE = 0.03
RING_POINTS = 72
REACHED = 0.004
MAX_STEPS = 150


def _point_box_distance(p, box):
    inside, best = True, np.inf
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]
        edge = b - a
        if edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) < 0:
            inside = False
        t = np.clip(np.dot(p - a, edge) / np.dot(edge, edge), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * edge))))
    return 0.0 if inside else best


def letter_clearance(p, name, pose):
    return min(_point_box_distance(p, box) for box in L.polygons(name, *pose))


def descent_point(active, poses, tcp_xy):
    """Closest point to the tool on a ring CLEARANCE from the active letter and from all other letters."""
    x, y, _ = poses[active]
    best, best_distance = None, np.inf
    for angle in np.linspace(-np.pi, np.pi, RING_POINTS, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        for radius in np.arange(0.02, 0.12, 0.002):
            p = np.array([x, y]) + radius * direction
            if letter_clearance(p, active, poses[active]) >= CLEARANCE:
                break
        if not (np.all(p > np.array(cfg.TCP_LOW) + 0.005) and np.all(p < np.array(cfg.TCP_HIGH) - 0.005)):
            continue
        if any(letter_clearance(p, n, pose) < CLEARANCE for n, pose in poses.items() if n != active):
            continue
        d = float(np.linalg.norm(p - tcp_xy))
        if d < best_distance:
            best, best_distance = p, d
    return best


def action_towards(target_xyz, controller_target):
    error = np.asarray(target_xyz) - controller_target
    return np.clip(error / [cfg.ACTION_SCALE, cfg.ACTION_SCALE, cfg.ACTION_SCALE_Z], -1, 1)


def safe_descent(sim, active):
    """Returns a failure reason ("" if none). Does nothing when no free point exists."""
    poses = {n: (*sim.pose(n)[0], sim.pose(n)[1]) for n in L.NAMES}
    tcp = sim.tcp
    point = descent_point(active, poses, tcp[:2])
    if point is None:
        return ""
    for target in ([*point, tcp[2]], [*point, cfg.TCP_HEIGHT]):
        for _ in range(MAX_STEPS):
            _, reason = sim.act(action_towards(target, sim.controller.target))
            if reason:
                return reason
            if np.linalg.norm(sim.tcp - np.array(target)) < REACHED:
                break
    return ""
