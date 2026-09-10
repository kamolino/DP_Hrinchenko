"""Deterministic held-out, stratified FR3 workspace tasks.

Goals are forward-kinematics endpoints from valid joint configurations. Their
``witness_q`` certifies endpoint feasibility and MUST NOT be read by a controller.
No claim about every point of the continuous workspace or path connectivity is
made. Stratification counters the Cartesian bias of uniform joint sampling.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from precision_environment import PrecisionConfig, PrecisionEnv


AZIMUTH_BINS = 12
RADIUS_EDGES = np.array([0., .2, .4, .6, .8, 1.])
HEIGHT_EDGES = np.array([.03, .25, .45, .65, .85, 1.05, 1.3])
GROUPS = ('global_home', 'boundary_home', 'random_start')


def valid_workspace_pose(env, q, min_tcp_height=.03):
    """FK validation with floor clearance, joint bounds and penetrating contacts.

    Uses the environment's private scratch data, never changes the live robot.
    The 3 cm clearance is an operational restriction, not a physical arm limit.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        return None
    if np.any(q < env.low) or np.any(q > env.high):
        return None
    mujoco.mj_resetDataKeyframe(env.model, env.scratch, env.ids.home_key_id)
    env.scratch.qpos[env.ids.qpos_ids] = q
    mujoco.mj_forward(env.model, env.scratch)
    p = env.scratch.site_xpos[env.ids.ee_site_id].copy()
    return p if p[2] >= min_tcp_height and not env.collision(env.scratch) else None


class WorkspaceSamplingEnv(PrecisionEnv):
    """Same physical model with the documented global-workspace clearance."""
    def __init__(self, config=None, seed=0, min_tcp_height=.03):
        self.min_tcp_height = min_tcp_height
        super().__init__(config, seed)

    def valid_pose(self, q):
        return valid_workspace_pose(self, q, self.min_tcp_height)


def point_bins(points):
    p = np.atleast_2d(np.asarray(points, dtype=float))
    angles = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    radii = np.linalg.norm(p[:, :2], axis=1)
    azimuth = np.floor(((angles + 180.) % 360.) / (360 / AZIMUTH_BINS)).astype(int)
    radial = np.clip(np.searchsorted(RADIUS_EDGES, radii, side='right') - 1, 0, len(RADIUS_EDGES)-2)
    height = np.clip(np.searchsorted(HEIGHT_EDGES, p[:, 2], side='right') - 1, 0, len(HEIGHT_EDGES)-2)
    return np.column_stack((azimuth, radial, height))


def _candidate_pool(env, rng, pool_size):
    qs, ps = [], []
    # 70% full joint-range samples, 30% samples biased toward one joint limit.
    # The full-range component reaches intermediate folds as well as extremes.
    for i in range(pool_size):
        unit = rng.random(7)
        if i % 10 >= 7:
            joint = int(rng.integers(7))
            edge_distance = rng.uniform(0., .025)
            unit[joint] = edge_distance if rng.integers(2) == 0 else 1.-edge_distance
        q = env.low + unit * (env.high-env.low)
        p = env.valid_pose(q)
        if p is not None:
            qs.append(q); ps.append(p)
    if not qs:
        raise RuntimeError('No valid workspace endpoints in candidate pool')
    return np.asarray(qs), np.asarray(ps)


def _choose_stratified(indices, bins, used_cells, radial_use, height_use, rng):
    cells = {tuple(bins[i]) for i in indices}
    best_count = min(used_cells[c] for c in cells)
    cells = [c for c in sorted(cells) if used_cells[c] == best_count]
    scores = [radial_use[c[1]] + height_use[c[2]] for c in cells]
    best = min(scores)
    cells = [c for c, s in zip(cells, scores) if s == best]
    cell = cells[int(rng.integers(len(cells)))]
    candidates = indices[np.all(bins[indices] == cell, axis=1)]
    return int(rng.choice(candidates))


def _coverage(tasks, qs, ps, pool_bins, env, seed, pool_size):
    goals = np.asarray([t['goal'] for t in tasks])
    goal_bins = point_bins(goals)
    goal_counts = Counter(map(tuple, goal_bins.tolist()))
    pool_counts = Counter(map(tuple, pool_bins.tolist()))
    starts = np.asarray([t['start_q'] for t in tasks])
    witnesses = np.asarray([t['witness_q'] for t in tasks])
    def xyz_bounds(p):
        return dict(min_m=p.min(axis=0).tolist(), max_m=p.max(axis=0).tolist(),
                    radius_min_m=float(np.linalg.norm(p[:,:2], axis=1).min()),
                    radius_max_m=float(np.linalg.norm(p[:,:2], axis=1).max()))
    return {
        'schema_version': 1, 'seed': seed, 'tasks': len(tasks),
        'sampling_attempts': pool_size, 'valid_candidate_endpoints': len(ps),
        'config': asdict(env.config), 'tcp_floor_clearance_m': env.min_tcp_height,
        'joint_low_rad': env.low.tolist(), 'joint_high_rad': env.high.tolist(),
        'radius_edges_m': RADIUS_EDGES.tolist(), 'height_edges_m': HEIGHT_EDGES.tolist(),
        'azimuth_edges_deg': np.linspace(-180, 180, AZIMUTH_BINS+1).tolist(),
        'candidate_bounds': xyz_bounds(ps), 'goal_bounds': xyz_bounds(goals),
        'groups': dict(Counter(t['group'] for t in tasks)),
        'goal_azimuth_counts': np.bincount(goal_bins[:,0], minlength=AZIMUTH_BINS).tolist(),
        'goal_radius_counts': np.bincount(goal_bins[:,1], minlength=len(RADIUS_EDGES)-1).tolist(),
        'goal_height_counts': np.bincount(goal_bins[:,2], minlength=len(HEIGHT_EDGES)-1).tolist(),
        'occupied_candidate_cells': len(pool_counts), 'occupied_goal_cells': len(goal_counts),
        'cell_coverage': [{'azimuth': int(c[0]), 'radius': int(c[1]), 'height': int(c[2]),
                           'valid_pool_endpoints': pool_counts[c], 'tasks': goal_counts[c]}
                          for c in sorted(pool_counts)],
        'witness_joint_min_rad': witnesses.min(0).tolist(),
        'witness_joint_max_rad': witnesses.max(0).tolist(),
        'start_joint_min_rad': starts.min(0).tolist(),
        'start_joint_max_rad': starts.max(0).tolist(),
        'limitations': [
            'Finite sampled benchmark, not a proof of coverage of every reachable Cartesian point.',
            'Only endpoint feasibility is certified; no collision-free path is supplied or assumed.',
            'Position only: the task does not prescribe tool orientation, payload, or added obstacles.',
            '3 cm TCP floor clearance and a joint margin restrict the tested operational workspace.',
            'Candidate extrema approximate boundaries; unsampled cells are not certified unreachable.',
            'Task selection uses geometry only, never controller success or failure.',
            'All generated tasks must remain in evaluation, including planning failures and collisions.',
        ],
    }


def generate_suite(seed, count=216, config=None, pool_size=24000, min_tcp_height=.03):
    """Return ``(tasks, coverage)`` with equal groups and azimuth-sector counts.

    ``count`` is the total count, a positive multiple of 36. ``global_home``
    and ``random_start`` span occupied radial/height cells. ``boundary_home``
    targets near sampled outer/inner radii, upper/lower heights and joint limits.
    """
    if count < 36 or count % (len(GROUPS)*AZIMUTH_BINS):
        raise ValueError('count must be a positive multiple of 36')
    if pool_size < max(1000, 5*count):
        raise ValueError('pool_size must be at least max(1000, 5*count)')
    env = WorkspaceSamplingEnv(config, seed=seed, min_tcp_height=min_tcp_height)
    rng = np.random.default_rng(seed)
    qs, ps = _candidate_pool(env, rng, pool_size)
    bins = point_bins(ps)
    p_home = env.valid_pose(env.home)
    available = np.linalg.norm(ps-p_home, axis=1) >= .03
    used_cells, radial_use, height_use = Counter(), Counter(), Counter()
    start_cells, start_radial, start_height = Counter(), Counter(), Counter()
    tasks = []
    per_sector = count // (len(GROUPS)*AZIMUTH_BINS)
    kinds = ['outer_radius', 'inner_radius', 'lower_height', 'upper_height', 'joint_lower', 'joint_upper']
    for group in GROUPS:
        for slot in range(per_sector):
            for sector in range(AZIMUTH_BINS):
                candidates = np.flatnonzero(available & (bins[:,0] == sector))
                if not len(candidates):
                    raise RuntimeError(f'Candidate pool exhausted in azimuth sector {sector}; increase pool_size')
                boundary_kind = None
                if group == 'boundary_home':
                    boundary_kind = kinds[(slot + (sector % 3)*2) % len(kinds)]
                    radius = np.linalg.norm(ps[candidates,:2], axis=1)
                    if boundary_kind == 'outer_radius': score = -radius
                    elif boundary_kind == 'inner_radius': score = radius
                    elif boundary_kind == 'lower_height': score = ps[candidates,2]
                    elif boundary_kind == 'upper_height': score = -ps[candidates,2]
                    elif boundary_kind == 'joint_lower': score = qs[candidates, sector % 7]
                    else: score = -qs[candidates, sector % 7]
                    # A tiny boundary band preserves variety among near extrema.
                    candidates = candidates[np.argsort(score, kind='stable')[:8]]
                index = _choose_stratified(candidates, bins, used_cells, radial_use, height_use, rng)
                available[index] = False
                cell = tuple(bins[index])
                used_cells[cell] += 1; radial_use[cell[1]] += 1; height_use[cell[2]] += 1
                start = env.home.copy()
                if group == 'random_start':
                    start_candidates = np.flatnonzero(np.linalg.norm(ps-ps[index], axis=1) >= .03)
                    si = _choose_stratified(start_candidates, bins, start_cells, start_radial, start_height, rng)
                    start = qs[si].copy()
                    scell = tuple(bins[si])
                    start_cells[scell] += 1; start_radial[scell[1]] += 1; start_height[scell[2]] += 1
                task = {'id': f'{seed}-{len(tasks):04d}', 'start_q': start.tolist(),
                        'goal': ps[index].tolist(), 'witness_q': qs[index].tolist(), 'group': group,
                        'bin': {'azimuth': int(cell[0]), 'radius': int(cell[1]), 'height': int(cell[2]),
                                'azimuth_deg': float(np.degrees(np.arctan2(ps[index,1],ps[index,0]))),
                                'radius_m': float(np.linalg.norm(ps[index,:2]))}}
                if boundary_kind:
                    task['boundary_kind'] = boundary_kind
                    if boundary_kind.startswith('joint_'): task['boundary_joint'] = sector % 7
                tasks.append(task)
    return tasks, _coverage(tasks, qs, ps, bins, env, seed, pool_size)


def validate_suite(tasks, config=None, min_tcp_height=.03):
    """Validate every stored endpoint; raise rather than silently drop a task."""
    env = WorkspaceSamplingEnv(config, min_tcp_height=min_tcp_height)
    ids = set()
    for i, task in enumerate(tasks):
        if task.get('id', i) in ids:
            raise ValueError(f'Duplicate task id at task {i}')
        ids.add(task.get('id', i))
        start = env.valid_pose(np.asarray(task['start_q']))
        witness = env.valid_pose(np.asarray(task['witness_q']))
        goal = np.asarray(task['goal'], dtype=float)
        if start is None or witness is None or goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError(f'Invalid endpoint at task {i}')
        if not np.allclose(witness, goal, atol=1e-10, rtol=0):
            raise ValueError(f'Goal and FK witness disagree at task {i}')
    return {'validated_tasks': len(tasks), 'invalid_tasks': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'workspace')
    p.add_argument('--validation-count', type=int, default=72)
    p.add_argument('--test-count', type=int, default=216)
    p.add_argument('--validation-seed', type=int, default=731003)
    p.add_argument('--test-seed', type=int, default=982451)
    p.add_argument('--pool-size', type=int, default=24000)
    args = p.parse_args()
    if args.validation_seed == args.test_seed:
        raise ValueError('Validation and final test seeds must differ')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output/f'{name}_{suffix}.json' for name in ('validation','test') for suffix in ('tasks','coverage')]
    if any(path.exists() for path in paths):
        raise ValueError('Task suites already exist; use a fresh output directory')
    for name, seed, count in [('validation', args.validation_seed, args.validation_count),
                              ('test', args.test_seed, args.test_count)]:
        tasks, report = generate_suite(seed, count, pool_size=args.pool_size)
        report['endpoint_validation'] = validate_suite(tasks)
        (args.output/f'{name}_tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')
        (args.output/f'{name}_coverage.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'suite': name, 'tasks': count, 'goals': report['goal_bounds'],
                          'azimuth_counts': report['goal_azimuth_counts'],
                          'height_counts': report['goal_height_counts']}))


if __name__ == '__main__':
    main()
