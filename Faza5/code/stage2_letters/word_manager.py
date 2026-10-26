"""Task manager that assembles the word ARM: letters A -> R -> M, then repair passes.

For every letter: scripted move to the start point, scripted descent next to the letter, the straight
path to its slot cut into pieces <= 12 cm, and the letter policy pushes along the pieces.

    python -m stage2_letters.word_manager --scenes 100 --seed 542000
    python -m stage2_letters.word_manager --shared ../models/stage3_shared.zip --scenes 300 --seed 992000
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from common import settings as cfg
from common.geometry import wrap
from stage1_cube.cube_env import observation
from stage2_letters import conditions, letters as L
from stage2_letters.approach import safe_descent
from stage2_letters.word_sim import ORDER, WordSim

os.environ.setdefault("OMP_NUM_THREADS", "1")
DEV_SEED, TEST_SEED = 542000, 992000
LETTER_POLICIES = {n: cfg.MODELS / f"stage2_letter_{n}.zip" for n in ORDER}
PHASE_STEPS = 1000
TOTAL_STEPS = 9000
BUDGET = {"segment": 500, "final": 600}
SEGMENT_LENGTH = 0.12
SOFT_DISTANCE, SOFT_ANGLE = 0.03, np.deg2rad(15)
DISTURB_DISTANCE, DISTURB_ANGLE = 0.01, np.deg2rad(5)
EDGE = 0.06


def make_scenes(count, seed):
    rng = np.random.default_rng(seed)
    return [dict(id=i, starts={k: [float(v) for v in p] for k, p in L.sample_starts(rng, edge=EDGE).items()})
            for i in range(count)]


def segment_route(sim, name):
    gx, gy, gyaw = L.GOALS[name]
    xy = sim.pose(name)[0]
    delta = np.array([gx, gy]) - xy
    pieces = max(1, int(np.ceil(np.linalg.norm(delta) / SEGMENT_LENGTH)))
    route = [("segment", (*(xy + delta * k / pieces), gyaw)) for k in range(1, pieces)]
    return route + [("final", (gx, gy, gyaw))]


def at_goal(sim, name):
    d, a = sim.errors(name)
    return d < cfg.SUCCESS_DISTANCE and a < cfg.SUCCESS_ANGLE


class Manager:
    def __init__(self, sim, policies, repair=4, record=False, observe=observation):
        self.sim, self.policies, self.repair, self.record, self.observe = sim, policies, repair, record, observe
        self.recorded = []

    def run_phase(self, name, placed):
        sim = self.sim
        sim.activate(name)
        sim.contact_letter = ""
        reason = sim.move_to_start()
        mode = "move" if reason else "policy"
        if not reason:
            reason = safe_descent(sim, name)
            sim.applied_xy[:] = 0
            sim.hold_steps = 0
            mode = "descent" if reason else "policy"
        sim.hold_steps, steps, success = 0, 0, False
        start_cm = float(100 * np.linalg.norm(sim.pose(name)[0] - np.array(L.GOALS[name][:2])))
        route = [("final", tuple(L.GOALS[name]))] if reason else segment_route(sim, name)
        for kind, (x, y, yaw) in route:
            if reason:
                break
            sim.goal, sim.goal_yaw, sim.hold_steps = np.array([x, y]), float(yaw), 0
            if self.record:
                self.recorded.append(dict(letter=name, kind=kind, goal=[float(x), float(y), float(yaw)],
                                          qpos=sim.data.qpos.tolist(), target=sim.controller.target.tolist(),
                                          placed=sorted(placed)))
            budget = BUDGET[kind] if len(route) > 1 else PHASE_STEPS
            for _ in range(budget):
                if sim.steps >= TOTAL_STEPS:
                    break
                action, _ = self.policies[name].predict(self.observe(sim), deterministic=True)
                success, reason = sim.act(action)
                steps += 1
                if reason or success:
                    break
                if kind != "final" and sim.distance() < SOFT_DISTANCE and sim.angle_error() < SOFT_ANGLE:
                    break
            if kind != "final":
                success = False
        sim.activate(name)
        d, a = sim.errors(name)
        return dict(success=bool(success), steps=steps, cm=round(100 * d, 2), deg=round(float(np.rad2deg(a)), 1),
                    start_cm=round(start_cm, 1), route=[k for k, _ in route], mode=mode), reason

    def run_scene(self, scene):
        sim = self.sim
        sim.reset(scene["starts"])
        self.recorded = []
        out = dict(id=scene["id"], phases={}, repairs=[], failure="", disturbed=[])
        placed, reason = {}, ""
        for repair in range(self.repair + 1):
            todo = sim.order if repair == 0 else [n for n in sim.order if not at_goal(sim, n)]
            if repair and not todo:
                break
            for name in todo:
                if sim.steps >= TOTAL_STEPS:
                    out["total_limit"] = True
                    break
                phase, reason = self.run_phase(name, placed)
                if repair == 0:
                    out["phases"][name] = phase
                else:
                    out["repairs"].append(dict(pass_=repair, letter=name, **phase))
                if phase["success"]:
                    placed[name] = sim.pose(name)
                for other, (xy, yaw) in list(placed.items()):
                    if other == name:
                        continue
                    now_xy, now_yaw = sim.pose(other)
                    if np.linalg.norm(now_xy - xy) > DISTURB_DISTANCE or abs(wrap(now_yaw - yaw)) > DISTURB_ANGLE:
                        out["disturbed"].append(other)
                        del placed[other]
                if reason:
                    out["failure"] = f"{name}: {reason}"
                    out["failure_mode"] = phase["mode"]
                    out["contact_letter"] = sim.contact_letter
                    break
            if reason or out.get("total_limit"):
                break
        final = {}
        for name in ORDER:
            d, a = sim.errors(name)
            final[name] = dict(cm=round(100 * d, 2), deg=round(float(np.rad2deg(a)), 1), ok=bool(at_goal(sim, name)))
        out.update(final=final, word_success=all(f["ok"] for f in final.values()), steps=sim.steps)
        if self.record:
            out["states"] = self.recorded
        return out


def load_policies(letter_paths=None, shared=None):
    from common import policy
    if shared:
        from stage3_shared.shared_env import OneHotPolicy
        model = policy.load(shared)
        return {n: OneHotPolicy(model, n) for n in ORDER}
    paths = letter_paths or LETTER_POLICIES
    return {n: policy.load(paths[n]) for n in ORDER}


def _worker(job):
    import torch
    torch.set_num_threads(1)
    scenes, letter_paths, shared, repair, record, condition = job
    setup = conditions.apply(condition)
    sim = WordSim()
    sim.order = setup.order
    manager = Manager(setup.wrap_sim(sim), load_policies(letter_paths, shared), repair, record, setup.observe)
    return [manager.run_scene(s) for s in scenes]


def evaluate(scenes, letter_paths=None, shared=None, workers=12, repair=4, record=False, condition="nominal"):
    paths = {k: str(v) for k, v in (letter_paths or LETTER_POLICIES).items()}
    chunks = [c for c in (scenes[i::workers] for i in range(workers)) if c]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        results = pool.map(_worker, [(c, paths, str(shared) if shared else None, repair, record, condition)
                                     for c in chunks])
    return sorted([r for res in results for r in res], key=lambda r: r["id"])


def summarize(rows):
    s = dict(scenes=len(rows), word_success=sum(r["word_success"] for r in rows))
    for name in ORDER:
        s[f"{name}_final_ok"] = sum(r["final"][name]["ok"] for r in rows)
        s[f"{name}_median_cm_deg"] = (float(np.median([r["final"][name]["cm"] for r in rows])),
                                      float(np.median([r["final"][name]["deg"] for r in rows])))
    s["disturbances"] = sum(len(r["disturbed"]) for r in rows)
    s["scenes_with_disturbance"] = sum(bool(r["disturbed"]) for r in rows)
    s["failures"] = {}
    for r in rows:
        if r["failure"]:
            key = r["failure"].split(": ")[1]
            s["failures"][key] = s["failures"].get(key, 0) + 1
    s["mean_steps"] = float(np.mean([r["steps"] for r in rows]))
    s["repair_phases"] = sum(len(r["repairs"]) for r in rows)
    s["total_limit"] = sum(bool(r.get("total_limit")) for r in rows)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--repair", type=int, default=4)
    parser.add_argument("--shared", help="one shared policy (stage 3) instead of the three letter policies")
    for n in ORDER:
        parser.add_argument(f"--policy-{n}", default=str(LETTER_POLICIES[n]))
    parser.add_argument("--condition", default="nominal", choices=conditions.NAMES)
    parser.add_argument("--record", action="store_true", help="save the state at the start of every route piece")
    parser.add_argument("--out", default=str(cfg.ROOT / "results" / "word"))
    args = parser.parse_args()
    conditions.apply(args.condition)
    scenes = make_scenes(args.scenes, args.seed)
    letter_paths = {n: getattr(args, f"policy_{n}") for n in ORDER}
    rows = evaluate(scenes, letter_paths, args.shared, args.workers, args.repair, args.record, args.condition)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.record:
        states = [dict(scene=r["id"], **st) for r in rows for st in r.pop("states", [])]
        (out / "states.json").write_text(json.dumps(states))
    s = summarize(rows)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(dict(summary=s, seed=args.seed, repair=args.repair,
                                                      condition=args.condition, shared=args.shared), indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
