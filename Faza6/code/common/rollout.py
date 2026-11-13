"""Parallel episodes on fixed tasks, driven by a teacher, a student or a DAgger mix of both."""
import multiprocessing as mp
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _worker(job):
    import torch
    torch.set_num_threads(1)
    from common import policy
    tasks, make_env, make_teacher, driver, student_path, label, beta, seed = job
    env = make_env()
    teacher = make_teacher() if make_teacher and (label or driver == "teacher" or beta > 0) else None
    student = policy.load(student_path) if driver == "student" else None
    rng = np.random.default_rng(seed)
    rows, observations, actions = [], [], []
    for task in tasks:
        obs, info = env.reset(options={"task": task})
        phases = {}
        while True:
            teacher_action, phase = teacher.act(env) if teacher else (None, "-")
            if label:
                observations.append(obs)
                actions.append(teacher_action)
            if driver == "teacher" or (beta > 0 and rng.random() < beta):
                action = teacher_action
            else:
                action, _ = student.predict(obs, deterministic=True)
            phases[phase] = phases.get(phase, 0) + 1
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        row = dict(id=task["id"], success=info["is_success"], reason=info["reason"], steps=info["steps"],
                   position_error_cm=round(info["position_error_cm"], 2),
                   angle_error_deg=round(info["angle_error_deg"], 1), phases=phases)
        if "letter" in task:
            row["letter"] = task["letter"]
        rows.append(row)
    env.close()
    return rows, observations, actions


def run(tasks, make_env, make_teacher=None, driver="student", student_path=None, label=False, beta=0.0,
        workers=12, seed=0):
    workers = max(1, min(workers, len(tasks)))
    jobs = [(tasks[i::workers], make_env, make_teacher, driver, str(student_path) if student_path else None,
             label, beta, seed + i) for i in range(workers)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    rows = sorted([r for res in results for r in res[0]], key=lambda r: r["id"])
    obs = np.array([o for res in results for o in res[1]], dtype=np.float32)
    act = np.array([a for res in results for a in res[2]], dtype=np.float32)
    return rows, obs, act


def summary(rows):
    failures = {}
    for r in rows:
        if not r["success"]:
            failures[r["reason"]] = failures.get(r["reason"], 0) + 1
    done = [r for r in rows if r["success"]]
    return dict(success=sum(r["success"] for r in rows), episodes=len(rows), failures=failures,
                median_cm=float(np.median([r["position_error_cm"] for r in rows])),
                median_deg=float(np.median([r["angle_error_deg"] for r in rows])),
                mean_steps_success=float(np.mean([r["steps"] for r in done])) if done else None)
