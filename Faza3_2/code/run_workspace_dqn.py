import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from precision_environment import PrecisionConfig
from joint_waypoint_dqn import (
    WorkspaceEnv,
    load_joint,
    joint_observation,
    joint_commands,
    apply_joint_action,
)
from workspace_planner import WorkspacePlanner

# ============================================================
# 1. INTERMEDIATE JOINT POSITIONS
# ============================================================


def densify(path, max_delta=0.12):
    result = []
    for start_q, end_q in zip(path[:-1], path[1:]):
        step_count = max(1, int(np.ceil(np.max(np.abs(end_q - start_q)) / max_delta)))
        result.extend(
            start_q + (end_q - start_q) * step_index / step_count
            for step_index in range(1, step_count + 1)
        )
    return result


# ============================================================
# 2. RUNNING ONE EPISODE
# ============================================================


def run_episode(env, net, start_q, goal, seed=0, viewer=None):
    env.reset(dict(start_q=start_q, goal=goal))
    planner = WorkspacePlanner(env, seed)
    current_q = env.data.qpos[env.ids.qpos_ids].copy()
    path, planning = planner.plan(current_q, np.asarray(goal))
    positions = [env.ee().tolist()]
    joint_positions = [current_q.tolist()]
    info = dict(
        distance=env.distance,
        success=False,
        collision=False,
        steps=0,
        tcp_speed=0.0,
        terminated=False,
        truncated=False,
    )
    if path is None:
        return dict(
            **info,
            planning=planning,
            status=planning["status"],
            positions=positions,
            qpos=joint_positions,
            shield_interventions=0
        )
    waypoints = densify(path)
    waypoint_index = 0
    shield_interventions = 0
    status = "timeout"
    for _ in range(env.config.max_steps):
        step_start = time.monotonic()
        if viewer is not None and not viewer.is_running():
            status = "viewer_closed"
            break
        current_q = env.data.qpos[env.ids.qpos_ids].copy()
        while (
            waypoint_index < len(waypoints) - 1
            and np.max(np.abs(current_q - waypoints[waypoint_index])) < 0.012
        ):
            waypoint_index += 1
        target_q = waypoints[waypoint_index]
        state = joint_observation(env, target_q)
        with torch.no_grad():
            ranked_actions = (
                net(torch.from_numpy(state)).argsort(descending=True).numpy()
            )
        commands = joint_commands(env, target_q)
        action = None
        for candidate in ranked_actions:
            # Take the highest-ranked action that passes the collision check.
            if planner.edge(current_q, commands[candidate], resolution=0.015):
                action = int(candidate)
                break
        if action is None:
            status = "shield_blocked"
            break
        shield_interventions += int(action != ranked_actions[0])
        _, _, done, info = apply_joint_action(env, target_q, action)
        positions.append(env.ee().tolist())
        joint_positions.append(env.data.qpos[env.ids.qpos_ids].tolist())
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, env.dt - (time.monotonic() - step_start)))
        if done:
            status = (
                "success"
                if info["success"]
                else "collision" if info["collision"] else "timeout"
            )
            break
    return dict(
        **info,
        status=status,
        planning=planning,
        positions=positions,
        qpos=joint_positions,
        shield_interventions=shield_interventions,
        waypoint_count=len(waypoints),
        waypoints=[waypoint.tolist() for waypoint in waypoints]
    )


# ============================================================
# 3. RUNNING THE DEMO OR TEST
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/workspace/joint_run_001/double_dqn.pt",
    )
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--goal", type=float, nargs=3)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    network, metadata = load_joint(args.model)
    env = WorkspaceEnv(PrecisionConfig(max_steps=args.max_steps))
    if args.tasks:
        tasks = json.loads(args.tasks.read_text())
    elif args.goal:
        tasks = [dict(start_q=env.home.tolist(), goal=args.goal, group="custom")]
    else:
        raise ValueError("Provide --tasks or --goal X Y Z")
    if args.output and args.output.exists():
        raise ValueError("Use a fresh results path")
    from contextlib import nullcontext

    if args.viewer:
        import mujoco.viewer
    context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if args.viewer
        else nullcontext(None)
    )
    episodes = []
    with context as viewer:
        for episode_index, task in enumerate(tasks):
            episode_result = run_episode(
                env,
                network,
                task["start_q"],
                task["goal"],
                seed=8100 + episode_index,
                viewer=viewer,
            )
            episodes.append(dict(task=task, **episode_result))
            print(
                episode_index,
                task["group"],
                episode_result["status"],
                round(episode_result["distance"] * 1000, 3),
                "mm",
                episode_result["steps"],
                "steps",
                flush=True,
            )
            if episode_result["status"] == "viewer_closed":
                break
    distances = np.array([episode_result["distance"] for episode_result in episodes])
    successes = sum(episode_result["success"] for episode_result in episodes)
    summary = dict(
        controller="Global IK + RRT-Connect + joint-action Double DQN + collision shield",
        episodes=len(episodes),
        success_count=successes,
        success_rate=successes / len(episodes),
        mean_error_mm=float(distances.mean() * 1000),
        p95_error_mm=float(np.percentile(distances, 95) * 1000),
        collisions=sum(episode_result["collision"] for episode_result in episodes),
        tolerance_mm=1.0,
        hold_seconds=0.3,
        max_tcp_speed_mm_s=5.0,
        control_dt=env.dt,
        max_steps=args.max_steps,
        model=str(args.model.resolve()),
        planning_failures=sum(
            episode_result["status"] in ("ik_failed", "rrt_failed", "invalid_start")
            for episode_result in episodes
        ),
    )
    result = dict(summary=summary, episodes=episodes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
