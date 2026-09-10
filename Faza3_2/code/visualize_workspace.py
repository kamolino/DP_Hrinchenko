"""Visualize recorded workspace evaluations, without rerunning a policy.

Accepts a JSON object with summary and episodes, or an evaluate_precision_dqn
output directory for comparison. Each episode has task, success, distance,
collision, steps, positions and optionally qpos. Recorded qpos is used only for
replay; it never changes an evaluation result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


def load_results(path):
    path = Path(path)
    if path.is_dir():
        paths = json.loads((path / "trajectories.json").read_text())
        with (path / "episodes.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(paths) != len(rows):
            raise ValueError("Episode rows and recorded trajectories do not match")
        episodes = []
        for row, trajectory in zip(rows, paths):
            episodes.append({**trajectory, **row,
                             "success": str(row["success"]).lower() in ("true", "1"),
                             "collision": str(row["collision"]).lower() in ("true", "1"),
                             "distance": float(row["distance"]), "steps": int(row["steps"])})
        return {"summary": json.loads((path / "summary.json").read_text()), "episodes": episodes}
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or not result.get("episodes"):
        raise ValueError("Expected a result object with a nonempty episodes list")
    for episode in result["episodes"]:
        goal = np.asarray(episode["task"]["goal"], dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Every goal must contain three finite coordinates")
        if not np.isfinite(float(episode["distance"])):
            raise ValueError("Every episode must report a finite final distance")
    return result


def episode_group(episode):
    return str(episode.get("group", episode["task"].get("group", "workspace")))


def representative_indices(episodes, limit=24):
    """Take failures and successes across groups by fixed, reproducible rules."""
    chosen = set()
    groups = sorted({episode_group(e) for e in episodes})
    failures = [i for i, e in enumerate(episodes) if not e["success"]]
    if failures:
        chosen.add(max(failures, key=lambda i: float(episodes[i]["distance"])))
    for group in groups:
        members = [i for i, e in enumerate(episodes) if episode_group(e) == group]
        for success in (False, True):
            bucket = sorted([i for i in members if bool(episodes[i]["success"]) == success],
                            key=lambda i: float(episodes[i]["distance"]))
            if bucket:
                chosen.add(bucket[len(bucket) // 2])
                chosen.add(bucket[-1])
    for bucket in (failures, list(range(len(episodes)))):
        remaining = max(0, limit - len(chosen))
        if bucket and remaining:
            chosen.update(bucket[int(i)] for i in np.linspace(0, len(bucket)-1, min(remaining, len(bucket))))
    return sorted(chosen)[:limit]


def coverage_plot(result, output, tolerance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    episodes = result["episodes"]
    goals = np.asarray([e["task"]["goal"] for e in episodes])
    good = np.asarray([bool(e["success"]) for e in episodes])
    error = np.asarray([float(e["distance"]) * 1000 for e in episodes])
    collision = np.asarray([bool(e.get("collision", False)) for e in episodes])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    colors = {True: "#176d67", False: "#cb492f"}
    for ax, yi in ((axes[0, 0], 1), (axes[0, 1], 2)):
        for state, marker, label in ((True, "o", "Success: stopped and held"), (False, "x", "Failure")):
            selected = good == state
            if selected.any():
                ax.scatter(goals[selected, 0], goals[selected, yi], c=colors[state], marker=marker,
                           s=17 if state else 40, alpha=.6 if state else .95, label=label)
        if collision.any():
            ax.scatter(goals[collision, 0], goals[collision, yi], facecolors="none", edgecolors="#3e315f",
                       marker="s", s=70, linewidths=1.2, label="Collision")
        ax.scatter(0, 0, c="#333333", marker="s", s=30, label="Robot base")
        ax.set(xlabel="X (m)", ylabel=f"{'Y' if yi == 1 else 'Z'} (m)",
               title="Tested target coverage: " + ("top view" if yi == 1 else "side view"))
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.16)
    axes[0, 0].legend(fontsize=9, loc="best")
    ax = axes[1, 0]
    angle = np.degrees(np.arctan2(goals[:, 1], goals[:, 0]))
    edges = np.linspace(-180, 180, 13)
    counts, _ = np.histogram(angle, edges)
    successes, _ = np.histogram(angle[good], edges)
    values = np.divide(successes, counts, out=np.full(12, np.nan), where=counts > 0) * 100
    ax.bar((edges[:-1] + edges[1:]) / 2, values, width=25, color=colors[True])
    for x, value, n in zip((edges[:-1] + edges[1:]) / 2, values, counts):
        if n:
            ax.text(x, value + 1, f"{int(n)}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(x, 3, "n=0", ha="center", fontsize=8, rotation=90)
    ax.set(xlabel="Target azimuth around base (degrees)", ylabel="Success rate (%)",
           title="Coverage by direction · labels show episode counts", ylim=(0, 110), xticks=np.arange(-180,181,60))
    ax.grid(axis="y", alpha=.16)
    ax = axes[1, 1]
    for group in sorted({episode_group(e) for e in episodes}):
        d = np.sort([max(float(e["distance"])*1000, 1e-5) for e in episodes if episode_group(e) == group])
        n_success = sum(e["success"] for e in episodes if episode_group(e) == group)
        ax.step(d, np.arange(1, len(d)+1)/len(d)*100, where="post", label=f"{group}: {n_success}/{len(d)}")
    ax.axvline(tolerance*1000, color="#333333", linestyle="--", linewidth=1, label=f"Tolerance {tolerance*1000:g} mm")
    ax.set(xlabel="Final TCP error (mm)", ylabel="Episodes at or below error (%)",
           xscale="log", ylim=(0, 102), title="Final errors · all episodes, including failures")
    ax.grid(alpha=.16)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"FR3 workspace evaluation · {int(good.sum())}/{len(good)} successes ({good.mean()*100:.1f}%)"
                 f" · {int(collision.sum())} collisions", fontsize=15)
    fig.savefig(output / "workspace_coverage.png", dpi=180)
    plt.close(fig)


def arm_frames(qpositions, frame_indices):
    import mujoco
    from fr3_common import load_model_and_data, resolve_indices
    model, data = load_model_and_data()
    ids = resolve_indices(model)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fr3_link{i}") for i in range(8)]
    result = []
    for i in frame_indices:
        data.qpos[ids.qpos_ids] = qpositions[i]
        mujoco.mj_forward(model, data)
        result.append(np.vstack([data.xpos[bodies], data.site_xpos[ids.ee_site_id]]).round(4).tolist())
    return result


def compact_payload(result, tolerance, max_paths=24, max_frames=70):
    episodes = result["episodes"]
    selected = set(representative_indices(episodes, max_paths))
    compact = []
    for i, ep in enumerate(episodes):
        entry = {"id": i, "g": np.asarray(ep["task"]["goal"]).round(5).tolist(),
                 "s": bool(ep["success"]), "e": round(float(ep["distance"])*1000, 4),
                 "c": bool(ep.get("collision", False)), "n": int(ep.get("steps", 0)),
                 "k": episode_group(ep)}
        if i in selected and ep.get("positions"):
            n = len(ep["positions"])
            indices = np.unique(np.linspace(0, n-1, min(n, max_frames)).astype(int))
            entry["p"] = np.asarray(ep["positions"])[indices].round(5).tolist()
            entry["t"] = indices.tolist()
            qpos = ep.get("qpos", ep.get("joint_positions"))
            if qpos is not None and len(qpos) == n:
                entry["a"] = arm_frames(qpos, indices)
        compact.append(entry)
    dt = result.get("control_dt", result.get("summary", {}).get("control_dt", .05))
    return {"episodes": compact, "tolerance": tolerance*1000, "dt": dt,
            "successes": sum(e["success"] for e in episodes),
            "collisions": sum(e.get("collision", False) for e in episodes)}


def write_interactive(result, destination, tolerance):
    template = Path(__file__).with_name("workspace_viewer_template.html").read_text()
    payload = compact_payload(result, tolerance)
    document = template.replace("__WORKSPACE_DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    if len(document.encode()) > 950_000:
        payload = compact_payload(result, tolerance, max_paths=16, max_frames=45)
        document = template.replace("__WORKSPACE_DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    if len(document.encode()) >= 1_000_000:
        raise ValueError("Interactive visualization exceeds 1 MB; reduce episode metadata")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    return len(document.encode())


def render_replay(result, output, episode_index=None, max_frames=160):
    """Render real MuJoCo meshes at recorded joint configurations as a GIF."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw
    from fr3_common import load_model_and_data, resolve_indices
    episodes = result["episodes"]
    available = [i for i, e in enumerate(episodes) if e.get("qpos") and e.get("positions")]
    if not available:
        raise ValueError("Robot replay requires recorded qpos and positions")
    if episode_index is None:
        # Longest successful path illustrates the full motion; not lowest error.
        successes = [i for i in available if episodes[i]["success"]]
        episode_index = max(successes or available, key=lambda i: int(episodes[i].get("steps", 0)))
    ep = episodes[episode_index]
    qpos = np.asarray(ep["qpos"])
    xyz = np.asarray(ep["positions"])
    if len(qpos) != len(xyz):
        raise ValueError("qpos and positions must have matching frames")
    model, data = load_model_and_data()
    ids = resolve_indices(model)
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    model.geom_size[target_geom, 0] = .012
    data.mocap_pos[model.body_mocapid[target_body]] = ep["task"]["goal"]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .5]
    camera.distance = 2.7
    camera.azimuth = 135
    camera.elevation = -23
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model, height=480, width=720)
    frames = []
    indices = np.unique(np.linspace(0, len(qpos)-1, min(len(qpos), max_frames)).astype(int))
    for i in indices:
        data.qpos[ids.qpos_ids] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        path_indices = np.unique(np.linspace(0, i, min(i+1, 80)).astype(int))
        for point in xyz[path_indices]:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                              [.003, .003, .003], point, np.eye(3).reshape(9), [.95, .55, .08, .8])
            scene.ngeom += 1
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        distance = np.linalg.norm(xyz[i]-ep["task"]["goal"])*1000
        draw.rectangle((0, 0, 720, 31), fill=(18, 28, 36))
        draw.text((12, 10), f"FR3 | recorded episode {episode_index} | TCP error {distance:.2f} mm | frame {i}/{len(qpos)-1}", fill=(245, 247, 249))
        frames.append(frame)
    renderer.close()
    destination = output / "robot_replay.gif"
    frames[0].save(destination, save_all=True, append_images=frames[1:] + [frames[-1]]*15,
                   duration=50, loop=1, optimize=False)
    frames[-1].save(output / "robot_replay_final.png")
    manifest = {"episode_index": episode_index, "selection": "longest successful recorded episode" if episode_index is None else "explicit or default selected episode",
                "success": bool(ep["success"]), "error_mm": float(ep["distance"])*1000,
                "source_steps": int(ep.get("steps", len(qpos)-1)), "rendered_frames": len(indices),
                "playback": "50 ms per downsampled frame; compressed motion, not physical timing",
                "target_marker_radius_m": .012, "note": "Marker enlarged for visibility; does not represent tolerance"}
    (output / "robot_replay.json").write_text(json.dumps(manifest, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inline", type=Path, help="Write an interactive visualization fragment to this absolute path")
    parser.add_argument("--tolerance", type=float, default=.001, help="Criterion in metres")
    parser.add_argument("--render", action="store_true", help="Render a MuJoCo GIF from recorded qpos")
    parser.add_argument("--episode", type=int, help="Episode index for rendered replay")
    args = parser.parse_args()
    result = load_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    coverage_plot(result, args.output, args.tolerance)
    if args.inline:
        size = write_interactive(result, args.inline, args.tolerance)
        print(f"Interactive visualization: {args.inline} ({size} bytes)")
    if args.render:
        print(f"MuJoCo replay: {render_replay(result, args.output, args.episode)}")
    print(f"Coverage figure: {args.output / 'workspace_coverage.png'}")


if __name__ == "__main__":
    main()
