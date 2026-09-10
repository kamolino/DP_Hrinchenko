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
    """Replay measured joint positions with visible goal, TCP and orthogonal views."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from fr3_common import load_model_and_data, resolve_indices

    episodes = result["episodes"]
    available = [i for i, e in enumerate(episodes) if e.get("qpos") and e.get("positions")]
    if not available:
        raise ValueError("Robot replay requires recorded qpos and positions")
    automatic_selection = episode_index is None
    if automatic_selection:
        successes = [i for i in available if episodes[i]["success"]]
        episode_index = max(successes or available, key=lambda i: int(episodes[i].get("steps", 0)))
    ep = episodes[episode_index]
    qpos = np.asarray(ep["qpos"])
    xyz = np.asarray(ep["positions"])
    goal = np.asarray(ep["task"]["goal"])
    if len(qpos) != len(xyz):
        raise ValueError("qpos and positions must have matching frames")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, data = load_model_and_data()
    ids = resolve_indices(model)
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    model.geom_size[target_geom, 0] = .018
    model.geom_rgba[target_geom] = [.1, .95, .45, 1.]
    data.mocap_pos[model.body_mocapid[target_body]] = goal
    camera = mujoco.MjvCamera()
    bounds = np.vstack((xyz, goal, [0, 0, .35]))
    camera.lookat[:] = (bounds.min(0) + bounds.max(0)) / 2
    camera.distance = max(1.65, float(np.ptp(bounds, axis=0).max()) * 2.8)
    camera.azimuth = float(np.degrees(np.arctan2(goal[1], goal[0]))) + 45
    camera.elevation = -24
    width, height, view_width = 1120, 720, 760
    model.vis.global_.offwidth = view_width
    model.vis.global_.offheight = 580
    renderer = mujoco.Renderer(model, height=580, width=view_width)
    try:
        regular = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
        title = ImageFont.truetype("DejaVuSans-Bold.ttf", 23)
        strong = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        regular = small = title = strong = ImageFont.load_default()
    green, blue, orange = "#43ec91", "#41baff", "#ffbf65"
    ink, muted = "#f1f5f9", "#becbd7"
    control_dt = float(result.get("summary", {}).get("control_dt", .05))
    indices = np.unique(np.linspace(0, len(qpos)-1, min(len(qpos), max_frames)).astype(int))
    frames = []

    def project(point, scene):
        # Average the two eye cameras used by the monoscopic MuJoCo renderer.
        cameras = scene.camera
        pos = (cameras[0].pos + cameras[1].pos) / 2
        forward = (cameras[0].forward + cameras[1].forward) / 2
        up = (cameras[0].up + cameras[1].up) / 2
        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        delta = point - pos
        depth = float(delta @ forward)
        if depth <= 0:
            return None
        c = cameras[0]
        scale = 580 * float(c.frustum_near) / (float(c.frustum_top-c.frustum_bottom)*depth)
        return (view_width/2 + float(delta @ right)*scale,
                84 + 290 - float(delta @ up)*scale)

    def ring(draw, point, color, radius=9):
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="#122333", width=5)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)

    def plan_view(draw, rect, axes, heading, i):
        left, top, right, bottom = rect
        draw.rounded_rectangle(rect, radius=12, fill="#192c3e", outline="#344b5f")
        draw.text((left+15, top+10), heading, font=strong, fill=ink)
        points = np.vstack((xyz[:, axes], goal[list(axes)]))
        low, high = points.min(0), points.max(0)
        span = np.maximum(high-low, .1)
        scale = min((right-left-60)/span[0], (bottom-top-75)/span[1])
        center = (low+high)/2
        def xy(p):
            p = np.asarray(p)[list(axes)]
            return ((left+right)/2+(p[0]-center[0])*scale,
                    (top+bottom+18)/2-(p[1]-center[1])*scale)
        # Grey line is the complete recorded trajectory, not a planned path.
        draw.line([xy(p) for p in xyz], fill="#637586", width=2)
        if i:
            draw.line([xy(p) for p in xyz[:i+1]], fill=blue, width=4)
        sx, sy = xy(xyz[0])
        draw.rectangle((sx-4, sy-4, sx+4, sy+4), fill=orange)
        ring(draw, xy(goal), green, 9)
        cx, cy = xy(xyz[i])
        draw.ellipse((cx-4, cy-4, cx+4, cy+4), fill=blue)
        draw.text((left+15, bottom-25), f"{'XYZ'[axes[0]]} / {'XYZ'[axes[1]]} [m]", font=small, fill=muted)

    try:
        for i in indices:
            data.qpos[ids.qpos_ids] = qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene
            path_indices = np.unique(np.linspace(0, i, min(i+1, 110)).astype(int))
            for a, b in zip(path_indices[:-1], path_indices[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                  np.zeros(3), np.zeros(3), np.eye(3).reshape(9), [.15, .7, 1., 1.])
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0035, xyz[a], xyz[b])
                scene.ngeom += 1
            frame = Image.new("RGB", (width, height), "#101e2b")
            frame.paste(Image.fromarray(renderer.render()), (0, 84))
            draw = ImageDraw.Draw(frame)
            distance = float(np.linalg.norm(xyz[i]-goal)*1000)
            finished = i == len(qpos)-1
            status = "Cieľ dosiahnutý" if finished and ep["success"] else "Úloha neúspešná" if finished else "Pohyb k cieľu"
            draw.text((22, 13), "FR3 – pohyb k zadanému cieľu", font=title, fill=ink)
            draw.text((22, 49), f"Úloha {episode_index+1} / {len(episodes)}  |  Čas simulácie: {i*control_dt:.2f} s  |  {status}", font=regular, fill=green if finished and ep['success'] else muted)
            # Overlay markers remain visible when the target is hidden by the arm.
            gp, tp = project(goal, scene), project(xyz[i], scene)
            for point, label, color, box in [(gp, "CIEĽ – pevný bod", green, (490, 106, 735, 143)),
                                              (tp, "TCP – koniec ramena", blue, (22, 106, 275, 143))]:
                if point is not None:
                    anchor = ((box[0]+box[2])/2, box[3])
                    draw.line((anchor, point), fill="#15293a", width=5)
                    draw.line((anchor, point), fill=color, width=2)
                    ring(draw, point, color)
                draw.rounded_rectangle(box, radius=7, fill="#102331", outline=color, width=2)
                draw.text((box[0]+12, box[1]+8), label, font=strong, fill=color)
            draw.text((782, 101), "Vzdialenosť TCP od cieľa", font=regular, fill=muted)
            draw.text((782, 128), f"{distance:.2f} mm", font=title, fill=green if distance <= 1 else ink)
            draw.text((782, 163), "Podmienka: ≤ 1 mm, ≤ 5 mm/s", font=small, fill=muted)
            draw.text((782, 183), "nepretržite počas 0,3 s", font=small, fill=muted)
            plan_view(draw, (780, 215, 1100, 412), (0, 1), "Pohľad zhora · X–Y", i)
            plan_view(draw, (780, 425, 1100, 622), (0, 2), "Pohľad zboku · X–Z", i)
            draw.text((782, 635), "Zelená: cieľ   Modrá: prejdená dráha", font=small, fill=muted)
            draw.text((782, 654), "Oranžová: štart   Sivá: celý záznam", font=small, fill=muted)
            draw.rectangle((0, 675, width, height), fill="#101e2b")
            draw.text((22, 688), "Zrýchlený záznam zo simulácie. Značky sú zväčšené; ich veľkosť nepredstavuje toleranciu 1 mm.", font=small, fill=muted)
            frames.append(frame)
    finally:
        renderer.close()
    destination = output / "robot_replay.gif"
    durations = [70]*len(frames)
    durations[0] = 1200
    durations[-1] = 2500
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False)
    frames[0].save(output / "robot_replay_start.png")
    frames[len(frames)//2].save(output / "robot_replay_middle.png")
    frames[-1].save(output / "robot_replay_final.png")
    manifest = {"episode_index": int(episode_index), "episode_number": int(episode_index)+1,
                "selection": "longest successful recorded episode" if automatic_selection else "explicit selected episode",
                "success": bool(ep["success"]), "error_mm": float(ep["distance"])*1000,
                "source_steps": int(ep.get("steps", len(qpos)-1)), "control_dt": control_dt,
                "rendered_frames": len(indices), "frame_durations_ms": durations,
                "playback": "Downsampled replay with start/end pauses; not physical timing",
                "target_marker_radius_m": .018,
                "overlays": "Projected goal and TCP markers, measured TCP error, XY/XZ recorded paths",
                "note": "Markers enlarged for visibility; grey path is the full measured trajectory, not the planner output"}
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
