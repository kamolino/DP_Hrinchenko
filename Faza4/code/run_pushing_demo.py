import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from pushing_env import PushingEnv


def labelled_frame(env, trail, title, reason=""):
    frame = Image.fromarray(env.render())
    panel = Image.new("RGB", (1280, 720), "#132236")
    panel.paste(frame, (0, 0))
    draw = ImageDraw.Draw(panel)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
        small = ImageFont.truetype(font_path, 16)
    except OSError:
        font = small = ImageFont.load_default()
    draw.rectangle((0, 0, 959, 70), fill="#132236")
    draw.text((20, 12), title, font=font, fill="white")
    draw.text((20, 40), "Orange: cube   |   Green: goal (2 cm)   |   Blue: tool", font=small, fill="#d1deeb")
    draw.text((980, 25), "TOP VIEW", font=font, fill="white")
    bounds = (0.27, 0.68, -0.22, 0.22)
    def xy(point):
        return (980 + (point[0] - bounds[0]) / (bounds[1] - bounds[0]) * 280,
                385 - (point[1] - bounds[2]) / (bounds[3] - bounds[2]) * 300)
    draw.rectangle((980, 85, 1260, 385), fill="#283b51", outline="#8092a7")
    gx, gy = xy(env.goal)
    rx, ry = 0.02 / (bounds[1] - bounds[0]) * 280, 0.02 / (bounds[3] - bounds[2]) * 300
    draw.ellipse((gx-rx, gy-ry, gx+rx, gy+ry), fill="#237b50", outline="#5bffad", width=2)
    draw.text((gx-20, gy-32), "GOAL", font=small, fill="#9dffd0")
    if len(trail) > 1:
        draw.line([xy(p) for p in trail], fill="#ffb04d", width=3)
    ox, oy = xy(env.object_position)
    draw.rectangle((ox-6, oy-6, ox+6, oy+6), fill="#ff9029")
    tx, ty = xy(env.tcp)
    draw.ellipse((tx-5, ty-5, tx+5, ty+5), fill="#51b9ff")
    draw.text((980, 425), f"Error: {env.distance() * 100:.2f} cm", font=font, fill="white")
    draw.text((980, 462), f"Sim time: {env.data.time:.2f} s", font=font, fill="white")
    draw.text((980, 500), f"Hold: {env.hold_steps * 0.002:.2f} / 0.30 s", font=small, fill="white")
    draw.text((980, 540), reason or "RUNNING", font=font, fill="#9dffd0" if reason == "success" else "#ff9a9a" if reason else "#e3ebf4")
    draw.text((980, 578), f"TCP height: {env.tcp[2]*100:.1f} cm", font=small, fill="white")
    draw.text((980, 615), "Replay: 2x speed", font=small, fill="#aebed1")
    draw.text((980, 645), "Orange line: cube path", font=small, fill="#ffb04d")
    draw.text((980, 670), f"Goal XY: ({env.goal[0]:.3f}, {env.goal[1]:+.3f}) m", font=small, fill="#9dffd0")
    draw.text((980, 693), f"Start XY: ({trail[0][0]:.3f}, {trail[0][1]:+.3f}) m", font=small, fill="#aebed1")
    return panel


def save_replay(policy, output, stage="C", seed=10000, options=None, title="PPO validation"):
    env = PushingEnv(stage=stage, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed, options=options)
    frames = []
    trail = [env.object_position.copy()]
    frames.append(labelled_frame(env, trail, title))
    info = {}
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, timeout, info = env.step(action)
        trail.append(env.object_position.copy())
        if env.steps % 4 == 0 or done or timeout:
            frames.append(labelled_frame(env, trail, title, info["reason"]))
        if done or timeout:
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [100] * len(frames)
    durations[0] = 700
    durations[-1] = 1400
    temporary = output.with_suffix(".tmp.gif")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    temporary.replace(output)
    frames[-1].save(output.with_suffix(".png"))
    env.close()
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--stage", choices=["A", "B", "C"], default="C")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--mp4", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    policy = PPO.load(args.model, device="cpu")
    options = None
    if args.tasks:
        import json
        tasks = json.loads(args.tasks.read_text())
        matches = [task for task in tasks if task["id"] == args.task_id]
        if len(matches) != 1:
            parser.error("Task ID must occur exactly once in the task file.")
        options = matches[0]
        args.seed = options["seed"]
    if args.gif:
        print(save_replay(policy, args.gif, args.stage, args.seed, options,
                          title=f"PPO stage {args.stage} | task {args.task_id}"))
        if args.mp4:
            import imageio.v2 as imageio
            with Image.open(args.gif) as source, imageio.get_writer(args.mp4, fps=10) as writer:
                for index in range(source.n_frames):
                    source.seek(index)
                    repeats = max(1, round(source.info.get("duration", 100) / 100))
                    for _ in range(repeats):
                        writer.append_data(np.asarray(source.convert("RGB")))
        return
    if args.mp4:
        parser.error("--mp4 requires --gif")
    import mujoco.viewer
    env = PushingEnv(stage=args.stage)
    obs, _ = env.reset(seed=args.seed, options=options)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = env.camera.lookat
        viewer.cam.distance = env.camera.distance
        viewer.cam.azimuth = env.camera.azimuth
        viewer.cam.elevation = env.camera.elevation
        while viewer.is_running():
            start = time.monotonic()
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, timeout, info = env.step(action)
            viewer.sync()
            time.sleep(max(0, 0.05 - (time.monotonic() - start)))
            if done or timeout:
                print(info)
                time.sleep(1)
                args.seed += 1
                obs, _ = env.reset(seed=args.seed, options=options)
    env.close()


if __name__ == "__main__":
    main()
