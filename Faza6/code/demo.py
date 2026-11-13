"""Watch the final models in the MuJoCo viewer, or record a labelled replay (GIF / MP4).

    python demo.py word                         # viewer, three letter policies
    python demo.py word-shared --gif ../results/word_shared.gif --index 0
    python demo.py cube --gif ../results/cube.gif --index 3
modes: cube (one policy), cube-manager (rotation + pushing), word (stage 2), word-shared (stage 3)
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import policy, settings as cfg
from common.geometry import block_corners
from stage1_cube.cube_env import CubeEnv
from stage1_cube.evaluate import POLICY as CUBE_POLICY
from stage1_cube.manager import Manager as CubeManager
from stage1_cube.tasks import TEST_SEED as CUBE_SEED, make_tasks as cube_tasks
from stage2_letters import letters as L, word_manager
from stage2_letters.word_sim import WordSim

WIDTH, HEIGHT, PANEL = 960, 720, 320
BOUNDS = (0.22, 0.78, -0.30, 0.30)
COLORS = {"A": "#ff8a3d", "R": "#3d9bff", "M": "#c46be0", "block": "#ff9029"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def fonts():
    try:
        return ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


class Recorder:
    def __init__(self, sim, title, every=2):
        self.sim, self.title, self.every = sim, title, every
        self.renderer = mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.50, 0.0, 0.18]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.95, 180, -60
        self.frames, self.trail, self.phase = [], [], ""
        self.big, self.small = fonts()

    def to_panel(self, x, y):
        x0, x1, y0, y1 = BOUNDS
        return WIDTH + 20 + (y1 - y) / (y1 - y0) * (PANEL - 40), 110 + (x1 - x) / (x1 - x0) * (PANEL - 40) * 1.1

    def shapes(self):
        sim = self.sim
        if hasattr(sim, "pose"):
            current = [(n, poly) for n in L.NAMES for poly in L.polygons(n, *sim.pose(n)[0], sim.pose(n)[1])]
            goals = [poly for n in L.NAMES for poly in L.polygons(n, *L.GOALS[n])]
            return current, goals, sim.pose(sim.active)[0]
        return [("block", block_corners(sim.object_xy, sim.yaw))], [block_corners(sim.goal, sim.goal_yaw)], sim.object_xy

    def capture(self, status=""):
        sim = self.sim
        self.renderer.update_scene(sim.data, self.camera)
        image = Image.new("RGB", (WIDTH + PANEL, HEIGHT), "#132236")
        image.paste(Image.fromarray(self.renderer.render()), (0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 56), fill="#132236")
        draw.text((16, 8), self.title, font=self.big, fill="white")
        draw.text((16, 34), "Green outlines: goals  |  Blue dot: tool  |  replay at 2x speed", font=self.small, fill="#d1deeb")
        draw.text((WIDTH + 20, 20), "TOP VIEW", font=self.big, fill="white")
        top_left, bottom_right = self.to_panel(BOUNDS[1], BOUNDS[3]), self.to_panel(BOUNDS[0], BOUNDS[2])
        draw.rectangle((*top_left, *bottom_right), fill="#283b51", outline="#8092a7")
        current, goals, active_xy = self.shapes()
        if getattr(sim, "active", None) != getattr(self, "last_active", None):
            self.trail, self.last_active = [], getattr(sim, "active", None)
        for poly in goals:
            draw.polygon([self.to_panel(*p) for p in poly], outline="#5bffad")
        for name, poly in current:
            draw.polygon([self.to_panel(*p) for p in poly], fill=COLORS[name])
        self.trail.append(np.array(active_xy))
        if len(self.trail) > 1:
            draw.line([self.to_panel(*p) for p in self.trail[-400:]], fill="#ffd27a", width=2)
        tx, ty = self.to_panel(*sim.tcp[:2])
        draw.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill="#51b9ff")
        y = 500
        lines = [f"Step: {sim.steps}  ({sim.steps * 0.05:.1f} s)",
                 f"Active: {getattr(sim, 'active', 'block')}   {self.phase}",
                 f"Error: {100 * sim.distance():.1f} cm / {np.rad2deg(sim.angle_error()):.0f} deg",
                 f"TCP height: {100 * sim.tcp[2]:.1f} cm"]
        for line in lines:
            draw.text((WIDTH + 20, y), line, font=self.small, fill="white")
            y += 26
        if status:
            draw.text((WIDTH + 20, y + 10), status, font=self.big,
                      fill="#9dffd0" if status.startswith(("SUCCESS", "WORD")) else "#ff9a9a")
        self.frames.append(image)

    def wrap(self, act):
        def recording(*args):
            result = act(*args)
            if self.sim.steps % self.every == 0:
                self.capture()
            return result
        return recording

    def save(self, path, status):
        self.capture(status)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gif = self.frames[::2] + [self.frames[-1]]
        durations = [200] * len(gif)
        durations[-1] = 2500
        small = [f.resize((int((WIDTH + PANEL) * 0.4), int(HEIGHT * 0.4))) for f in gif]
        small[0].save(path, save_all=True, append_images=small[1:], duration=durations, loop=0)
        mp4 = path.with_suffix(".mp4")
        import imageio.v2 as imageio
        with imageio.get_writer(mp4, fps=10) as writer:
            for frame in self.frames + [self.frames[-1]] * 20:
                writer.append_data(np.asarray(frame))
        self.renderer.close()


_CACHE = {}


def context(mode):
    if mode not in _CACHE:
        if mode.startswith("word"):
            shared = cfg.MODELS / "stage3_shared.zip" if mode == "word-shared" else None
            _CACHE[mode] = (WordSim(), word_manager.load_policies(shared=shared))
        elif mode == "cube-manager":
            _CACHE[mode] = (CubeManager(),)
        else:
            _CACHE[mode] = (CubeEnv(), policy.load(CUBE_POLICY))
    return _CACHE[mode]


def run(mode, index, seed, gif=None, viewer=None):
    objects = context(mode)
    sim = objects[0] if mode.startswith("word") else objects[0].sim
    if viewer and not getattr(sim, "_viewer", False):
        sim.act, sim._viewer = viewer(sim, sim.act), True
    recorder = None
    if gif:
        names = {"cube": "Block pose, one policy", "cube-manager": "Block pose, manager",
                 "word": "Word ARM, three letter policies", "word-shared": "Word ARM, one shared policy"}
        recorder = Recorder(sim, f"{names[mode]} | {'scene' if mode.startswith('word') else 'task'} {index}")
        plain, sim.act = sim.act, recorder.wrap(sim.act)
    if mode.startswith("word"):
        result = word_manager.Manager(sim, objects[1]).run_scene(word_manager.make_scenes(index + 1, seed)[index])
        status = "WORD ARM BUILT" if result["word_success"] else f"FAILED: {result['failure'] or 'letter not at goal'}"
        result = {k: result[k] for k in ("word_success", "failure", "steps")}
    elif mode == "cube-manager":
        result = objects[0].run(cube_tasks(index + 1, seed)[index])
        status = "SUCCESS" if result["success"] else f"FAILED: {result['reason']}"
    else:
        env, model = objects
        obs, _ = env.reset(options={"task": cube_tasks(index + 1, seed)[index]})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        result = dict(success=info["is_success"], reason=info["reason"], steps=info["steps"])
        status = "SUCCESS" if result["success"] else f"FAILED: {result['reason']}"
    if recorder:
        recorder.save(gif, status)
        sim.act = plain
    return result


def live_viewer():
    import mujoco.viewer
    state = {}

    def attach(sim, act):
        if "viewer" not in state:
            state["viewer"] = mujoco.viewer.launch_passive(sim.model, sim.data)
            state["viewer"].cam.lookat[:] = [0.50, 0.0, 0.18]
            state["viewer"].cam.distance, state["viewer"].cam.azimuth, state["viewer"].cam.elevation = 1.0, 180, -50

        def step(*args):
            start = time.monotonic()
            result = act(*args)
            if not state["viewer"].is_running():
                raise SystemExit
            state["viewer"].sync()
            time.sleep(max(0.0, 0.05 - (time.monotonic() - start)))
            return result
        return step
    return attach


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cube", "cube-manager", "word", "word-shared"])
    parser.add_argument("--index", type=int, default=0, help="task / scene number in the test set")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gif", help="record a replay instead of opening the viewer")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    seed = args.seed or (word_manager.TEST_SEED if args.mode.startswith("word") else CUBE_SEED)
    if args.gif:
        print(run(args.mode, args.index, seed, gif=args.gif))
        return
    viewer, index = live_viewer(), args.index
    while True:
        print(index, run(args.mode, index, seed, viewer=viewer), flush=True)
        index += 1


if __name__ == "__main__":
    main()
