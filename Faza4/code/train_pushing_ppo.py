import argparse
import csv
import hashlib
import json
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import settings as cfg
from pushing_env import PushingEnv
from plot_pushing_results import update_dashboard


class TrainingLog(BaseCallback):
    def __init__(self, folder, stage, eval_every, eval_episodes, replay, curriculum_steps=0, stage_offset=0, previous_run=None):
        super().__init__()
        self.folder = folder
        self.curriculum_steps = curriculum_steps
        self.stage_offset = stage_offset
        self.stage = stage
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.replay = replay
        self.last_eval = 0
        self.last_save = 0
        self.best_score = (-1, -np.inf)
        if previous_run is not None and (previous_run / "best.json").exists():
            best = json.loads((previous_run / "best.json").read_text())
            self.best_score = (best["success_rate"], -best["mean_error"])
            shutil.copy2(previous_run / "best.zip", folder / "best.zip")
            best.setdefault("inherited_from", str(previous_run.resolve()))
            (folder / "best.json").write_text(json.dumps(best, indent=2))
        self.validation_env = PushingEnv(stage)
        self.started = time.monotonic()
        self.episode_file = (folder / "episodes.csv").open("w")
        self.episodes = csv.DictWriter(self.episode_file, fieldnames=[
            "timesteps", "reward", "length", "is_success", "object_error", "reason"])
        self.episodes.writeheader()
        self.validation_file = (folder / "validation.csv").open("w")
        self.validations = csv.DictWriter(self.validation_file, fieldnames=[
            "timesteps", "success_rate", "mean_error", "mean_reward"])
        self.validations.writeheader()

    def dashboard(self, state):
        update_dashboard(self.folder, dict(stage=self.stage, state=state,
                         timesteps=self.num_timesteps, target_steps=self.model._total_timesteps,
                         curriculum_fraction=min(1.0, (self.stage_offset + self.num_timesteps) / self.curriculum_steps)
                         if self.curriculum_steps else 1.0,
                         elapsed_seconds=time.monotonic() - self.started,
                         stage_steps=self.stage_offset + self.num_timesteps,
                         stage_steps_before_run=self.stage_offset))

    def save_model(self, name):
        temporary = self.folder / (name + ".tmp.zip")
        self.model.save(temporary)
        temporary.replace(self.folder / (name + ".zip"))

    def _on_training_start(self):
        if self.stage == "C":
            self.validate()
        self.dashboard("training")

    def _on_step(self):
        if self.curriculum_steps:
            fraction = min(1.0, (self.stage_offset + self.num_timesteps) / self.curriculum_steps)
            self.training_env.env_method("set_curriculum", fraction)
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes.writerow(dict(timesteps=self.num_timesteps, reward=info["episode"]["r"],
                    length=info["episode"]["l"], is_success=int(info["is_success"]),
                    object_error=info["object_error"], reason=info["reason"]))
        return True

    def validate(self):
        scores, errors, rewards = [], [], []
        tasks = []
        for index in range(self.eval_episodes):
            obs, _ = self.validation_env.reset(seed=10000 + index)
            start = self.validation_env.object_position[:2].copy()
            goal = self.validation_env.goal.copy()
            total = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, timeout, info = self.validation_env.step(action)
                total += reward
                if done or timeout:
                    break
            scores.append(info["is_success"])
            errors.append(info["object_error"])
            rewards.append(total)
            tasks.append(dict(seed=10000 + index, object=start.tolist(), goal=goal.tolist(),
                              success=bool(info["is_success"]), error=info["object_error"],
                              steps=info["steps"], reason=info["reason"], reward=total))
        success = float(np.mean(scores))
        error = float(np.mean(errors))
        self.validations.writerow(dict(timesteps=self.num_timesteps, success_rate=success,
                                      mean_error=error, mean_reward=float(np.mean(rewards))))
        self.validation_file.flush()
        (self.folder / f"validation_tasks_{self.num_timesteps}.json").write_text(
            json.dumps(tasks, indent=2))
        self.save_model("latest")
        score = (success, -error)
        if score > self.best_score:
            self.best_score = score
            self.save_model("best")
            (self.folder / "best.json").write_text(json.dumps(dict(timesteps=self.num_timesteps,
                stage_steps=self.stage_offset + self.num_timesteps,
                success_rate=success, mean_error=error, selection="validation success, then error"), indent=2))
        print(f"VALIDATION stage={self.stage} steps={self.num_timesteps} success={success:.1%} error={error*100:.2f}cm", flush=True)
        if self.stage != "A":
            side_folder = self.folder / f"side_checks_{self.num_timesteps}"
            subprocess.run([sys.executable, str(cfg.ROOT / "code" / "check_side_goals.py"),
                            str(self.folder / "latest.zip"), "--output", str(side_folder)],
                           check=True, stdout=subprocess.DEVNULL)
            shutil.copy2(side_folder / "side_goals.png", self.folder / "side_goals.tmp.png")
            (self.folder / "side_goals.tmp.png").replace(self.folder / "side_goals.png")
        if self.replay:
            from run_pushing_demo import save_replay
            save_replay(self.model, self.folder / "latest_demo.gif", self.stage,
                        title=f"PPO stage {self.stage} | {self.num_timesteps:,} training steps | validation task 1")
        self.last_eval = self.num_timesteps

    def _on_rollout_end(self):
        self.episode_file.flush()
        if self.num_timesteps - self.last_save >= 20000:
            self.save_model("recovery")
            self.last_save = self.num_timesteps
        if self.num_timesteps - self.last_eval >= self.eval_every:
            self.validate()
        self.dashboard("training")

    def _on_training_end(self):
        self.episode_file.flush()
        if self.last_eval != self.num_timesteps:
            self.validate()
        self.dashboard("completed")
        self.episode_file.close()
        self.validation_file.close()
        self.validation_env.close()


def stop_training(signum, frame):
    raise KeyboardInterrupt("Training stopped by signal " + str(signum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--previous-run", type=Path, help="Continue the same stage, preserving curriculum and best validation model")
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--curriculum-steps", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.eval_episodes <= 0 or args.curriculum_steps < 0 or args.n_envs < 1:
        parser.error("Step counts and evaluation episodes must be positive.")
    stage_offset = 0
    if args.previous_run:
        if not args.resume or args.resume.resolve().parent != args.previous_run.resolve():
            parser.error("--previous-run requires --resume from that run folder.")
        previous = json.loads((args.previous_run / "config.json").read_text())
        expected = dict(stage=args.stage, seed=args.seed, curriculum_steps=args.curriculum_steps,
                        validation_seeds=list(range(10000, 10000 + args.eval_episodes)))
        if any(previous.get(key) != value for key, value in expected.items()):
            parser.error("Continuation must keep the stage, seed, curriculum and validation tasks.")
        if previous.get("n_envs", 1) != getattr(args, "n_envs", 1):
            parser.error("Continuation must keep the number of simulation workers.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    factories = [lambda: Monitor(PushingEnv(args.stage)) for _ in range(args.n_envs)]
    env = DummyVecEnv(factories) if args.n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu",
                         learning_rate=cfg.PPO_SETTINGS["learning_rate"])
        model.set_random_seed(args.seed)
        model.ent_coef = cfg.PPO_SETTINGS["ent_coef"]
    else:
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0, **cfg.PPO_SETTINGS)
    if args.previous_run:
        stage_offset = previous.get("stage_steps_before_run", 0) + int(model.num_timesteps)
    if args.curriculum_steps:
        env.env_method("set_curriculum", min(1.0, stage_offset / args.curriculum_steps))
    model.set_logger(configure(str(args.output), ["csv"]))
    config = {k: v for k, v in vars(cfg).items() if k.isupper() and not isinstance(v, Path)}
    config.update(stage=args.stage, seed=args.seed, requested_steps=args.steps,
                  n_envs=args.n_envs, rollout_transitions=args.n_envs*model.n_steps,
                  curriculum_steps=args.curriculum_steps, stage_steps_before_run=stage_offset,
                  previous_run=str(args.previous_run.resolve()) if args.previous_run else None,
                  resume=str(args.resume.resolve()) if args.resume else None,
                  parent_timesteps=int(model.num_timesteps), python=platform.python_version(),
                  validation_seeds=list(range(10000, 10000 + args.eval_episodes)))
    if args.resume:
        config["parent_sha256"] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copytree(cfg.ROOT / "code", args.output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    scene_copy = args.output / "source" / "mujoco"
    scene_copy.mkdir()
    shutil.copy2(cfg.SCENE, scene_copy / "scene.xml")
    shutil.copy2(cfg.SCENE.parent / "franka_fr3" / "fr3.xml", scene_copy / "fr3.xml")
    callback = TrainingLog(args.output, args.stage, args.eval_every, args.eval_episodes, args.replay, args.curriculum_steps, stage_offset, args.previous_run)
    signal.signal(signal.SIGTERM, stop_training)
    try:
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=True)
        model.save(args.output / "last.zip")
        config["actual_steps"] = model.num_timesteps
        config["stage_steps"] = stage_offset + model.num_timesteps
        config["elapsed_seconds"] = time.monotonic() - callback.started
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
    except BaseException:
        callback.save_model("interrupted")
        config["actual_steps"] = model.num_timesteps
        config["stage_steps"] = stage_offset + model.num_timesteps
        config["state"] = "interrupted"
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        callback.dashboard("interrupted — inspect training.log")
        raise
    finally:
        callback.episode_file.close()
        callback.validation_file.close()
        callback.validation_env.close()
        try:
            env.close()
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass


if __name__ == "__main__":
    main()
