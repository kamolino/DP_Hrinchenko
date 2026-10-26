"""One policy for A, R and M: the 30 letter observations + a one-hot letter ID [A, R, M] = 33 inputs."""
import gymnasium as gym
import numpy as np

from common import policy
from stage2_letters.letter_env import DEV_SEEDS, LetterEnv, make_tasks
from stage2_letters.word_manager import LETTER_POLICIES

LETTERS = ("A", "R", "M")
ONE_HOT = {letter: np.eye(3, dtype=np.float32)[i] for i, letter in enumerate(LETTERS)}
DAGGER_SEED = 630000
PPO_SEED = 640000


def shared_tasks(per_letter, seed_of):
    tasks = []
    for k, letter in enumerate(LETTERS):
        for task in make_tasks(letter, per_letter, seed_of(k, letter)):
            tasks.append(dict(task, id=k * 100000 + task["id"]))
    return tasks


def dev_tasks(per_letter=200):
    return shared_tasks(per_letter, lambda k, letter: DEV_SEEDS[letter])


def dagger_tasks(per_letter, iteration):
    return shared_tasks(per_letter, lambda k, letter: DAGGER_SEED + 100 * iteration + k)


class SharedLetterEnv(gym.Env):
    """A random letter (1/3 each) per episode, or the task's letter."""

    def __init__(self, task_seed=None):
        self.envs = {letter: LetterEnv(letter) for letter in LETTERS}
        self.rng = np.random.default_rng(task_seed)
        self.letter = "A"
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)

    def _obs(self, obs):
        self.last_obs = np.concatenate([obs, ONE_HOT[self.letter]]).astype(np.float32)
        return self.last_obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        task = (options or {}).get("task")
        if task:
            self.letter = task["letter"]
            obs, info = self.envs[self.letter].reset(options={"task": task})
        else:
            self.letter = LETTERS[int(self.rng.integers(3))]
            obs, info = self.envs[self.letter].reset(seed=int(self.rng.integers(1 << 30)))
        return self._obs(obs), dict(info, letter=self.letter)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[self.letter].step(action)
        return self._obs(obs), reward, terminated, truncated, dict(info, letter=self.letter)

    def close(self):
        for env in self.envs.values():
            env.close()


class Teachers:
    """The three stage-2 letter policies; the active letter's policy labels the state."""

    def __init__(self, paths=None):
        paths = paths or LETTER_POLICIES
        self.policies = {letter: policy.load(paths[letter]) for letter in LETTERS}

    def act(self, env):
        action, _ = self.policies[env.letter].predict(env.last_obs[:30], deterministic=True)
        return np.asarray(action, dtype=np.float32), env.letter


class OneHotPolicy:
    """Gives the word manager a 30-input interface for one letter of the shared policy."""

    def __init__(self, model, letter):
        self.model, self.one_hot = model, ONE_HOT[letter]

    def predict(self, obs, deterministic=True):
        return self.model.predict(np.concatenate([obs, self.one_hot]).astype(np.float32), deterministic=deterministic)
