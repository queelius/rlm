"""Modern dm_control with the pixel/action semantics of CURL's dmc2gym wrapper.

Independent small adapter; upstream agent and replay code are not modified.
The simulator/software version differs from the unpinned 2020 environment.
"""

import copy
from collections import deque

import gym
import numpy as np
from dm_control import suite


class PixelControlEnv(gym.Env):
    """Old-Gym interface, 100px images, three frames, reward-summing repeats."""

    def __init__(self, domain: str, task: str, seed: int, action_repeat: int):
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        self._random = np.random.RandomState(seed)
        self._env = suite.load(domain, task, task_kwargs={"random": self._random},
                               visualize_reward=False)
        self._repeat = action_repeat
        self._max_episode_steps = (1000 + action_repeat - 1) // action_repeat
        spec = self._env.action_spec()
        self._low = np.asarray(spec.minimum, dtype=np.float32)
        self._high = np.asarray(spec.maximum, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=spec.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, shape=(9, 100, 100),
                                               dtype=np.uint8)
        self._frames = deque(maxlen=3)
        self._env_steps = 0
        self._done = True
        self.seed(seed)

    def seed(self, seed: int):
        """Unlike old dmc2gym.seed, also reseed the actual task's initial states."""
        self._random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def get_rng_state(self) -> dict:
        """State sufficient for the next reset at an episode boundary."""
        return copy.deepcopy({"task": self._random.get_state(),
                              "action": self.action_space.np_random.bit_generator.state,
                              "observation": self.observation_space.np_random.bit_generator.state})

    def set_rng_state(self, state: dict) -> None:
        self._random.set_state(state["task"])
        self.action_space.np_random.bit_generator.state = copy.deepcopy(state["action"])
        self.observation_space.np_random.bit_generator.state = copy.deepcopy(state["observation"])

    def _pixels(self):
        pixels = self._env.physics.render(height=100, width=100, camera_id=0)
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        self._env.reset()
        self._frames.clear()
        obs = self._pixels()
        self._frames.extend([obs] * 3)
        self._env_steps = 0
        self._done = False
        return np.concatenate(self._frames, axis=0)

    def step(self, action):
        if self._done:
            raise RuntimeError("reset is required before another episode")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("action is outside normalized action space")
        converted = ((action.astype(np.float64) + 1) / 2 *
                     (self._high - self._low) + self._low).astype(np.float32)
        reward, repeats = 0.0, 0
        before = self._env.physics.get_state().copy()
        for _ in range(self._repeat):
            time_step = self._env.step(converted)
            repeats += 1
            self._env_steps += 1
            reward += float(time_step.reward or 0.0)
            self._done = bool(time_step.last() or self._env_steps >= 1000)
            if self._done:
                break
        self._frames.append(self._pixels())
        discount = float(time_step.discount)
        info = {"env_steps": repeats, "discount": discount,
                "internal_state": before,
                "TimeLimit.truncated": self._done and discount != 0.0}
        return np.concatenate(self._frames, axis=0), reward, self._done, info

    def close(self):
        self._env.close()


def make_env(domain: str, task: str, seed: int, action_repeat: int):
    return PixelControlEnv(domain, task, seed, action_repeat)
