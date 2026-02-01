from typing import Any, Tuple

import gymnasium as gym
import numpy as np


class ScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Scale Atari Ram's uint8 observations([0, 255]) to float32 observations([0, 1])
        :param env: Gymnasium environment
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, env.observation_space.shape, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs / 255.


class FrameSkipWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, frame_skip: int) -> None:
        super().__init__(env)
        self.frame_skip = frame_skip

        assert frame_skip >= 1

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        observation: Any = None
        total_reward: float = 0.0
        terminated, truncated = False, False
        info: dict[str, Any] = {}

        for t in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
