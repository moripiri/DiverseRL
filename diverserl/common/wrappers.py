from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class ScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Scale Atari Ram's uint8 observations([0, 255]) to float32 observations([0, 1])
        :param env: Gymnasium environment
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, env.observation_space.shape, dtype=np.float32)

    def observation(self, obs: ObsType) -> WrapperObsType:
        return obs / 255.


class FrameSkipWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, frame_skip: int):
        super().__init__(env)
        self.frame_skip = frame_skip

        assert frame_skip >= 1

    def step(self, action: WrapperActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        observation = None
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
