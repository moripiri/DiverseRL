from copy import deepcopy
from typing import Any, Dict, Optional, SupportsFloat, Tuple, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import PixelObservationWrapper

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


class PixelOnlyObservationWrapper(PixelObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                 ):
        """
        Replace Env rendering to observation.
        :param env:
        """
        super().__init__(env=env, pixels_only=True, render_kwargs=render_kwargs)
        self.observation_space = self.observation_space['pixels']

    def observation(self, observation):
        """Updates the observations with the pixel observations.

        Args:
            observation: The observation to add pixel observations for

        Returns:
            The updated pixel observations
        """
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation['pixels']


class FrameSkipWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, frame_skip: int):
        super().__init__(env)
        self.frame_skip = frame_skip

        assert frame_skip >= 1

    def step(self, action: WrapperActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
