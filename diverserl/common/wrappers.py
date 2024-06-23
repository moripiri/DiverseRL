from typing import TypeVar

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
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
