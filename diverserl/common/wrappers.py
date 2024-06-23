from copy import deepcopy
from typing import Any, Dict, SupportsFloat, Tuple, TypeVar

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


class PixelEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Replace Env rendering to observation.
        :param env:
        """
        super().__init__(env=env, pixels_only=False)
    def reset(self):
        pass
    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation(observation)
        info['state'] = observation['state']
        print("wow")
        del observation['state']

        return observation['pixels'], reward, terminated, truncated, info
