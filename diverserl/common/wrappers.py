"""
Wrappers for Gymnasium's toy text environments.
"""
from typing import Any, Dict, SupportsFloat, TypeVar

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper

from diverserl.common.utils import env_namespace

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")

class ToyTextRewardWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        """
        Wrapper that modifies the toy text environment reward for easier classic RL algorithm training.
        :param env:
        """
        Wrapper.__init__(self, env)

        assert env_namespace(env.spec) == 'toy_text', f"{env.spec.id} is not a toy_text environment."

        self.state_dim = (
            env.observation_space.n
            if isinstance(env.observation_space, spaces.Discrete)
            else tuple(map(lambda x: x.n, env.observation_space))
        )

        self.action_dim = env.action_space.n

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""

        # s, a, r, ns, d, t, info = self.env.step(action)
        # if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1"] and r == 0:
        #     r -= 0.001
        #
        # if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
        #     if s == ns:
        #         r -= 1
        #
        #     if d and ns != self.algo.state_dim - 1:
        #         r -= 1
        # step_result = (s, a, r, ns, d, t, info)
        #
        # return step_result


    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        pass



if __name__ == '__main__':
    print(env_namespace(gym.make("Blackjack-v1").spec))
    print(ToyTextRewardWrapper(gym.make("Blackjack-v1")))
