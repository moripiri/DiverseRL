from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import gymnasium as gym

from diverserl.algos.base import BaseRL


class ClassicRL(BaseRL, ABC):
    """
    Abstract base class for classic RL algorithms.
    """

    def __init__(self, env: gym.Env, eval_env: gym.Env) -> None:
        """
        The base of Classic RL algorithms.
        Check environment validity for Classic RL algorithms.

        :param env: The environment for RL agent to learn from
        """
        super().__init__(env, eval_env)

    def _type_assertion(self):
        assert isinstance(self.observation_space, (gym.spaces.Discrete, gym.spaces.Tuple)) and isinstance(
            self.action_space, gym.spaces.Discrete)

    @abstractmethod
    def __repr__(self) -> str:
        return "ClassicRL"

    @abstractmethod
    def get_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the policy action from an observation (in training mode)

        :param observation: The input observation
        :return: The RL agent's action
        """
        pass

    @abstractmethod
    def eval_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the policy action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The RL agent's action (in evaluation mode)
        """
        pass

    @abstractmethod
    def train(self, step_result: Tuple[Any, ...]) -> None:
        """
        Train the classic RL policy.

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        """
        pass
