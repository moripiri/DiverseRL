from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import gymnasium as gym
from gymnasium import spaces


class ClassicRL(ABC):
    def __init__(self, env: gym.Env) -> None:
        """
        The base of Classic RL algorithms.
        Check environment validity for Classic RL algorithms.

        :param env: The environment for RL agent to learn from
        """
        assert isinstance(env.observation_space, (spaces.Discrete, spaces.Tuple))
        if isinstance(env.observation_space, spaces.Tuple):
            for item in env.observation_space:
                assert isinstance(item, spaces.Discrete)

        assert isinstance(env.action_space, spaces.Discrete)

        self.state_dim = (
            int(env.observation_space.n)
            if isinstance(env.observation_space, spaces.Discrete)
            else tuple(map(lambda x: x.n, env.observation_space))
        )

        self.action_dim = int(env.action_space.n)

    @abstractmethod
    def __repr__(self) -> str:
        return "ClassicRLAlgorithm"

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
