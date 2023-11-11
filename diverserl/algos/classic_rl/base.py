from abc import ABC, abstractmethod
from typing import Union

import gymnasium as gym
from gymnasium import spaces


class ClassicRL(ABC):
    def __init__(self, env: gym.Env) -> None:
        assert isinstance(env.observation_space, (spaces.Discrete, spaces.Tuple))
        if isinstance(env.observation_space, spaces.Tuple):
            for item in env.observation_space:
                assert isinstance(item, spaces.Discrete)

        assert isinstance(env.action_space, spaces.Discrete)

        self.state_dim = (
            env.observation_space.n
            if isinstance(env.observation_space, spaces.Discrete)
            else tuple(map(lambda x: x.n, env.observation_space))
        )

        self.action_dim = env.action_space.n

    @abstractmethod
    def __repr__(self) -> str:
        return "ClassicRLAlgorithm"

    @abstractmethod
    def get_action(self, observation: Union[int, tuple[int]]):
        pass

    @abstractmethod
    def eval_action(self, observation: Union[int, tuple[int]]):
        pass

    @abstractmethod
    def train(self, step_result: tuple):
        pass
