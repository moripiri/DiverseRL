from gymnasium import spaces
import gymnasium as gym
from typing import Any


class ClassicRL:
    def __init__(self, env: gym.Env):
        assert isinstance(env.observation_space, (spaces.Discrete, spaces.Tuple))
        if isinstance(env.observation_space, spaces.Tuple):
            for item in env.observation_space:
                assert isinstance(item, spaces.Discrete)

        assert isinstance(env.action_space, spaces.Discrete)

        self.state_dim = env.observation_space.n if isinstance(env.observation_space, spaces.Discrete) \
            else tuple(map(lambda x: x.n, env.observation_space))

        self.action_dim = env.action_space.n

    def __repr__(self):
        return "ClassicRLAlgorithm"

    def get_action(self, observation):
        raise NotImplementedError

    def train(self, observation, action, reward, next_observation,
              terminated: bool, truncated: bool, info: dict[str, Any]):
        raise NotImplementedError
