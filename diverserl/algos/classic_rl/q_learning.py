from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diverserl.algos.classic_rl.base import ClassicRL


class QLearning(ClassicRL):
    def __init__(self, env: gym.Env, gamma: float = 0.9, alpha: float = 0.1, eps: float = 0.1) -> None:
        super().__init__(env)

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = np.zeros([self.state_dim, self.action_dim]) if isinstance(env.observation_space, spaces.Discrete) \
            else np.zeros([*self.state_dim, self.action_dim])

    def __repr__(self) -> str:
        return "Q-learning"

    def get_action(self, observation: Union[int, tuple[int]]) -> int:
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation])

        return action

    def train(self, step_result: tuple) -> None:
        s, a, r, ns, d, t, info = step_result

        self.q[s, a] = self.q[s, a] + self.alpha * (
                r + self.gamma * np.max(self.q[ns]) - self.q[s, a])
