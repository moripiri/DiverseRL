from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diverserl.algos.classic_rl.base import ClassicRL


class SARSA(ClassicRL):
    def __init__(self,  observation_space: spaces.Space, action_space: spaces.Space, gamma: float = 0.9, alpha: float = 0.1, eps: float = 0.1, **kwargs: Optional[Dict[str, Any]]) -> None:
        """
        Tabular SARSA algorithm.

        Reinforcement Learning: An Introduction Chapter 6, Richard S. Sutton and Andrew G. Barto

        :param gamma: The discount factor
        :param alpha: Step-size parameter (learning rate)
        :param eps: Probability to conduct random action during training.
        """
        super().__init__(observation_space, action_space)

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = (
            np.zeros([self.state_dim, self.action_dim])
            if isinstance(self.state_dim, int)
            else np.zeros([*self.state_dim, self.action_dim])
        )

    def __repr__(self) -> str:
        return "SARSA"

    def get_action(self, observation: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Get the SARSA action in probability 1-self.eps from an observation (in training mode)

        :param observation: The input observation
        :return: The SARSA agent's action
        """
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation], axis=-1)

        return action

    def eval_action(self, observation: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Get the SARSA action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The SARSA agent's action (in evaluation mode)
        """
        action = np.argmax(self.q[observation], axis=-1)

        return action

    def train(self, step_result: Tuple[Any, ...]) -> None:
        """
        Train the SARSA agent.

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        """
        s, a, r, ns, d, t, info = step_result

        na = self.get_action(ns)

        self.q[s, a] = self.q[s, a] + self.alpha * (r + self.gamma * self.q[ns, na] - self.q[s, a])
