from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diverserl.algos.classic_rl.base import ClassicRL


class QLearning(ClassicRL):
    def __init__(self, env: gym.Env, gamma: float = 0.9, alpha: float = 0.1, eps: float = 0.1, **kwargs: Optional[Dict[str, Any]]) -> None:
        """
        Tabular Q-learning algorithm.

        Reinforcement Learning: An Introduction Chapter 6, Richard S. Sutton and Andrew G. Barto

        :param env: The environment for Q-learning agent to learn from
        :param gamma: The discount factor
        :param alpha: Step-size parameter (learning rate)
        :param eps: Probability to conduct random action during training.
        """
        super().__init__(env)

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = (
            np.zeros([self.state_dim, self.action_dim])
            if isinstance(env.observation_space, spaces.Discrete)
            else np.zeros([*self.state_dim, self.action_dim])
        )

    def __repr__(self) -> str:
        return "Q-learning"

    def get_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the Q-learning action in probability 1-self.eps from an observation (in training mode)

        :param observation: The input observation
        :return: The Q-learning agent's action
        """
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation])

        return action

    def eval_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the Q-learning action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The Q-learning agent's action (in evaluation mode)
        """
        action = np.argmax(self.q[observation])

        return action

    def train(self, step_result: Tuple[Any, ...]) -> None:
        """
        Train the Q-learning agent.

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        """

        s, a, r, ns, d, t, info = step_result

        self.q[s, a] = self.q[s, a] + self.alpha * (r + self.gamma * np.max(self.q[ns]) - self.q[s, a])
