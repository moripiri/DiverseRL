from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diverserl.algos.classic_rl.base import ClassicRL


class DynaQ(ClassicRL):
    def __init__(
        self, env: gym.Env, gamma: float = 0.8, alpha: float = 0.1, model_n: int = 10, eps: float = 0.1, **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """
        Tabular Dyna-Q algorithm.

        Reinforcement Learning: An Introduction Chapter 8, Richard S. Sutton and Andrew G. Barto

        :param env: The environment for Dyna-Q agent to learn from
        :param gamma: The discount factor
        :param alpha: Step-size parameter (learning rate)
        :param model_n: Number of times to train from simulated experiences for every train.
        :param eps: Probability to conduct random action during training.
        """
        super().__init__(env)
        assert env.spec.id != "Blackjack-v1", f"Currently {self.__repr__()} does not support {env.spec.id}."

        self.gamma = gamma
        self.alpha = alpha
        self.model_n = model_n
        self.eps = eps

        self.q = (
            np.zeros([self.state_dim, self.action_dim])
            if isinstance(self.state_dim, int)
            else np.zeros([*self.state_dim, self.action_dim])
        )
        self.model_r = np.zeros_like(self.q)
        self.model_ns = np.zeros_like(self.q)

        self.ob_traj = list()
        self.a_traj = list()

    def __repr__(self) -> str:
        return "Dyna-Q"

    def get_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the Dyna-Q action in probability 1-self.eps from an observation (in training mode)

        :param observation: The input observation
        :return: The Dyna-Q agent's action
        """
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation])

        return action

    def eval_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the Dyna-Q action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The Dyna-Q agent's action (in evaluation mode)
        """
        action = np.argmax(self.q[observation])

        return action

    def train(self, step_result: Tuple[Any, ...]) -> None:
        """
        Train the Dyna-Q agent.

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        """
        s, a, r, ns, d, t, info = step_result

        self.q[s, a] = self.q[s, a] + self.alpha * (r + self.gamma * np.max(self.q[ns]) - self.q[s, a])

        self.model_r[s, a] = r

        self.model_ns[s, a] = ns

        self.ob_traj.append(s)
        self.a_traj.append(a)

        if d or t:
            for _ in range(self.model_n):
                sample = np.random.randint(low=0, high=len(self.ob_traj) - 1)
                sample_s = self.ob_traj[sample]
                sample_a = self.a_traj[sample]

                sample_r = self.model_r[sample_s, sample_a]
                sample_ns = int(self.model_ns[sample_s, sample_a])

                self.q[sample_s, sample_a] = self.q[sample_s, sample_a] + self.alpha * (
                    sample_r + self.gamma * np.max(self.q[sample_ns]) - self.q[sample_s, sample_a]
                )

            self.ob_traj = list()
            self.a_traj = list()
