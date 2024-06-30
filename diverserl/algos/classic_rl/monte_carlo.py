from functools import reduce
from itertools import product
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from diverserl.algos.classic_rl.base import ClassicRL


class MonteCarlo(ClassicRL):
    def __init__(self, env: gym.Env, eval_env: gym.Env, gamma: float = 0.9, eps: float = 0.1) -> None:
        """
        Tabular Model-free Monte-Carlo control algorithm.

        Reinforcement Learning: An Introduction Chapter 5, Richard S. Sutton and Andrew G. Barto

        :param env: The environment for MC-control agent to learn from
        :param gamma: The discount factor
        :param eps: Probability to conduct random action during training.
        """
        super().__init__(env=env, eval_env=eval_env)

        self.gamma = gamma
        self.eps = eps

        self.trajectory = []
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.pi = np.ones([self.state_dim, self.action_dim]) / self.action_dim
            self.q = np.zeros([self.state_dim, self.action_dim])
            self.returns = [[[] for _ in range(self.action_dim)] for _ in range(self.state_dim)]

        else:
            # Needs improvement
            self.pi = np.ones([*self.state_dim, self.action_dim]) / self.action_dim
            self.q = np.zeros([*self.state_dim, self.action_dim])

            return_list = lambda x, y: [x for _ in range(y)]
            self.returns = list(reduce(return_list, reversed(self.state_dim), [[] for _ in range(self.action_dim)]))
            self.state_index = product(*[range(i) for i in self.state_dim])

    def __repr__(self) -> str:
        return "Monte-Carlo Control"

    def get_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the epsilon-soft Monte-Carlo action in from an observation (in training mode)

        :param observation: The input observation
        :return: The MC control agent's action
        """
        # epsilon-soft policy
        assert all(list(map(lambda x: x >= (self.eps / self.action_dim), self.pi[observation])))

        action = np.random.choice(list(range(self.action_dim)), p=self.pi[observation])

        return action

    def eval_action(self, observation: Union[int, Tuple[int, ...]]) -> int:
        """
        Get the Monte-Carlo action in from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The MC control agent's action
        """
        action = np.argmax(self.pi[observation], axis=-1)

        return action

    def train(self, step_result: Tuple[Any, ...]) -> None:
        """
        Train the MC-control agent.

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        """
        s, a, r, ns, d, t, info = step_result
        self.trajectory.append({"s": s, "a": a, "r": r})

        if d or t:
            self.trajectory.reverse()
            G = 0
            traj_sa_list = list(map(lambda x: [x["s"], x["a"]], self.trajectory))

            for i in range(len(self.trajectory)):
                G = self.gamma * G + self.trajectory[i]["r"]
                cur_s = self.trajectory[i]["s"]
                cur_a = self.trajectory[i]["a"]

                if [cur_s, cur_a] not in traj_sa_list[i + 1:]:
                    if isinstance(self.observation_space, gym.spaces.Discrete):
                        self.returns[cur_s][cur_a].append(G)
                        self.q[cur_s][cur_a] = sum(self.returns[cur_s][cur_a]) / len(self.returns[cur_s][cur_a])
                        optimal_action = np.argmax(self.q, axis=-1)

                        for i in range(self.state_dim):
                            for j in range(self.action_dim):
                                if optimal_action[i] == j:
                                    self.pi[i][j] = 1 - self.eps + self.eps / self.action_dim
                                else:
                                    self.pi[i][j] = self.eps / self.action_dim

                    else:
                        s1, s2, s3 = cur_s
                        self.returns[s1][s2][s3][cur_a].append(G)
                        self.q[cur_s][cur_a] = sum(self.returns[s1][s2][s3][cur_a]) / len(
                            self.returns[s1][s2][s3][cur_a])
                        optimal_action = np.argmax(self.q, axis=-1)

                        for i1, i2, i3 in list(self.state_index):
                            for j in range(self.action_dim):
                                if optimal_action[i1][i2][i3] == j:
                                    self.pi[i1][i2][i3][j] = 1 - self.eps + self.eps / self.action_dim
                                else:
                                    self.pi[i1][i2][i3][j] = self.eps / self.action_dim
            self.trajectory = []
