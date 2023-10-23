import numpy as np
from diverserl.algos.classic_rl.base import ClassicRL
import gymnasium as gym


class MonteCarlo(ClassicRL):
    def __init__(self, env: gym.Env, gamma:float=0.9, eps:float=0.1):
        super().__init__(env)
        assert env.spec.id != 'Blackjack-v1', f"Currently {self.__repr__()} does not support {env.spec.id}."

        self.gamma = gamma
        self.eps = eps

        self.pi = np.ones([self.state_dim, self.action_dim]) / self.action_dim
        self.q = np.zeros([self.state_dim, self.action_dim])

        self.trajectory = []
        self.returns = [[[] for _ in range(self.action_dim)] for _ in range(self.state_dim)]

    def __repr__(self):
        return "Monte-Carlo Control"

    def get_action(self, observation):
        # epsilon-soft policy
        assert all(list(map(lambda x: x >= (self.eps / self.action_dim), self.pi[observation])))

        action = np.random.choice(list(range(self.action_dim)), p=self.pi[observation])

        return action

    def train(self, observation, action, reward, next_observation, terminated, truncated, info):
        self.trajectory.append({'s': observation, 'a': action, 'r': reward})

        if terminated or truncated:
            self.trajectory.reverse()
            G = 0
            traj_sa_list = list(map(lambda x: [x['s'], x['a']], self.trajectory))

            for i in range(len(self.trajectory)):
                G = self.gamma * G + self.trajectory[i]['r']
                cur_s = self.trajectory[i]['s']
                cur_a = self.trajectory[i]['a']

                if [cur_s, cur_a] not in traj_sa_list[i + 1:]:
                    self.returns[cur_s][cur_a].append(G)

                    self.q[cur_s][cur_a] = sum(self.returns[cur_s][cur_a]) / len(self.returns[cur_s][cur_a])

                    optimal_action = np.argmax(self.q, axis=1)

                    for i in range(self.state_dim):
                        for j in range(self.action_dim):
                            if optimal_action[i] == j:
                                self.pi[i][j] = 1 - self.eps + self.eps / self.action_dim
                            else:
                                self.pi[i][j] = self.eps / self.action_dim

            self.trajectory = []

