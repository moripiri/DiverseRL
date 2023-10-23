import numpy as np
from gymnasium import spaces
from diverserl.algos.classic_rl.base import ClassicRL
import gymnasium as gym


class DynaQ(ClassicRL):
    def __init__(self, env: gym.Env, gamma: float = 0.8, alpha:float = 0.1, model_n: int=10, eps:float=0.1):
        super().__init__(env)
        assert env.spec.id != 'Blackjack-v1', f"Currently {self.__repr__()} does not support {env.spec.id}."

        self.gamma = gamma
        self.alpha = alpha
        self.model_n = model_n
        self.eps = eps

        self.q = np.zeros([self.state_dim, self.action_dim]) if isinstance(env.observation_space, spaces.Discrete) \
            else np.zeros([*self.state_dim, self.action_dim])
        self.model_r = np.zeros_like(self.q)
        self.model_ns = np.zeros_like(self.q)

        self.ob_traj = list()
        self.a_traj = list()

    def __repr__(self):
        return "Dyna-Q"

    def get_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation])

        return action

    def train(self, observation, action, reward, next_observation, terminated, truncated, info):

        self.q[observation, action] = self.q[observation, action] + self.alpha * (
                reward + self.gamma * np.max(self.q[next_observation]) - self.q[observation, action])

        self.model_r[observation, action] = reward
        self.model_ns[observation, action] = next_observation

        self.ob_traj.append(observation)
        self.a_traj.append(action)

        if terminated or truncated:
            for _ in range(self.model_n):
                sample = np.random.randint(low=0, high=len(self.ob_traj) - 1)
                s = self.ob_traj[sample]
                a = self.a_traj[sample]

                r = (self.model_r[s, a])
                ns = int(self.model_ns[s, a])

                self.q[s, a] = self.q[s, a] + self.alpha * (r + self.gamma * np.max(self.q[ns]) - self.q[s, a])

            self.ob_traj = list()
            self.a_traj = list()
