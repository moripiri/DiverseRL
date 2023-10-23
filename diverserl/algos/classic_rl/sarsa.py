import numpy as np
from gymnasium import spaces
from diverserl.algos.classic_rl.base import ClassicRL
import gymnasium as gym

class SARSA(ClassicRL):
    def __init__(self, env: gym.Env, gamma:float=0.9, alpha:float=0.1, eps:float=0.1):
        super().__init__(env)

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = np.zeros([self.state_dim, self.action_dim]) if isinstance(env.observation_space, spaces.Discrete) \
            else np.zeros([*self.state_dim, self.action_dim])

    def __repr__(self):
        return "SARSA"

    def get_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)

        else:
            action = np.argmax(self.q[observation])

        return action

    def train(self, observation, action, reward, next_observation, terminated, truncated, info):

        next_action = self.get_action(next_observation)

        self.q[observation, action] = self.q[observation, action] + self.alpha * (
                reward + self.gamma * self.q[next_observation, next_action] - self.q[observation, action])
