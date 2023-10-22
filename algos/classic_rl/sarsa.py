import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space, flatdim
class SARSA:
    def __init__(self, env, gamma=0.9, alpha=0.1, eps=0.1):

        self.state_dim = flatdim(env.observation_space)
        self.action_dim = flatdim(env.action_space)

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):

        if np.random.random() < self.eps:
            act = np.random.randint(low=0, high=self.action_dim - 1)
            print("random", act)

        else:
            act = np.argmax(self.q[s,:])
            print("deterministic", act)

        return act

    def train(self, s, a, r, ns, d, t, info):

        na = self.action(ns)

        self.q[s, a] = self.q[s, a] + self.alpha * (
                r + self.gamma * self.q[ns, na] - self.q[s, a])

        # self.q[s, a] = self.q[s, a] + self.alpha * (
        #             r + self.gamma * np.max(self.q[ns, :]) - self.q[s, a])


if __name__ == '__main__':
    env = gym.make(id='Blackjack-v1', natural=False, sab=False)
    #env = gym.make(id="FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    #env = gym.make("CliffWalking-v0", render_mode='human')
    #env = gym.make("Taxi-v3", render_mode='human')
    #print(env.observation_space)
    # sarsa = SARSA(env, render=False)
    # sarsa.run()
    #
    #print(sarsa.success)
    #exit()
    observation, info = env.reset(seed=42)

    print(observation, info)

    print(flatten(env.observation_space, observation), type(flatten(env.observation_space, observation)[0]), info)
    print(flatten_space(env.observation_space), flatdim(env.observation_space))
    exit()
    terminated = False
    while not terminated:
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        #env.render()
        print(observation, reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
