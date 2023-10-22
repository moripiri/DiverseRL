from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatten, flatten_space, flatdim
import numpy as np


class ClassicTrainer:
    def __init__(self, algo, env, max_episode=1000):

        # assert observation and action space is discrete, or can be flattened to discrete space
        self.algo = algo
        self.env = env

        self.max_episode = max_episode

    def run(self):
        """
        train algorithm
        """
        total_step = 0
        success_num = 0

        for episode in range(self.max_episode):
            observation, info = self.env.reset()

            terminated, truncated = False, False
            success = False
            episode_reward = 0
            local_step = 0

            while not (terminated or truncated):
                action = self.algo.get_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)

                processed_reward = self.process_reward(observation, action, reward, next_observation, terminated,
                                                       truncated, info)

                self.algo.train(observation, action, processed_reward, next_observation, terminated, truncated, info)

                observation = next_observation
                episode_reward += reward

                success = self.distinguish_success(reward, next_observation)
                local_step += 1
                total_step += 1

            success_num += int(success)
            print(f"Episode: {episode} -> Step: {local_step}, Episode_reward: {episode_reward}, success: {success}")

        print(success_num)

    def process_reward(self, s, a, r, ns, d, t, info):
        """
        Post-process reward for better training of gymnasium toy_text environment
        """

        if self.env.spec.id in ['FrozenLake-v1', 'FrozenLake8x8-v1'] and r == 0:
            r -= 0.001

        if self.env.spec.id in ['FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v0']:
            if s == ns:
                r -= 1

            if d and ns != self.env.observation_space.n - 1:
                r -= 1

        return r

    def distinguish_success(self, r, ns):
        if self.env.spec.id in ['FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v0']:
            if ns == self.env.observation_space.n - 1:
                return True

        elif self.env.spec.id in ['Blackjack-v1', 'Taxi-v3']:
            if r > 0:
                return True
        else:
            return None

        return False
