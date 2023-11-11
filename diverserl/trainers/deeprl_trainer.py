from typing import Union

import gymnasium as gym
from gymnasium import spaces

from diverserl.trainers.base import Trainer


class DeepRLTrainer(Trainer):
    def __init__(
        self, algo, env: gym.Env, training_start=256, training_num=1, train_type="online", max_step: int = 10000
    ) -> None:
        super().__init__(algo, env, max_step)

        self.training_start = training_start
        self.training_num = training_num
        self.train_type = train_type

        self.max_step = max_step

    def run(self) -> None:
        """
        train algorithm
        """
        with self.progress as progress:
            episode = 0
            total_step = 0

            while True:
                observation, info = self.env.reset()
                terminated, truncated = False, False
                episode_reward = 0
                local_step = 0

                while not (terminated or truncated):
                    progress.advance(self.task)

                    if total_step < self.training_start:
                        action = self.env.action_space.sample()

                    else:
                        action = self.algo.get_action(observation)

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated)

                    if total_step >= self.training_start:
                        for _ in range(self.training_num):
                            self.algo.train()

                    observation = next_observation
                    episode_reward += reward

                    local_step += 1
                    total_step += 1

                episode += 1

                progress.console.print(
                    f"Episode: {episode:06d} -> Local_step: {local_step:04d}, Total_step: {total_step:08d}, Episode_reward: {episode_reward:04.4f}",
                )

                if total_step >= self.max_step:
                    break

            progress.console.print("=" * 100, style="bold")
