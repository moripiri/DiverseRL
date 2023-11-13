from typing import Any, Tuple, Union

import gymnasium as gym

from diverserl.algos.classic_rl.base import ClassicRL
from diverserl.trainers.base import Trainer


class ClassicTrainer(Trainer):
    def __init__(self, algo: ClassicRL, env: gym.Env, max_episode: int = 1000) -> None:
        """
        Trainer for Classic RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param max_episode: Maximum episode to train the classic RL algorithm.
        """
        super().__init__(algo, env, max_episode)

        self.max_episode = max_episode

    def run(self) -> None:
        """
        Train classic RL algorithm
        """
        with self.progress as progress:
            total_step = 0
            success_num = 0

            for episode in range(self.max_episode):
                progress.advance(self.task)

                observation, info = self.env.reset()
                terminated, truncated = False, False
                success = False
                episode_reward = 0
                local_step = 0

                while not (terminated or truncated):
                    action = self.algo.get_action(observation)
                    (
                        next_observation,
                        env_reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    step_result = self.process_reward(
                        (
                            observation,
                            action,
                            env_reward,
                            next_observation,
                            terminated,
                            truncated,
                            info,
                        )
                    )

                    self.algo.train(step_result)

                    observation = next_observation
                    episode_reward += env_reward

                    success = self.distinguish_success(float(env_reward), next_observation)
                    local_step += 1
                    total_step += 1

                success_num += int(success)
                progress.console.print(
                    f"Episode: {episode:06d} -> Step: {local_step:04d}, Episode_reward: {episode_reward:4d}, success: {success}",
                )
            progress.console.print("=" * 100, style="bold")
            progress.console.print(f"Success ratio: {success_num / self.max_episode:.3f}")

    def process_reward(self, step_result: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Post-process reward for better training of gymnasium toy_text environment

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        :return: Step_result with processed reward
        """

        s, a, r, ns, d, t, info = step_result
        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1"] and r == 0:
            r -= 0.001

        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
            if s == ns:
                r -= 1

            if d and ns != self.algo.state_dim - 1:
                r -= 1
        step_result = (s, a, r, ns, d, t, info)
        return step_result

    def distinguish_success(self, r: float, ns: int) -> Union[bool, None]:
        """
        Determine whether the agent succeeded

        :param r: Environment reward
        :param ns: Next state
        :return: Whether the agent succeeded
        """
        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
            if ns == self.algo.state_dim - 1:
                return True

        elif self.env.spec.id in ["Blackjack-v1", "Taxi-v3"]:
            if r > 0.0:
                return True
        else:
            return None

        return False
