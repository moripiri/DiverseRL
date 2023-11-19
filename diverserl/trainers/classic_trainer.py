from typing import Any, Tuple, Union

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from diverserl.algos.classic_rl.base import ClassicRL
from diverserl.trainers.base import Trainer


class ClassicTrainer(Trainer):
    def __init__(
        self,
        algo: ClassicRL,
        env: gym.Env,
        eval_env: gym.Env,
        max_episode: int = 1000,
        eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
    ) -> None:
        """
        Trainer for Classic RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param max_episode: Maximum episode to train the classic RL algorithm
        :param eval: Whether to perform evaluation during training
        :param eval_every: Perform evalaution every n episode
        :param eval_ep: How many episodes to run to perform evaluation
        """
        super().__init__(algo, env, eval_env, max_episode, eval, eval_every, eval_ep)

        self.max_episode = max_episode

    def evaluate(self) -> None:
        """
        Evaluate classic RL algorithm
        """
        ep_reward_list = []
        success_list = []
        local_step_list = []

        for episode in range(self.eval_ep):
            observation, info = self.eval_env.reset()
            terminated, truncated = False, False
            success = False
            episode_reward = 0
            local_step = 0

            while not (terminated or truncated):
                action = self.algo.eval_action(observation)
                (
                    next_observation,
                    env_reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(action)

                observation = next_observation
                episode_reward += env_reward

                success = self.distinguish_success(float(env_reward), next_observation)

                local_step += 1

            success_list.append(int(success))
            ep_reward_list.append(episode_reward)
            local_step_list.append(local_step)

        success_rate = sum(success_list) / len(success_list)
        avg_ep_reward = sum(ep_reward_list) / len(ep_reward_list)
        avg_local_step = sum(local_step_list) / len(local_step_list)

        self.progress.console.print("=" * 100, style="bold")
        self.progress.console.print(
            f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}, success_rate: {success_rate:04.2f}",
        )
        self.progress.console.print("=" * 100, style="bold")

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

                if self.eval and episode % self.eval_every == 0:
                    self.evaluate()

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
