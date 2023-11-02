from typing import Union

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from diverserl.algos.classic_rl.base import ClassicRL


class ClassicTrainer:
    def __init__(self, algo: ClassicRL, env: gym.Env, max_episode: int = 1000) -> None:
        self.algo = algo
        self.env = env

        self.max_episode = max_episode

        self.console = Console(style="bold black")
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        )

    def run(self) -> None:
        """
        train algorithm
        """
        with self.progress as progress:
            total_step = 0
            success_num = 0
            task = self.progress.add_task(
                description=f"[bold]Training [red]{self.algo}[/red] in [grey42]{self.env.spec.id}[/grey42]...[/bold]",
                total=self.max_episode,
            )

            for episode in range(self.max_episode):
                progress.advance(task)

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

    def process_reward(self, step_result: tuple) -> tuple:
        """
        Post-process reward for better training of gymnasium toy_text environment
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
        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
            if ns == self.algo.state_dim - 1:
                return True

        elif self.env.spec.id in ["Blackjack-v1", "Taxi-v3"]:
            if r > 0.0:
                return True
        else:
            return None

        return False
