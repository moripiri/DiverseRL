from abc import ABC, abstractmethod

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn


class Trainer(ABC):
    def __init__(self, algo, env: gym.Env, total: int):
        """
        Base trainer for RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param total: Ending point of the progress bar (max_episode or max_step)
        """
        self.algo = algo
        self.env = env

        self.console = Console(style="bold black")
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        )

        self.task = self.progress.add_task(
            description=f"[bold]Training [red]{self.algo}[/red] in [grey42]{self.env.spec.id}[/grey42]...[/bold]",
            total=total,
        )

    @abstractmethod
    def run(self):
        pass
