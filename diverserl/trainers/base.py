from abc import ABC, abstractmethod

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn


class Trainer(ABC):
    def __init__(self, algo, env: gym.Env, eval_env: gym.Env, total: int, do_eval: bool, eval_every: int, eval_ep: int):
        """
        Base trainer for RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param total: Ending point of the progress bar (max_episode or max_step)
        :param do_eval: Whether to perform evaluation during training
        :param eval_every: Perform evalaution every n episode or steps
        :param eval_ep: How many episodes to run to perform evaluation
        """
        self.algo = algo
        self.env = env
        self.eval_env = eval_env

        self.do_eval = do_eval
        self.eval_every = eval_every
        self.eval_ep = eval_ep

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
    def evaluate(self):
        pass

    @abstractmethod
    def run(self):
        pass
