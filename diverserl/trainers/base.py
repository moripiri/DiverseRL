from abc import ABC, abstractmethod
from typing import Any, Dict, List

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from torch.utils.tensorboard import SummaryWriter


class Trainer(ABC):
    def __init__(
        self,
        algo,
        env: gym.Env,
        eval_env: gym.Env,
        total: int,
        do_eval: bool,
        eval_every: int,
        eval_ep: int,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
    ):
        """
        Base trainer for RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param total: Ending point of the progress bar (max_episode or max_step)
        :param do_eval: Whether to perform evaluation during training
        :param eval_every: Perform evalaution every n episode or steps
        :param eval_ep: How many episodes to run to perform evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        """
        self.algo = algo
        self.env = env
        self.eval_env = eval_env

        self.do_eval = do_eval
        self.eval_every = eval_every
        self.eval_ep = eval_ep

        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb

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

        if self.log_tensorboard:
            from datetime import datetime

            start_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

            self.tensorboard = SummaryWriter(log_dir=f"runs/{start_time}_{self.algo}_{self.env.spec.id}")

        if self.log_wandb:
            raise NotImplementedError("Wandb not yet")

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def log(self, result: Dict[str, Any], step: int) -> None:
        if self.log_tensorboard:
            for tag, value in result.items():
                self.tensorboard.add_scalar(tag=tag, scalar_value=value, global_step=step)

        if self.log_wandb:
            pass
