from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import gymnasium as gym
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn


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
        wandb_group: Optional[str] = None,
        save_model: bool = False,
        save_freq: int = 10**6,
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
        :param wandb_group: Group name of the wandb
        """
        self.algo = algo
        self.env = env
        self.eval_env = eval_env

        self.do_eval = do_eval
        self.eval_every = eval_every
        self.eval_ep = eval_ep

        self.episode = 0
        self.total_step = 0

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
        self.start_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        if self.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=f"./tensorboard/{self.start_time}_{self.algo}_{self.env.spec.id}")

        if self.log_wandb:
            import wandb

            self.wandb = wandb.init(
                project=f"{self.algo}_{self.env.spec.id}",
                group=wandb_group,
                name=f"{self.start_time}",
                id=f"{self.algo}_{self.env.spec.id}",
                notes=f"{self.start_time}_{self.algo}_{self.env.spec.id}",
                tags=["RL", "DiverseRL", f"{self.algo}", f"{self.env.spec.id}"],
            )

        self.save_model = save_model
        self.save_freq = save_freq

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def log_scalar(self, result: Dict[str, Any], step: int) -> None:
        """
        Log training result scalars to tensorboard or/and wandb.

        :param result: Dict that contains scalars as its values and its names as keys.
        :param step: global total step of the result.
        """
        if self.log_tensorboard:
            for tag, value in result.items():
                self.tensorboard.add_scalar(tag=tag, scalar_value=value, global_step=step)

        if self.log_wandb:
            self.wandb.log(result, step=step)
