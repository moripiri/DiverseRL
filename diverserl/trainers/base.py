import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Union

import gymnasium as gym
import yaml
from gymnasium.wrappers import RecordVideo
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from diverserl.common.utils import env_namespace

ROOT_PATH = sys.path[1]
LOG_PATH = f"{ROOT_PATH}/logs"
WANDB_PATH = f"{ROOT_PATH}/wandb"  # Wandb in LOG_PATH causes error.


class Trainer(ABC):
    def __init__(
            self,
            algo,
            env: Union[gym.Env, gym.vector.SyncVectorEnv],
            eval_env: gym.Env,
            total: int,
            do_eval: bool,
            eval_every: int,
            eval_ep: int,
            log_tensorboard: bool = False,
            log_wandb: bool = False,
            save_model: bool = False,
            save_freq: int = 10 ** 6,
            record_video: bool = False,
            config: Dict[str, Any] = None,
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
        :param save_model: Whether to save the RL model
        :param save_freq: How frequently save the RL model
        :param configs: Configuration of the run.
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
        self.record_video = record_video

        self.config = config

        self.console = Console(style="bold black")

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        )
        if isinstance(self.env, gym.vector.SyncVectorEnv):
            self.env_id = self.env.envs[0].spec.id.replace('ALE/', '')
            self.env_namespace = env_namespace(env.envs[0].spec)

        else:
            self.env_id = self.env.spec.id.replace('ALE/', '')
            self.env_namespace = env_namespace(env.spec)

        self.task = self.progress.add_task(
            description=f"[bold]Training [red]{self.algo}[/red] in [grey42]{self.env_id}[/grey42]...[/bold]",
            total=total,
        )
        self.start_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.start_time}_{self.algo}_{self.env_id}"

        if self.record_video:
            self.eval_env = RecordVideo(self.eval_env, video_folder=f"{LOG_PATH}/{self.run_name}/video",
                                        name_prefix='eval_ep')

        if self.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=f"{LOG_PATH}/{self.run_name}/tensorboard")
            with open(f"{LOG_PATH}/{self.run_name}/config.yaml", 'w') as file:
                yaml.dump(self.config, file, sort_keys=False)

        if self.log_wandb:
            import wandb
            os.makedirs(f"{LOG_PATH}/{self.run_name}", exist_ok=True)

            self.wandb = wandb.init(
                project="DiverseRL",
                dir=f"{LOG_PATH}/{self.run_name}",
                config=self.config,
                group=f"{self.algo}_{self.env_id}",
                name=f"{self.start_time}",
                id=self.run_name,
                notes=self.run_name,
                monitor_gym=record_video,
                tags=["RL", "DiverseRL", f"{self.algo}", self.env_namespace, f"{self.env_id}"],
            )

        self.save_model = save_model
        self.save_freq = save_freq if save_freq < total else total

        if self.save_model:
            self.ckpt_folder = f"{LOG_PATH}/{self.run_name}/ckpt"
            os.makedirs(self.ckpt_folder, exist_ok=True)

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
