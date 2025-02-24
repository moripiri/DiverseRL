import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import yaml
from gymnasium.wrappers import RecordVideo
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from diverserl.common.make_env import env_namespace
from diverserl.common.utils import get_project_root, set_seed

ROOT_PATH = get_project_root() #./DiverseRL
LOG_PATH = f"{ROOT_PATH}/logs"


class Trainer(ABC):
    """
    Base trainer class for RL algorithms.
    """

    def __init__(
            self,
            algo,
            seed: int,
            total: int,
            do_eval: bool,
            eval_every: int,
            eval_ep: int,
            log_tensorboard: bool = False,
            log_wandb: bool = False,
            record: bool = False,
            save_model: bool = False,
            save_freq: int = 10 ** 6,
            configs: Optional[Union[str, Dict[str, Any]]] = None,
    ):

        """
        Base trainer for RL algorithms.

        :param algo: RL algorithm
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
        self.eval_env = self.algo.eval_env

        self.seed = set_seed(seed)

        self.do_eval = do_eval
        self.eval_every = eval_every
        self.eval_ep = eval_ep

        self.episode = 0
        self.total_step = 0

        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb
        self.record = record
        self.configs = yaml.safe_load(configs) if isinstance(configs, str) else configs

        self.console = Console()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        )

        if isinstance(self.eval_env, gym.vector.VectorEnv):
            env_spec = self.eval_env.envs[0].spec
        else:
            env_spec = self.eval_env.spec

        self.env_id = env_spec.id.replace('ALE/', '').replace('/', '_') #remove 'ALE/', and replace / to _
        self.env_namespace = env_namespace(env_spec)

        self.task = self.progress.add_task(
            description=f"[bold]Training [red]{self.algo}[/red] in [grey42]{self.env_id}[/grey42]...[/bold]",
            total=total,
        )
        self.start_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.start_time}_{self.algo}_{self.env_id}"


        if self.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=f"{LOG_PATH}/{self.run_name}/tensorboard")

            with open(f"{LOG_PATH}/{self.run_name}/configs.yaml", 'w') as file:
                yaml.dump(self.configs, file, sort_keys=False, default_flow_style=False)

        if self.log_wandb:
            import wandb
            os.makedirs(f"{LOG_PATH}/{self.run_name}", exist_ok=True)

            self.wandb = wandb.init(
                project="DiverseRL",
                dir=f"{LOG_PATH}/{self.run_name}",
                config=self.configs,
                group=f"{self.algo}_{self.env_id}",
                name=f"{self.start_time}",
                id=self.run_name,
                notes=self.run_name,
                monitor_gym=record,
                tags=["RL", "DiverseRL", f"{self.algo}", self.env_namespace, f"{self.env_id}"],
            )

        self.save_model = save_model
        self.save_freq = save_freq if save_freq < total else total

        if self.save_model:
            self.ckpt_folder = f"{LOG_PATH}/{self.run_name}/ckpt"
            os.makedirs(self.ckpt_folder, exist_ok=True)

        if record:
            try:
                self.eval_env.envs[0] = RecordVideo(self.eval_env.envs[0], video_folder=f"{LOG_PATH}/{self.run_name}/video", name_prefix='eval_ep')
            except:
                self.eval_env = RecordVideo(self.eval_env, video_folder=f"{LOG_PATH}/{self.run_name}/video", name_prefix='eval_ep')

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
