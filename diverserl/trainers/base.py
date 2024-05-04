import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

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
            env: gym.Env,
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
        :param config: Configuration of the run.
        """

        self.algo = algo
        self.env = env

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

        self.env_id = gym.make(config['env_id']).spec.id.replace('ALE/', '')
        self.env_namespace = env_namespace(gym.make(config['env_id']).spec)

        self.task = self.progress.add_task(
            description=f"[bold]Training [red]{self.algo}[/red] in [grey42]{self.env_id}[/grey42]...[/bold]",
            total=total,
        )
        self.start_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.start_time}_{self.algo}_{self.env_id}"

        if self.do_eval:
            self.eval_env = self.make_eval_env()
        else:
            assert not (self.config['render'] or self.config['record_video']),\
                "Rendering or Recording video is only supported in Evaluation."

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

    def make_eval_env(self):
        from diverserl.common.utils import make_envs
        assert not (self.config['render'] and self.config['record_video']), ValueError(
            "Cannot specify both render and record_video")

        if self.config['render']:
            self.config['env_option']['render_mode'] = 'human'
        elif self.config['record_video']:
            self.config['env_option']['render_mode'] = 'rgb_array'
        else:
            self.config['env_option']['render_mode'] = None

        eval_env = make_envs(**self.config, sync_vector_env=False)

        if self.config['record_video']:
            eval_env = RecordVideo(eval_env, video_folder=f"{LOG_PATH}/{self.run_name}/video", name_prefix='eval_ep')
        return eval_env

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


    def log_episodes(self, infos: Dict[str, Any]) -> None:
        """
        Print episode results to console and Log episode results to tensorboard or/and wandb.

        :param infos: Gymnasium info that contains episode results recorded by Gymnasium's RecordEpisodeStatistics wrapper.
        :return:
        """
        episode_infos = infos['final_info']

        for episode_info, episode_done in zip(episode_infos, infos['_final_info']):
            if episode_done:
                local_step = episode_info['episode']['l'].item()
                episode_reward = episode_info['episode']['r'].item()

                self.progress.console.print(
                    f"Episode: {self.episode:06d} -> Local_step: {local_step:04d}, Total_step: {self.total_step:08d}, Episode_reward: {episode_reward:04.4f}",
                )
                self.log_scalar(
                    {
                        "train/episode_reward": episode_reward,
                        "train/local_step": local_step,
                        "train/total_step": self.total_step,
                        "train/training_count": self.algo.training_count,
                    },
                    self.total_step,
                )

                self.episode += 1
