from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from diverserl.algos.base import DeepRL
from diverserl.trainers.base import Trainer


class DeepRLTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.vector.SyncVectorEnv,
        seed: int,
        training_start: int = 1000,
        training_freq: int = 1,
        training_num: int = 1,
        max_step: int = 100000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        record_video: bool = False,
        save_model: bool = False,
        save_freq: int = 10**6,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """
        Trainer for Deep RL (Off policy) algorithms.

        :param algo: Deep RL algorithm (Off policy)
        :param env: The environment for RL agent to learn from
        :param training_start: In which total_step to start the training of the Deep RL algorithm
        :param training_freq: How frequently train the algorithm (in total_step)
        :param training_num: How many times to run training function in the algorithm each time
        :param max_step: Maximum step to run the training
        :param do_eval: Whether to perform the evaluation
        :param eval_every: Do evaluation every N step
        :param eval_ep: Number of episodes to run evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        :param record_video: Whether to record the evaluation procedure.
        :param save_model: Whether to save the model (both in local and wandb)
        :param save_freq: How often to save the model
        """
        config = locals()
        for key in ['self', 'algo', 'env', '__class__']:
            del config[key]
        for key, value in config['kwargs'].items():
            config[key] = value
        del config['kwargs']

        super().__init__(
            algo=algo,
            env=env,
            total=max_step,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
            record_video=record_video,
            save_model=save_model,
            save_freq=save_freq,
            config=config
        )

        assert not self.algo.buffer.save_log_prob
        self.seed = seed

        self.training_start = training_start
        self.training_freq = training_freq
        self.training_num = training_num

        self.max_step = max_step

    def evaluate(self) -> None:
        """
        Evaluate Deep RL algorithm.
        """
        ep_reward_list = []
        local_step_list = []

        for episode in range(self.eval_ep):
            observation, info = self.eval_env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            local_step = 0

            while not (terminated or truncated):
                action = self.algo.eval_action(observation)

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(action[0])

                observation = next_observation
                episode_reward += reward

                local_step += 1

            ep_reward_list.append(episode_reward)
            local_step_list.append(local_step)

        avg_ep_reward = np.mean(ep_reward_list)
        avg_local_step = np.mean(local_step_list)

        self.log_scalar(
            {"eval/avg_episode_reward": avg_ep_reward, "eval/avg_local_step": avg_local_step}, self.total_step
        )

        self.progress.console.print("=" * 100, style="bold")
        self.progress.console.print(
            f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}",
        )
        self.progress.console.print("=" * 100, style="bold")

    def run(self) -> None:
        """
        Train Deep RL algorithm.
        """
        with self.progress as progress:
            observation, info = self.env.reset()

            while self.total_step <= self.max_step:
                progress.advance(self.task)

                if self.total_step < self.training_start:
                    action = self.env.action_space.sample()

                else:
                    action = self.algo.get_action(observation)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    infos,
                ) = self.env.step(action)

                self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated)

                # train algorithm
                if self.total_step >= self.training_start and (self.total_step % self.training_freq == 0):
                    for _ in range(int(self.training_num)):
                        result = self.algo.train(total_step=self.total_step, max_step=self.max_step)
                        self.log_scalar(result, self.total_step)

                observation = next_observation
                self.total_step += 1

                if self.do_eval and self.total_step % self.eval_every == 0:
                    self.evaluate()

                if self.save_model and self.total_step % self.save_freq == 0:
                    self.algo.save(f"{self.ckpt_folder}/{self.total_step}")

                if any(terminated) or any(truncated):
                    self.log_episodes(infos)

            if self.log_tensorboard:
                self.tensorboard.close()
            if self.log_wandb:
                if self.save_model:
                    self.wandb.save(f"{self.ckpt_folder}/*.pt")

            progress.console.print("=" * 100, style="bold")
