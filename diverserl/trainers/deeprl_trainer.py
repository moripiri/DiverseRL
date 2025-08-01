from typing import Any, Dict, Optional, Union

import numpy as np

from diverserl.algos.base import DeepRL
from diverserl.trainers.base import Trainer


class DeepRLTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        seed: int = 1234,
        training_start: int = 1000,
        training_freq: int = 1,
        training_num: int = 1,
        max_step: int = 100000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        record: bool = False,
        save_model: bool = False,
        save_freq: int = 10**6,
        configs: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:

        """
        Trainer for Deep RL algorithms.

        :param algo: Deep RL algorithm
        :param seed: Random seed
        :param training_start: In which total_step to start the training of the Deep RL algorithm
        :param training_freq: How frequently train the algorithm (in total_step)
        :param training_num: How many times to run training function in the algorithm each time
        :param max_step: Maximum step to run the training
        :param do_eval: Whether to perform the evaluation
        :param eval_every: Do evaluation every N step
        :param eval_ep: Number of episodes to run evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        :param record: Whether to record the evaluation procedure.
        :param save_model: Whether to save the model (both in local and wandb)
        :param save_freq: How often to save the model
        :param configs: The configuration of the training process
        """

        super().__init__(
            algo=algo,
            seed=seed,
            total=max_step,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
            record=record,
            save_model=save_model,
            save_freq=save_freq,
            configs=configs,
        )
        self.env = self.algo.env

        self.save_log_prob = self.algo.buffer.save_log_prob

        self.training_start = training_start
        self.training_freq = training_freq
        self.training_num = training_num
        self.max_step = max_step

        self.num_envs = self.algo.num_envs

    def log_episodes(self, infos: Dict[str, Any]) -> None:
        """
        Print episode results to console and Log episode results to tensorboard or/and wandb.

        :param infos: Gymnasium info that contains episode results recorded by Gymnasium's RecordEpisodeStatistics wrapper.
        :return:
        """

        for i, episode_done in enumerate(infos['_episode']):
            if episode_done:
                local_step = infos['episode']['l'][i]
                episode_reward = infos['episode']['r'][i]

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


    def evaluate(self) -> None:
        """
        Evaluate Deep RL algorithm.
        """
        ep_reward_list = []
        local_step_list = []

        for episode in range(self.eval_ep):
            observation, info = self.eval_env.reset()
            terminated, truncated = False, False

            while not (terminated or truncated):
                if self.save_log_prob:
                    action, _ = self.algo.eval_action(observation)
                else:
                    action = self.algo.eval_action(observation)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(action)

                observation = next_observation

            ep_reward_list.append(info['episode']['r'][0])
            local_step_list.append(info['episode']['l'][0])

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
                progress.advance(self.task, advance=1)
                log_prob = None
                # take action
                log_prob = None
                if self.total_step < self.training_start:
                    action = self.env.action_space.sample()
                else:
                    if self.save_log_prob:
                        action, log_prob = self.algo.get_action(observation)
                    else:
                        action = self.algo.get_action(observation)

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    infos,
                ) = self.env.step(action)

                # add buffer
                self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated, log_prob)

                observation = next_observation
                self.total_step += 1

                # train algorithm
                if self.total_step > self.training_start and (self.total_step % self.training_freq == 0):
                    for _ in range(int(self.training_num)):
                        result = self.algo.train(total_step=self.total_step, max_step=self.max_step)
                        self.log_scalar(result, self.total_step)



                # evaluate
                if self.do_eval and self.total_step % self.eval_every == 0:
                    self.evaluate()

                # save episode
                if self.save_model and self.total_step % self.save_freq == 0:
                    self.algo.save(f"{self.ckpt_folder}/{self.total_step}")

                # log episode
                if any(terminated) or any(truncated):
                    self.log_episodes(infos)

            if self.log_tensorboard:
                self.tensorboard.close()

            if self.log_wandb:
                if self.save_model:
                    self.wandb.save(f"{self.ckpt_folder}/*.pt")

            progress.console.print("=" * 100, style="bold")
