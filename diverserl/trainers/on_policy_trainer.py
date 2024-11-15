from typing import Any, Dict, Optional, Union

import numpy as np

from diverserl.algos.base import DeepRL
from diverserl.trainers.base import Trainer


class OnPolicyTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        seed: int,
        max_step: int = 1000000,
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
        Trainer for Deep RL(On policy) algorithms.

        :param algo: Deep RL algorithm (Off policy)
        :param env: The environment for RL agent to learn from
        :param max_step: Maximum step to run the training
        :param do_eval: Whether to perform the evaluation.
        :param eval_every: Do evaluation every N step.
        :param eval_ep: Number of episodes to run evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        :param record: Whether to record the evaluation procedure.
        :param save_model: Whether to save the model (both in local and wandb)
        :param save_freq: How often to save the model
        :param configs: The configuration of the training process.
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

        assert self.algo.buffer.save_log_prob, "Off-policy algorithms must use deeprl_trainer."

        self.horizon = self.algo.horizon
        self.num_envs = self.algo.num_envs

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

            while not (terminated or truncated):
                action, _ = self.algo.eval_action(observation)

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
        Train On policy RL algorithm.
        """

        observation, info = self.env.reset()

        with self.progress as progress:
            while self.total_step <= self.max_step:
                for _ in range(self.horizon):
                    progress.advance(self.task, advance=self.num_envs)

                    action, log_prob = self.algo.get_action(observation)

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        infos,
                    ) = self.env.step(action)

                    self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated, log_prob)

                    observation = next_observation

                    self.total_step += self.num_envs

                    if any(terminated) or any(truncated):
                        self.log_episodes(infos)

                    if self.do_eval and self.total_step % self.eval_every == 0:
                        self.evaluate()

                    if self.save_model and self.total_step % self.save_freq == 0:
                        self.algo.save(f"{self.ckpt_folder}/{self.total_step}")

                result = self.algo.train(total_step=self.total_step, max_step=self.max_step)
                self.log_scalar(result, self.total_step)

            if self.log_tensorboard:
                self.tensorboard.close()

            if self.log_wandb:
                if self.save_model:
                    self.wandb.save(f"{self.ckpt_folder}/*.pt")

            progress.console.print("=" * 100, style="bold")
