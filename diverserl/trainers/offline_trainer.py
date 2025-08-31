from typing import Any, Dict, Optional, Union

import numpy as np

from diverserl.algos.offline_rl.base import OfflineRL
from diverserl.trainers.base import Trainer


class OfflineTrainer(Trainer):
    def __init__(
            self,
            algo: OfflineRL,
            seed: int = 1234,
            max_step: int = 100000,
            do_eval: bool = True,
            eval_every: int = 1000,
            eval_ep: int = 10,
            log_tensorboard: bool = False,
            log_wandb: bool = False,
            record: bool = False,
            save_model: bool = False,
            save_freq: int = 10 ** 6,
            configs: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Trainer for Offline RL algorithms.

        :param algo: Offline RL algorithm
        :param seed: Random seed
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
        self.save_log_prob = False
        self.max_step = max_step



    def print_results(self, result: Dict[str, Any]) -> None:
        """
        Print training result to console.

        :param result: Training result
        :return:
        """
        self.progress.console.print(
            f"Step: {self.total_step:08d} -> loss: {result['loss']:04.4f}",
        )

    def evaluate(self) -> None:
        """
        Evaluate Offline RL algorithm performance.
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
            {"eval/avg_episode_reward": avg_ep_reward,
             "eval/avg_local_step": avg_local_step}, self.total_step
        )
        result = f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}"

        if isinstance(self.algo.buffer.ref_min_score, Union[float, int]) and isinstance(self.algo.buffer.ref_max_score, Union[float, int]):
            normalized_avg_ep_reward = ((avg_ep_reward - self.algo.buffer.ref_min_score)
                                        / (self.algo.buffer.ref_max_score - self.algo.buffer.ref_min_score))
            self.log_scalar(
                {"eval/normalized_avg_episode_reward": normalized_avg_ep_reward}, self.total_step
            )
            result += f", normalized_avg_ep_reward: {normalized_avg_ep_reward:04.2f}"

        self.progress.console.print("=" * 100, style="bold")
        self.progress.console.print(result)
        self.progress.console.print("=" * 100, style="bold")

    def run(self) -> None:
        """
        Train Offline RL algorithm.
        """
        with self.progress as progress:
            while self.total_step <= self.max_step:
                progress.advance(self.task, advance=1)

                result = self.algo.train()

                #self.print_results(result)
                self.log_scalar(result, self.total_step)

                self.total_step += 1

                # evaluate
                if self.do_eval and self.total_step % self.eval_every == 0:
                    self.evaluate()

                # save episode
                if self.save_model and self.total_step % self.save_freq == 0:
                    self.algo.save(f"{self.ckpt_folder}/{self.total_step}")

            if self.log_tensorboard:
                self.tensorboard.close()

            if self.log_wandb:
                if self.save_model:
                    self.wandb.save(f"{self.ckpt_folder}/*.pt")

            progress.console.print("=" * 100, style="bold")


class SequenceOfflineTrainer(OfflineTrainer):
    def __init__(
            self,
            algo: OfflineRL,
            seed: int = 1234,
            max_step: int = 100000,
            do_eval: bool = True,
            eval_every: int = 1000,
            eval_ep: int = 1,
            log_tensorboard: bool = False,
            log_wandb: bool = False,
            record: bool = False,
            save_model: bool = False,
            save_freq: int = 10 ** 6,
            configs: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            algo=algo,
            seed=seed,
            max_step=max_step,
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

        self.max_episode_length = self.eval_env.get_attr('spec')[0].max_episode_steps

    def evaluate(self) -> None:
        """
        Evaluate Offline RL algorithm performance.
        """
        ep_reward_list = []
        local_step_list = []
        target_return = self.eval_env.get_attr('spec')[0].reward_threshold * 0.001# maximum reward in eval_env

        for episode in range(self.eval_ep):
            states = np.zeros((1, self.max_episode_length + 1, self.algo.state_dim), dtype=np.float32)
            actions = np.zeros((1, self.max_episode_length, self.algo.action_dim), dtype=np.float32)
            returns = np.zeros((1, self.max_episode_length + 1, 1), dtype=np.float32)
            time_steps = np.arange(self.max_episode_length, dtype=np.int32)

            observation, info = self.eval_env.reset()
            states[:, 0] = observation[0]
            returns[:, 0] = target_return

            episode_return, step = 0., 0

            terminated, truncated = False, False

            while not (terminated or truncated):
                predicted_action = self.algo.predict_action(  # fix this noqa!!!
                    states[:, :step + 1],
                    actions[:, :step + 1],
                    returns[:, :step + 1],
                    time_steps[:step + 1],
                )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(np.expand_dims(predicted_action, axis=0))

                # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
                actions[:, step] = predicted_action
                states[:, step + 1] = next_observation[0]
                returns[:, step + 1] = returns[:, step] - reward[0]

                episode_return += reward
                step += 1

            ep_reward_list.append(info['episode']['r'][0])
            local_step_list.append(info['episode']['l'][0])

        avg_ep_reward = np.mean(ep_reward_list)
        avg_local_step = np.mean(local_step_list)

        self.log_scalar(
            {"eval/avg_episode_reward": avg_ep_reward,
             "eval/avg_local_step": avg_local_step}, self.total_step
        )
        result = f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}"

        if isinstance(self.algo.buffer.ref_min_score, Union[float, int]) and isinstance(self.algo.buffer.ref_max_score, Union[float, int]):
            normalized_avg_ep_reward = ((avg_ep_reward - self.algo.buffer.ref_min_score)
                                        / (self.algo.buffer.ref_max_score - self.algo.buffer.ref_min_score))
            self.log_scalar(
                {"eval/normalized_avg_episode_reward": normalized_avg_ep_reward}, self.total_step
            )
            result += f", normalized_avg_ep_reward: {normalized_avg_ep_reward:04.2f}"

        self.progress.console.print("=" * 100, style="bold")
        self.progress.console.print(result)
        self.progress.console.print("=" * 100, style="bold")
