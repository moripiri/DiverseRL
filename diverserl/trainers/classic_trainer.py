from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from diverserl.algos.classic_rl.base import ClassicRL
from diverserl.trainers.base import Trainer


class ClassicTrainer(Trainer):
    def __init__(
        self,
        algo: ClassicRL,
        env: gym.Env,
        eval_env: gym.Env,
        seed: int = 1234,
        max_episode: int = 1000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
        log_tensorboard: bool=False,
        log_wandb: bool = False,
        record_video: bool = False,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """
        Trainer for Classic RL algorithms.

        :param algo: RL algorithm
        :param env: The environment for RL agent to learn from
        :param eval_env: The environment for RL agent to evaluate from
        :param seed: The random seed
        :param max_episode: Maximum episode to train the classic RL algorithm
        :param do_eval: Whether to perform evaluation during training
        :param eval_every: Perform evalaution every n episode
        :param eval_ep: How many episodes to run to perform evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        :param record_video: Whether to record the evaluation procedure.
        """
        config = locals()
        for key in ['self', 'algo', 'env', 'eval_env', '__class__']:
            del config[key]
        for key, value in config['kwargs'].items():
            config[key] = value
        del config['kwargs']

        super().__init__(
            algo=algo,
            env=env,
            eval_env=eval_env,
            total=max_episode,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
            record_video=record_video,
            config=config
        )
        self.seed = seed
        self.max_episode = max_episode

    def evaluate(self) -> None:
        """
        Evaluate classic RL algorithm
        """
        ep_reward_list = []
        success_list = []
        local_step_list = []

        for episode in range(self.eval_ep):
            observation, info = self.eval_env.reset(seed=self.seed - 1)
            terminated, truncated = False, False
            success = False
            episode_reward = 0
            local_step = 0

            while not (terminated or truncated):
                action = self.algo.eval_action(observation)
                (
                    next_observation,
                    env_reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(action)

                observation = next_observation
                episode_reward += env_reward

                success = self.distinguish_success(float(env_reward), next_observation)

                local_step += 1

            success_list.append(int(success))
            ep_reward_list.append(episode_reward)
            local_step_list.append(local_step)

        success_rate = np.mean(success_list)
        avg_ep_reward = np.mean(ep_reward_list)
        avg_local_step = np.mean(local_step_list)

        self.progress.console.print("=" * 100, style="bold")
        self.progress.console.print(
            f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}, success_rate: {success_rate:04.2f}",
        )

        self.log_scalar(
            {"eval/avg_episode_reward": avg_ep_reward, "eval/avg_local_step": avg_local_step}, self.current_episode + 1
        )
        self.progress.console.print("=" * 100, style="bold")

    def run(self) -> None:
        """
        Train classic RL algorithm
        """
        with self.progress as progress:

            success_num = 0

            for episode in range(self.max_episode):
                progress.advance(self.task)
                self.current_episode = episode

                observation, info = self.env.reset(seed=self.seed)
                terminated, truncated = False, False
                success = False
                episode_reward = 0
                local_step = 0

                while not (terminated or truncated):
                    action = self.algo.get_action(observation)
                    (
                        next_observation,
                        env_reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    step_result = self.process_reward(
                        (
                            observation,
                            action,
                            env_reward,
                            next_observation,
                            terminated,
                            truncated,
                            info,
                        )
                    )

                    self.algo.train(step_result)

                    observation = next_observation
                    episode_reward += env_reward

                    success = self.distinguish_success(float(env_reward), next_observation)
                    local_step += 1

                success_num += int(success)
                progress.console.print(
                    f"Episode: {episode:06d} -> Step: {local_step:04d}, Episode_reward: {episode_reward}, success: {success}",
                )

                self.log_scalar(
                    {
                        "train/episode_reward": episode_reward,
                        "train/local_step": local_step,
                        "train/success": int(success),
                    },
                    self.current_episode,
                )
                if self.do_eval and self.current_episode % self.eval_every == 0:
                    self.evaluate()

            progress.console.print("=" * 100, style="bold")
            progress.console.print(f"Success ratio: {success_num / self.max_episode:.3f}")

            if self.log_tensorboard:
                self.tensorboard.close()


    def process_reward(self, step_result: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Post-process reward for better training of gymnasium toy_text environment

        :param step_result: One-step tuple of (state, action, reward, next_state, done, truncated, info)
        :return: Step_result with processed reward
        """

        s, a, r, ns, d, t, info = step_result
        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1"] and r == 0:
            r -= 0.001

        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
            if s == ns:
                r -= 1

            if d and ns != self.algo.state_dim - 1:
                r -= 1
        step_result = (s, a, r, ns, d, t, info)
        return step_result

    def distinguish_success(self, r: float, ns: int) -> Union[bool, None]:
        """
        Determine whether the agent succeeded

        :param r: Environment reward
        :param ns: Next state
        :return: Whether the agent succeeded
        """
        if self.env.spec.id in ["FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v0"]:
            if ns == self.algo.state_dim - 1:
                return True

        elif self.env.spec.id in ["Blackjack-v1", "Taxi-v3"]:
            if r > 0.0:
                return True
        else:
            return None

        return False
