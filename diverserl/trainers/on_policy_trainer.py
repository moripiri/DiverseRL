import gymnasium as gym

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.trainers.base import Trainer


class OnPolicyTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.Env,
        eval_env: gym.Env,
        max_step: int = 1000000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
    ) -> None:
        """
        Trainer for Deep RL(On policy) algorithms.

        :param algo: Deep RL algorithm (Off policy)
        :param env: The environment for RL agent to learn from
        :param eval_env: Then environment for RL agent to evaluate from
        :param max_step: Maximum step to run the training
        :param do_eval: Whether to perform the evaluation.
        :param eval_every: Do evaluation every N step.
        :param eval_ep: Number of episodes to run evaluation
        :param log_tensorboard: Whether to log the training records in tensorboard
        :param log_wandb: Whether to log the training records in Wandb
        """
        super().__init__(
            algo=algo,
            env=env,
            eval_env=eval_env,
            total=max_step,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
        )

        assert self.algo.buffer.save_log_prob
        self.max_step = max_step

        self.horizon = self.algo.horizon
        self.num_epochs = self.algo.num_epochs

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
                action, _ = self.algo.eval_action(observation)

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(action)

                observation = next_observation
                episode_reward += reward

                local_step += 1

            ep_reward_list.append(episode_reward)
            local_step_list.append(local_step)

        avg_ep_reward = sum(ep_reward_list) / len(ep_reward_list)
        avg_local_step = sum(local_step_list) / len(local_step_list)

        self.progress.console.print(
            f"Evaluation Average-> Local_step: {avg_local_step:04.2f}, avg_ep_reward: {avg_ep_reward:04.2f}",
        )
        self.progress.console.print("=" * 100, style="bold")

    def run(self) -> None:
        """
        Train On policy RL algorithm.
        """

        observation, info = self.env.reset()
        episode = 1
        episode_reward = 0
        local_step = 0

        with self.progress as progress:
            total_step = 0
            while True:
                for _ in range(self.horizon):
                    progress.advance(self.task)

                    action, log_prob = self.algo.get_action(observation)

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated, log_prob)

                    observation = next_observation
                    episode_reward += reward

                    local_step += 1
                    total_step += 1

                    if self.do_eval and total_step % self.eval_every == 0:
                        self.evaluate()

                    if terminated or truncated:
                        observation, info = self.env.reset()

                        # progress.console.print(
                        #     f"Episode: {episode:06d} -> Local_step: {local_step:04d}, Total_step: {total_step:08d}, Episode_reward: {episode_reward:04.4f}",
                        # )

                        episode += 1
                        episode_reward = 0
                        local_step = 0

                self.algo.train()

                if total_step >= self.max_step:
                    break

            progress.console.print("=" * 100, style="bold")
