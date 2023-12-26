import gymnasium as gym

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.trainers.base import Trainer


class DeepRLTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.Env,
        eval_env: gym.Env,
        training_start: int = 1000,
        training_num: int = 1,
        train_type: str = "online",
        max_step: int = 100000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
    ) -> None:
        """
        Trainer for Deep RL (Off policy) algorithms.

        :param algo: Deep RL algorithm (Off policy)
        :param env: The environment for RL agent to learn from
        :param eval_env: Then environment for RL agent to evaluate from
        :param training_start: In which total_step to start the training of the Deep RL algorithm
        :param training_num: How many times to run training function in the algorithm each time
        :param train_type: Type of training methods
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

        assert not self.algo.buffer.save_log_prob

        self.training_start = training_start
        self.training_num = training_num
        self.train_type = train_type

        self.max_step = max_step

    def check_train(self, episode_end: bool) -> bool:
        """
        Whether to conduct training according to the designated training type.

        :param episode_end: Whether the episode has ended
        :return: Whether to conduct training by train_type
        """
        if self.train_type == "online":
            return True
        elif self.train_type == "offline":
            return episode_end
        else:
            return True

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
                ) = self.eval_env.step(action)

                observation = next_observation
                episode_reward += reward

                local_step += 1

            ep_reward_list.append(episode_reward)
            local_step_list.append(local_step)

        avg_ep_reward = sum(ep_reward_list) / len(ep_reward_list)
        avg_local_step = sum(local_step_list) / len(local_step_list)

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
            episode = 0
            total_step = 0

            while True:
                observation, info = self.env.reset()
                terminated, truncated = False, False
                episode_reward = 0
                local_step = 0

                train_results = []

                while not (terminated or truncated):
                    progress.advance(self.task)

                    if total_step < self.training_start:
                        action = self.env.action_space.sample()

                    else:
                        action = self.algo.get_action(observation)
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    self.algo.buffer.add(observation, action, reward, next_observation, terminated, truncated)

                    if total_step >= self.training_start and self.check_train(terminated or truncated):
                        for _ in range(self.training_num):
                            result = self.algo.train()

                            train_results.append(result)

                    observation = next_observation
                    episode_reward += reward

                    local_step += 1
                    total_step += 1

                    if self.do_eval and total_step % self.eval_every == 0:
                        self.evaluate()

                if self.log_tensorboard:
                    for result in train_results:
                        pass

                episode += 1

                progress.console.print(
                    f"Episode: {episode:06d} -> Local_step: {local_step:04d}, Total_step: {total_step:08d}, Episode_reward: {episode_reward:04.4f}",
                )

                if total_step >= self.max_step:
                    break

            progress.console.print("=" * 100, style="bold")
