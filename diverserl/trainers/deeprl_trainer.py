import gymnasium as gym

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.trainers.base import Trainer


class DeepRLTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.Env,
        training_start: int = 256,
        training_num: int = 1,
        train_type: str = "online",
        max_step: int = 10000,
    ) -> None:
        """
        Trainer for Deep RL algorithms.

        :param algo: Deep RL algorithm
        :param env: The environment for RL agent to learn from
        :param training_start: In which total_step to start the training of the Deep RL algorithm
        :param training_num: How many times to run training function in the algorithm each time
        :param train_type: Type of training methods
        :param max_step: Maximum step to run the training
        """
        super().__init__(algo, env, max_step)

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
                            self.algo.train()

                    observation = next_observation
                    episode_reward += reward

                    local_step += 1
                    total_step += 1

                episode += 1

                progress.console.print(
                    f"Episode: {episode:06d} -> Local_step: {local_step:04d}, Total_step: {total_step:08d}, Episode_reward: {episode_reward:04.4f}",
                )

                if total_step >= self.max_step:
                    break

            progress.console.print("=" * 100, style="bold")
