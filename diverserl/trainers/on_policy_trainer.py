import gymnasium as gym

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.trainers.base import Trainer


class OnPolicyTrainer(Trainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.Env,
        eval_env: gym.Env,
        max_step: int = 10000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
    ) -> None:
        super().__init__(
            algo=algo,
            env=env,
            eval_env=eval_env,
            total=max_step,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
        )

        assert self.algo.buffer.save_log_prob

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
        terminated, truncated = False, False

        with self.progress as progress:
            episode = 0
            total_step = 0

            while True:
                episode_reward = 0
                local_step = 0
