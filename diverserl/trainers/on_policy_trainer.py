import gymnasium as gym

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.trainers import DeepRLTrainer


class OnPolicyTrainer(DeepRLTrainer):
    def __init__(
        self,
        algo: DeepRL,
        env: gym.Env,
        eval_env: gym.Env,
        training_start: int = 256,
        training_num: int = 1,
        train_type: str = "online",
        max_step: int = 10000,
        do_eval: bool = True,
        eval_every: int = 1000,
        eval_ep: int = 10,
    ) -> None:
        super().__init__(
            algo=algo,
            env=env,
            eval_env=eval_env,
            training_start=training_start,
            training_num=training_num,
            train_type=train_type,
            max_step=max_step,
            do_eval=do_eval,
            eval_every=eval_every,
            eval_ep=eval_ep,
        )
