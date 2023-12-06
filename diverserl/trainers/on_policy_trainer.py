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
        
        
    def evaluate(self) -> None:
        pass
    
    def run(self) -> None:
        pass
