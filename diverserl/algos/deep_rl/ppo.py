import numpy as np
import torch
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import GaussianActor, CategoricalActor, VNetwork

from typing import Optional, Dict, Any, Union, Type


class PPO(DeepRL):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 network_type: str = "MLP",
                 network_config: Optional[Dict[str, Any]] = None,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__(
            network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )

    def __repr__(self) -> str:
        return "PPO"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        ppo_network_list = {
            "MLP": {"Actor": {"Discrete": CategoricalActor, "Continuous": GaussianActor}, "Critic": VNetwork}}
        return ppo_network_list

    def _build_network(self) -> None:
        pass

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    def train(self) -> Dict[str, Any]:
        pass
