import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import GaussianActor, CategoricalActor

from typing import Optional, Dict, Any, Union, Type


class REINFORCE(DeepRL):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            network_type: str = "MLP",
            network_config: Optional[Dict[str, Any]] = None,
            gamma: float = 0.99,
            learning_rate: float = 0.001,
            optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            device: str = "cpu",
    ) -> None:
        super().__init__(network_type=network_type, network_list=self.network_list(), network_config=network_config,
                         device=device)

        assert isinstance(observation_space, spaces.Box) and isinstance(
        ), f"{self} supports only Box type observation space."

        self.state_dim = observation_space.shape[0]

        # REINFORCE supports both discrete and continuous action space.
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.discrete = True
        elif isinstance(action_space, spaces.Box):
            self.action_dim = action_space.shape[0]
            self.discrete = False
        else:
            raise TypeError

        optimizer, optimizer_kwargs = get_optimizer(optimizer, optimizer_kwargs)
        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.gamma = gamma


    def __repr__(self) -> str:
        return 'REINFORCE'

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {'MLP': {"Discrete": CategoricalActor, "Continuous": GaussianActor}}

    def _build_network(self) -> None:
        network_class = self.network_list()[self.network_type][self.discrete]
        network_config = self.network_config

        self.network = network_class(state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **network_config).train()

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    def train(self) -> Dict[str, Any]:
        pass

