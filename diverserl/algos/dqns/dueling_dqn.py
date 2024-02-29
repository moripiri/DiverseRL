from typing import Any, Dict, List, Optional, Type, Union

import torch
from gymnasium import spaces

from diverserl.algos.dqns import DDQN
from diverserl.networks import PixelEncoder
from diverserl.networks.dueling_network import DuelingNetwork


class DuelingDQN(DDQN):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            network_type: str = "MLP",
            network_config: Optional[Dict[str, Any]] = None,
            eps_initial: float = 1.0,
            eps_final: float = 0.05,
            decay_fraction: float = 0.5,
            gamma: float = 0.9,
            batch_size: int = 256,
            buffer_size: int = 10 ** 6,
            learning_rate: float = 0.001,
            optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            target_copy_freq: int = 10,
            training_start: int = 1000,
            max_step: int = 1000000,
            device: str = "cpu",
            **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super().__init__(observation_space, action_space, network_type, network_config, eps_initial, eps_final,
                         decay_fraction, gamma, batch_size, buffer_size, learning_rate, optimizer, optimizer_kwargs,
                         target_copy_freq, training_start, max_step, device, **kwargs)

    def __repr__(self) -> str:
        return "Dueling_DQN"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"MLP": {"Q_network": DuelingNetwork, "Encoder": PixelEncoder}}
