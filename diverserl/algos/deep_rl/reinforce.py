from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import CategoricalActor, GaussianActor


class REINFORCE(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "MLP",
        network_config: Optional[Dict[str, Any]] = None,
        buffer_size: int = 10**6,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """
        REINFORCE

        Paper: Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

        :param observation_space: The observation space of the environment.
        :param action_space: The action space of the environment.
        :param network_type: Type of the REINFORCE networks to be used.
        :param network_config: Configurations of the REINFORCE networks.
        :param buffer_size: Maximum length of replay buffer.
        :param gamma: The discount factor.
        :param learning_rate: Learning rate of the network
        :param optimizer: Optimizer class (or str) for the network
        :param optimizer_kwargs: Parameter dict for the optimizer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(
            network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )

        assert isinstance(observation_space, spaces.Box), f"{self} supports only Box type observation space."

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

        self._build_network()
        self.buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=1 if self.discrete else self.action_dim,
            max_size=buffer_size,
            device=self.device,
        )

        optimizer, optimizer_kwargs = get_optimizer(optimizer, optimizer_kwargs)
        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.gamma = gamma

    def __repr__(self) -> str:
        return "REINFORCE"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"MLP": {"Discrete": CategoricalActor, "Continuous": GaussianActor}}

    def _build_network(self) -> None:
        network_class = self.network_list()[self.network_type]
        network_class = network_class["Discrete" if self.discrete else "Continuous"]
        network_config = self.network_config

        self.network = network_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **network_config
        ).train()

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = super()._fix_ob_shape(observation)

        self.network.train()

        with torch.no_grad():
            action, _ = self.network(observation)

        return action.cpu().numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = super()._fix_ob_shape(observation)

        self.network.eval()

        with torch.no_grad():
            action, _ = self.network(observation, deterministic=True)

        return action.cpu().numpy()[0]

    def train(self) -> Dict[str, Any]:
        """
        Train the REINFORCE policy.

        :return: Training result (loss)
        """
        s, a, r, ns, d, t = self.buffer.all_sample()

        returns = torch.zeros_like(r)
        running_return = 0.0
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - d[t])
            returns[t] = running_return

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_prob = self.network.log_prob(s, a)
        loss = -(returns * log_prob).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.delete()

        return {"loss": loss.detach().cpu().numpy()}
