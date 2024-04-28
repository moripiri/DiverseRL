from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from gymnasium import spaces

from diverserl.algos.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import CategoricalActor, GaussianActor


class REINFORCE(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "Default",
        network_config: Optional[Dict[str, Any]] = None,
        buffer_size: int = 10**6,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
            **kwargs: Optional[Dict[str, Any]]

    ) -> None:
        """
        REINFORCE

        Paper: Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

        :param observation_space: Observation space of the environment for RL agent to learn from
        :param action_space: Action space of the environment for RL agent to learn from
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
            self.action_scale = (action_space.high[0] - action_space.low[0]) / 2
            self.action_bias = (action_space.high[0] + action_space.low[0]) / 2

            self.discrete = False
        else:
            raise TypeError
        self.buffer_size = buffer_size

        self._build_network()


        self.optimizer = get_optimizer(self.network.parameters(), learning_rate, optimizer, optimizer_kwargs)

        self.gamma = gamma

    def __repr__(self) -> str:
        return "REINFORCE"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Network": {"Discrete": CategoricalActor, "Continuous": GaussianActor}}}

    def _build_network(self) -> None:
        network_class = self.network_list()[self.network_type]["Network"]["Discrete" if self.discrete else "Continuous"]

        network_config = self.network_config["Network"]
        if not self.discrete:
            network_config["action_scale"] = self.action_scale
            network_config["action_bias"] = self.action_bias

        self.network = network_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **network_config
        ).train()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]

        self.buffer = buffer_class(
            state_dim=self.state_dim,
            action_dim=1 if self.discrete else self.action_dim,
            max_size=self.buffer_size,
            device=self.device,
            **buffer_config
        )

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = self._fix_observation(observation)

        self.network.train()

        with torch.no_grad():
            action, _ = self.network(observation)

        return action.cpu().numpy()

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = self._fix_observation(observation)
        observation = torch.unsqueeze(observation, dim=0)

        self.network.eval()

        with torch.no_grad():
            action, _ = self.network(observation, deterministic=True)

        return action.cpu().numpy()

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

        return {"loss/loss": loss.detach().cpu().numpy()}
