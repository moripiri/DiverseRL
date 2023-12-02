from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import DeterministicActor


class DQN(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "MLP",
        network_config: Optional[Dict[str, Any]] = None,
        eps: float = 0.1,
        gamma: float = 0.9,
        batch_size: int = 256,
        buffer_size: int = 10**6,
        learning_rate: float = 0.001,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        target_copy_freq: int = 10,
        device: str = "cpu",
    ) -> None:
        """
        DQN(Deep-Q Network).

        Paper: Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.

        :param observation_space: Observation space of the environment for RL agent to learn from
        :param action_space: Action space of the environment for RL agent to learn from
        :param network_type: Type of the DQN networks to be used.
        :param network_config: Configurations of the DQN networks.
        :param eps: Probability to conduct random action during training.
        :param gamma: The discount factor
        :param batch_size: Minibatch size for optimizer.
        :param buffer_size: Maximum length of replay buffer.\
        :param learning_rate: Learning rate of the Q-network
        :param optimizer: Optimizer class (or str) for the Q-network
        :param optimizer_kwargs: Parameter dict for the optimizer
        :param target_copy_freq: How many training step to pass to copy Q-network to target Q-network
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(
            network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )

        assert isinstance(observation_space, spaces.Box) and isinstance(
            action_space, spaces.Discrete
        ), f"{self} supports only Box type observation space and Discrete type action space."

        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.n

        self._build_network()

        self.buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=1, max_size=buffer_size, device=self.device)

        optimizer, optimizer_kwargs = get_optimizer(optimizer, optimizer_kwargs)
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.eps = eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_copy_freq = target_copy_freq

    def __repr__(self) -> str:
        return "DQN"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"MLP": {"q_network": DeterministicActor}}

    def _build_network(self) -> None:
        q_network_class = self.network_list()[self.network_type]["q_network"]
        q_network_config = self.network_config["q_network"]

        self.q_network = q_network_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **q_network_config
        ).train()
        self.target_q_network = deepcopy(self.q_network).eval()

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Union[int, List[int]]:
        """
        Get the DQN action from an observation (in training mode).

        :param observation: The input observation
        :return: The DQN agent's action
        """
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        else:
            observation = super()._fix_ob_shape(observation)

            self.q_network.train()
            with torch.no_grad():
                return self.q_network(observation).argmax(1).cpu().numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Union[int, List[int]]:
        """
        Get the DQN action from an observation (in evaluation mode).

        :param observation: The input observation
        :return: The DQN agent's action (in evaluation mode)
        """
        observation = super()._fix_ob_shape(observation)
        self.q_network.eval()

        with torch.no_grad():
            return self.q_network(observation).argmax(1).cpu().numpy()[0]

    def train(self) -> Dict[str, Any]:
        """
        Train the DQN policy.

        :return: Training result (loss)
        """
        self.training_count += 1
        self.q_network.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            target_value = r + self.gamma * (1 - d) * self.target_q_network(ns).max(1, keepdims=True)[0]

        selected_value = self.q_network(s).gather(1, a.to(torch.int64))

        loss = F.huber_loss(selected_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_count % self.target_copy_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.detach().cpu().numpy()}
