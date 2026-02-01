from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.offline_rl.base import OfflineRL
from diverserl.common.buffer import DatasetBuffer
from diverserl.common.utils import fix_observation, get_optimizer
from diverserl.networks import DeterministicActor
from diverserl.networks.d2rl_networks import D2RLDeterministicActor


class BC(OfflineRL):
    def __init__(self,
                 buffer: DatasetBuffer,
                 eval_env: gym.vector.VectorEnv,
                 network_type: str = "Default",
                 network_config: Optional[Dict[str, Any]] = None,
                 dataset_frac: float = 1.0,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: str = "cpu",
                 ) -> None:
        """
        Behavior Cloning(Any percent BC): Directly learns a policy by using supervised learning on observation-action pairs from expert demonstrations.

        :param buffer: Dataset Buffer that contains expert demonstrations.
        :param eval_env: Gymnasium environment to evaluate the BC algorithm
        :param network_type: Type of neural network to use
        :param network_config: Configurations for the neural networks
        :param dataset_frac: Fraction of dataset to use for training. Chosen by episode reward
        :param batch_size: Minibatch size for optimizer
        :param learning_rate: Learning rate for optimizer
        :param optimizer: Optimizer class (or str) for the network
        :param optimizer_kwargs: Parameter dict for the optimizer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(buffer=buffer,
                         env=None,
                         eval_env=eval_env,
                         network_type=network_type,
                         network_list=self.network_list(),
                         network_config=network_config,
                         device=device)

        self.dataset_frac = dataset_frac
        self.gamma = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._build_network()

        self.optimizer = get_optimizer(self.actor.parameters(), learning_rate, optimizer, optimizer_kwargs)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get the BC action from an observation (in training mode)

        :param observation: The input observation
        :return: The BC agents' action (in training mode)
        """
        observation = fix_observation(observation, device=self.device)

        self.actor.train()
        with torch.no_grad():
            action = self.actor(observation)
            if self.discrete_action:
                action = action.argmax(1)

        return action.cpu().numpy()

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get the BC action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The BC agents' action (in evaluation mode)
        """
        observation = fix_observation(observation, device=self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(observation)
            if self.discrete_action:
                action = action.argmax(1)

        return action.cpu().numpy()

    def __repr__(self) -> str:
        return "BC"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        bc_network_list = {
            "Default": {"Actor": DeterministicActor},
            "D2RL": {"Actor": D2RLDeterministicActor},
        }

        return bc_network_list

    def _build_network(self):
        actor_class = self.network_list()[self.network_type]["Actor"]
        actor_config = self.network_config["Actor"]

        self.actor = actor_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            **actor_config,
        ).train()

    def train(self) -> Dict[str, Any]:
        """
        Train the neural network by using Behavior Cloning(Any percent BC)

        :return: Train result
        """
        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)
        self.training_count += 1
        self.actor.train()

        loss = F.mse_loss(self.actor(s), a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
