# https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py

from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.offline_rl.base import OfflineRL
from diverserl.common.buffer import SequenceDatasetBuffer
from diverserl.common.utils import fix_observation, get_optimizer
from diverserl.networks import DecisionTransformer


class DT(OfflineRL):
    def __init__(self,
                 buffer: SequenceDatasetBuffer,
                 eval_env: gym.vector.VectorEnv,
                 network_type: str = "Default",
                 network_config: Dict[str, Any] = {},
                 sequence_length: int = 20,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: str = "cpu",
                 ) -> None:
        """
        Decision Transformer: Directly learns a policy by using supervised learning on observation-action pairs from expert demonstrations.

        :param buffer: Dataset Buffer that contains expert demonstrations.
        :param eval_env: Gymnasium environment to evaluate the DT algorithm
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

        self.gamma = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length

        self._build_network()

        self.optimizer = get_optimizer(self.transformer.parameters(), learning_rate, optimizer, optimizer_kwargs)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> None:
        pass

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> None:
        pass

    def predict_action(
            self,
            observation: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
            returns: Union[np.ndarray, torch.Tensor],
            time_steps: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict the DT action from an observation (in evaluation mode)

        :param observation: The input observation
        :param action:
        :param returns:
        :param time_steps:

        :return: The DT agents' action (in evaluation mode)
        """

        observation = fix_observation(observation, device=self.device)[:, -self.sequence_length:]
        action = fix_observation(action, device=self.device)[:, -self.sequence_length:]
        returns = fix_observation(returns, device=self.device)[:, -self.sequence_length:]
        time_steps = fix_observation(time_steps, device=self.device).to(torch.long)[-self.sequence_length:]

        self.transformer.eval()
        with torch.no_grad():
            predicted_action = self.transformer(observation, action, returns, time_steps)

        return predicted_action[0, -1].cpu().numpy()

    def __repr__(self) -> str:
        return "DT"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        bc_network_list = {
            "Default": {"Transformer": DecisionTransformer},
        }

        return bc_network_list

    def _build_network(self):
        transformer_class = self.network_list()[self.network_type]["Transformer"]
        transformer_config = self.network_config["Transformer"]

        self.transformer = transformer_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sequence_length=self.sequence_length,
            device=self.device,
            **transformer_config,
        ).train()

    def train(self) -> Dict[str, Any]:
        """
        Train the neural network by using Decision Transformer

        :return: Train result
        """
        s, a, r, timestep, mask = self.buffer.sample(self.batch_size)
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = self.transformer(s, a, r, timestep, padding_mask)
        loss = F.mse_loss(predicted_actions, a.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return {'loss': loss.item()}
