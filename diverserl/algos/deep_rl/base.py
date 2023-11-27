from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import torch


class DeepRL(ABC):
    def __init__(
        self, network_type: str, network_list: Dict[str, Any], network_config: Dict[str, Any], device: str = "cpu"
    ) -> None:
        """
        The base of Deep RL algorithms

        :param network_type: Type of DeepRL algorithm networks.
        :param network_list: Dicts of required networks for the algorithm and its network class.
        :param network_config: Configurations of the DeepRL algorithm networks.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """

        assert network_type in network_list.keys()
        if network_config is None:
            network_config = dict()

        assert set(network_config.keys()).issubset(network_list[network_type].keys())
        for network in network_list[network_type].keys():
            if network not in network_config.keys():
                network_config[network] = dict()

        self.network_type = network_type
        self.network_config = network_config

        self.device = device
        self.training_count = 0

    @abstractmethod
    def __repr__(self) -> str:
        return "DeepRL"

    @staticmethod
    @abstractmethod
    def network_list() -> Dict[str, Any]:
        """
        :return: Dicts of required networks for the algorithm and its network class.
        """
        pass

    @abstractmethod
    def _build_network(self) -> None:
        """
        Build neural networks required for the algorithm.
        :return:
        """
        pass

    def _fix_ob_shape(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.tensor:
        """
        Fix observation appropriate for torch neural network module.

        :param observation: The input observation
        :return: The input observation in the form of two dimension tensor
        """
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        else:
            observation = observation.to(dtype=torch.float32)

        if observation.ndim == 1:
            if isinstance(observation, np.ndarray):
                observation = np.expand_dims(observation, axis=0)
                observation = torch.from_numpy(observation)
            else:
                observation = torch.unsqueeze(observation, dim=0)

        return observation

    @abstractmethod
    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get the policy action from an observation (in training mode)

        :param observation: The input observation
        :return: The RL agent's action
        """
        pass

    @abstractmethod
    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get the policy action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The RL agent's action (in evaluation mode)
        """
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the Deep RL policy.

        :return: Training result (ex. loss, etc)
        """
        pass
