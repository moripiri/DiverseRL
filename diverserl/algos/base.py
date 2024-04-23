from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch


def get_optimizer(
    optimizer_network: List[torch.Tensor],
    optimizer_lr: float,
    optimizer_class: Union[str, Type[torch.optim.Optimizer]],
    optimizer_kwargs: Union[None, Dict[str, Any]],
) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_class(optimizer_network, lr=optimizer_lr, **optimizer_kwargs)

    return optimizer


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
        Build neural networks and buffer required for the algorithm.
        :return:
        """
        pass

    def _fix_observation(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.tensor:
        """
        Fix observation appropriate to torch neural network module.

        :param observation: The input observation
        :return: The input observation in the form of two dimension tensor
        """

        if isinstance(observation, torch.Tensor):
            observation = observation.to(dtype=torch.float32)

        else:
            observation = np.asarray(observation).astype(np.float32)
            observation = torch.from_numpy(observation)

        observation = observation.to(self.device)

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

        :return: Training result (ex. loss, etc.)
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current algorithm's networks and optimizers as a pickle file.

        :return: None
        """
        save_dict = dict()

        for key, value in self.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                save_dict[key] = value.state_dict()

        torch.save(save_dict, f"{path}.pt")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load the saved model.

        :param path: Path to the saved model.
        :return:
        """
        if isinstance(path, str):
            path = Path(path)

        ckpt = torch.load(path)

        for key, value in ckpt.items():
            getattr(self, key).load_state_dict(value)
