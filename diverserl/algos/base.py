from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from diverserl.common.utils import (find_action_space, find_observation_space,
                                    set_network_configs)


class BaseRL(ABC):
    def __init__(self) -> None:
        """
        Base class for all algorithms in DiverseRL.
        """

    @abstractmethod
    def __repr__(self) -> str:
        return "BaseRL"

    def _find_env_space(self, env: Union[gym.Env, gym.vector.VectorEnv]) -> None:
        """
        Find environment's observation_space and action_space, action space's discreteness and action scale, bias
        :param env:
        :return:
        """
        self.observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.action_space = getattr(env, "single_action_space", env.action_space)

        self.state_dim: Union[int, Tuple[int, ...]] = find_observation_space(self.observation_space)
        self.action_dim, self.discrete_action, self.action_scale, self.action_bias = find_action_space(self.action_space)

    @abstractmethod
    def _type_assertion(self):
        pass


class DeepRL(BaseRL, ABC):
    """
    Abstract base class for Deep RL algorithms.
    """

    def __init__(
            self, env: Optional[gym.vector.VectorEnv], eval_env: gym.vector.VectorEnv, network_type: str,
            network_list: Dict[str, Any],
            network_config: Optional[Dict[str, Any]],
            device: str = "cpu"
    ) -> None:
        """
        The base of Deep RL algorithms

        :param env: Gymnasium environment to train the algorithm.
        :param eval_env: Gymnasium environment to evaluate the algorithm.
        :param network_type: Type of DeepRL algorithm networks.
        :param network_list: Dicts of required networks for the algorithm and its network class.
        :param network_config: Configurations of the DeepRL algorithm networks.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__()
        self.env = env
        self.eval_env = eval_env

        if env is not None:
            self._find_env_space(env)
            self.num_envs = env.num_envs

        else:
            self._find_env_space(eval_env)
            self.num_envs = eval_env.num_envs

        self._type_assertion()

        self.network_type, self.network_config = set_network_configs(network_type, network_list, network_config)

        self.device = device
        self.training_count = 0

    def __repr__(self) -> str:
        return "DeepRL"

    def _type_assertion(self):
        assert isinstance(self.observation_space, gym.spaces.Box) and isinstance(self.state_dim, int)

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
    def train(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
