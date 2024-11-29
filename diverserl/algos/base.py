from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from diverserl.common.utils import set_network_configs


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
        try:
            self.observation_space = env.single_observation_space
        except:
            self.observation_space = env.observation_space
        try:
            self.action_space = env.single_action_space
        except:
            self.action_space = env.action_space

        # state_dim
        if isinstance(self.observation_space, gym.spaces.Box):
            # why use shape? -> Atari Ram envs have uint8 dtype and (256, ) observation_space.shape
            self.state_dim = int(self.observation_space.shape[0]) if len(
                self.observation_space.shape) == 1 else self.observation_space.shape

        elif isinstance(self.observation_space, gym.spaces.Discrete):
            self.state_dim = int(self.observation_space.n)

        elif isinstance(self.observation_space, gym.spaces.Tuple):
            # currently only supports tuple observation_space that consist of discrete spaces (toy_text environment)
            self.state_dim = tuple(map(lambda x: int(x.n), self.observation_space))

        else:
            raise TypeError(f"{self.observation_space} observation_space is currently not supported.")

        # action_dim
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = int(self.action_space.n)
            self.discrete_action = True

        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = int(self.action_space.shape[0])

            if self.action_space.high[0] == np.inf:
                self.action_scale = 1.  #(env.unwrapped.envs[0].action_space.high[0] - env.unwrapped.envs[0].action_space.low[0]) / 2
                self.action_bias = 0.  #(env.unwrapped.envs[0].action_space.high[0] + env.unwrapped.envs[0].action_space.low[0]) / 2
            else:
                self.action_scale = (self.action_space.high[0] - self.action_space.low[0]) / 2
                self.action_bias = (self.action_space.high[0] + self.action_space.low[0]) / 2

            self.discrete_action = False
        else:
            raise TypeError(f"{self.action_space} action_space is currently not supported.")

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
            network_config: Dict[str, Any],
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
