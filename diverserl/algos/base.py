from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch


class BaseRL(ABC):
    def __init__(self, env: gym.Env):
        self._find_env_space(env)

    @abstractmethod
    def __repr__(self) -> str:
        return "BaseRL"

    def _find_env_space(self, env: gym.Env):
        """
        Find environment's observation_space and action_space, action space's discreteness and action scale, bias
        :param env:
        :return:
        """
        self.observation_space = env.single_observation_space if isinstance(env,
                                                                            gym.vector.SyncVectorEnv) else env.observation_space
        self.action_space = env.single_action_space if isinstance(env, gym.vector.SyncVectorEnv) else env.action_space

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
            self.action_scale = (self.action_space.high[0] - self.action_space.low[0]) / 2
            self.action_bias = (self.action_space.high[0] + self.action_space.low[0]) / 2

            self.discrete_action = False
        else:
            raise TypeError(f"{self.action_space} action_space is currently not supported.")


class DeepRL(BaseRL, ABC):
    """
    Abstract base class for Deep RL algorithms.
    """

    def __init__(
            self, env: gym.Env, network_type: str, network_list: Dict[str, Any], network_config: Dict[str, Any],
            device: str = "cpu"
    ) -> None:
        """
        The base of Deep RL algorithms

        :param network_type: Type of DeepRL algorithm networks.
        :param network_list: Dicts of required networks for the algorithm and its network class.
        :param network_config: Configurations of the DeepRL algorithm networks.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(env)

        self._set_network_configs(network_type, network_list, network_config)

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

    def _set_network_configs(self, network_type: str, network_list: Dict[str, Any],
                             network_config: Dict[str, Any], ) -> None:
        assert network_type in network_list.keys()
        if network_config is None:
            network_config = dict()

        assert set(network_config.keys()).issubset(network_list[network_type].keys())
        for network in network_list[network_type].keys():
            if network not in network_config.keys():
                network_config[network] = dict()

        self.network_type = network_type
        self.network_config = network_config


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
