from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from diverserl.algos.base import DeepRL
from diverserl.common.buffer import DatasetBuffer


class OfflineRL(DeepRL):
    def __init__(self, buffer: DatasetBuffer, env: Optional[gym.vector.VectorEnv], eval_env: gym.vector.VectorEnv,
                 network_type: str, network_list: Dict[str, Any],
                 network_config: Optional[Dict[str, Any]],
                 device: str = "cpu"
                 ) -> None:
        super().__init__(env=env, eval_env=eval_env, network_type=network_type, network_list=network_list,
                         network_config=network_config, device=device)
        self.buffer = buffer
        self._type_assertion()

        self.num_envs = eval_env.num_envs
        self.device = device
        self.training_count = 0

        self.buffer.device = device

    @abstractmethod
    def __repr__(self) -> str:
        return "OfflineRL"

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
