from abc import ABC
from typing import Any, Dict, Union

import gymnasium as gym

from diverserl.algos.base import DeepRL


class PixelRL(DeepRL, ABC):
    def __init__(
            self, env: gym.vector.VectorEnv, eval_env: gym.vector.VectorEnv, network_type: str, network_list: Dict[str, Any],
            network_config: Dict[str, Any],
            device: str = "cpu"
    ) -> None:
        """
        Pixel-based RL base class

        :param env: Gymnasium environment to train the algorithm.
        :param eval_env: Gymnasium environment to evaluate the algorithm.
        :param network_type: Type of DeepRL algorithm networks.
        :param network_list: Dicts of required networks for the algorithm and its network class.
        :param network_config: Configurations of the DeepRL algorithm networks.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(env=env, eval_env=eval_env, network_type=network_type, network_list=network_list,
                         network_config=network_config,
                         device=device)

    def __repr__(self) -> str:
        return "PixelRL"

    def _type_assertion(self):
        assert isinstance(self.observation_space, gym.spaces.Box) and isinstance(self.state_dim, tuple)
