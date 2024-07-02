from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import gymnasium as gym

from diverserl.algos.base import DeepRL


class PixelRL(DeepRL):
    def __init__(
            self, env: gym.Env, eval_env: gym.Env, network_type: str, network_list: Dict[str, Any],
            network_config: Dict[str, Any],
            device: str = "cpu"
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, network_type=network_type, network_list=network_list,
                         network_config=network_config,
                         device=device)

    def __repr__(self) -> str:
        return "PixelRL"

    def _type_assertion(self):
        assert isinstance(self.observation_space, gym.spaces.Box) and isinstance(self.state_dim, tuple)
