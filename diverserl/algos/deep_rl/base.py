from abc import ABC, abstractmethod

import numpy as np
import torch


class DeepRL(ABC):
    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def __repr__(self):
        return "DeepRL"

    def _fix_ob_shape(self, observation: np.ndarray | torch.Tensor) -> torch.tensor:
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
    def get_action(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        pass

    @abstractmethod
    def train(self) -> dict:
        pass
