import random
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch.optim
from rich.pretty import pprint
from torch import nn


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent #./DiverseRL


def soft_update(network: nn.Module, target_network: nn.Module, tau: float) -> None:
    """
    Polyak averaging for target networks.

    :param network: Network for update
    :param target_network: Target network to be updated
    :param tau: Interpolation factor in polyak averaging for target networks.
    """
    for param, target_param in zip(network.parameters(), target_network.parameters(), strict=True):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(network: nn.Module, target_network: nn.Module) -> None:
    """
    Update target network to network parameters with hard updates.

    :param network: Network for update
    :param target_network: Target network to be updated
    """
    soft_update(network, target_network, 1)


def get_optimizer(
        optimizer_network: List[torch.Tensor],
        optimizer_lr: float,
        optimizer_class: Union[str, Type[torch.optim.Optimizer]],
        optimizer_kwargs: Union[None, Dict[str, Any]],
) -> torch.optim.Optimizer:
    """
    Return optimizer with wanted network, learning rate, optimizer class, and optimizer kwargs.

    :param optimizer_network: Network to be optimized by the designated optimizer.
    :param optimizer_lr: Learning rate of the optimizer.
    :param optimizer_class: Class name, or class of the optimizer.
    :param optimizer_kwargs: Additional arguments to be passed to the optimizer.

    :return: An optimizer with wanted network, learning rate, optimizer class, and optimizer kwargs.
    """
    optimizer_class = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
    optimizer_option = {}

    if optimizer_kwargs is not None:
        for key, value in optimizer_kwargs.items():
            assert isinstance(value, Union[
                int, float, bool, str]), "Value of optimizer_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                optimizer_option[key] = eval(value)
            else:
                optimizer_option[key] = value

    optimizer = optimizer_class(optimizer_network, lr=optimizer_lr, **optimizer_option)

    return optimizer


def set_seed(seed: int) -> int:
    """
    Sets random seed for deep RL training.

    :param seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return seed


def pprint_config(config: Dict[str, Any]) -> bool:
    """
    Prettily print the configuration.

    :param config: Configuration of the experiment.
    """
    print('='*100)
    pprint(config, expand_all=True)
    print('='*100)
    answer = input("Continue? [y/n]: ")
    if answer in ["y", "Y", "", " ", "ㅛ"]:
        return True
    else:
        print("Quitting...")
        return False
