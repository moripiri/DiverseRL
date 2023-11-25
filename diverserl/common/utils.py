from typing import Any, Dict, Tuple, Type, Union

import torch.optim
from torch import nn


def soft_update(network: nn.Module, target_network: nn.Module, tau: float) -> None:
    """
    Polyak averaging for target networks.

    :param network: Network to be updated
    :param target_network: Target network to be updated
    :param tau: Interpolation factor in polyak averaging for target networks.
    """
    for param, target_param in zip(network.parameters(), target_network.parameters(), strict=True):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def get_optimizer(
    optimizer_class: Union[str, Type[torch.optim.Optimizer]], optimizer_kwargs: Union[None, Dict[str, Any]]
) -> Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]:
    optimizer_class = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    return optimizer_class, optimizer_kwargs
