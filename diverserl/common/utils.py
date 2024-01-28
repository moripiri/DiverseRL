import random
from typing import Any, Dict, List, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch.optim
from gymnasium import Env
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


def set_seed(seed: int) -> None:
    """
    Sets random seed for deep RL training.
    :param seed: random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, env_option: Dict[str, Any], seed: int = 0) -> Tuple[Env, Env]:
    """
    Creates gym environments for deep RL training.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param seed: random seed.
    :return: environments for training and evaluation
    """

    env = gym.make(env_id, **env_option)
    env.action_space.seed(seed)

    eval_env = gym.make(env_id, **env_option)
    eval_env.action_space.seed(seed - 1)

    return env, eval_env
