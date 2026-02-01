import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.optim
from rich.pretty import pprint
from torch import nn


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent  #./DiverseRL


def find_observation_space(observation_space: gym.spaces.Space) -> Union[int, Tuple[int, ...]]:
    """
    Determine the shape or size of the observation space used in a Gymnasium environment.

    This function processes different types of observation spaces, including `Box`,
    `Discrete`, and `Tuple`, and extracts the appropriate shape or dimensionality.
    - For a `Box` space, it uses the `shape` attribute directly.
    - For a `Discrete` space, it retrieves the number of possible values (`n`).
    - For a `Tuple` space, it combines the sizes of all discrete subspaces.

    :param observation_space: The Gymnasium environment observation space to be processed.
        Types can include:
        - `gym.spaces.Box`: Any array-like observation space with shape information.
        - `gym.spaces.Discrete`: A finite set of discrete integers.
        - `gym.spaces.Tuple`: A collection of multiple subspaces, typically discrete.

    :return:
        - `int` if the space is a `Box` (1D or linearized), or `Discrete` space.
        - `tuple` if the space is a `Tuple` containing discrete subspaces.

    :raises TypeError: If the observation space type is not supported.
    """
    if isinstance(observation_space, gym.spaces.Box):
        # why use shape? -> Atari Ram envs have uint8 dtype and (256, ) observation_space.shape
        state_dim = int(observation_space.shape[0]) if len(
            observation_space.shape) == 1 else observation_space.shape

    elif isinstance(observation_space, gym.spaces.Discrete):
        state_dim = int(observation_space.n)

    elif isinstance(observation_space, gym.spaces.Tuple):
        # currently only supports tuple observation_space that consist of discrete spaces (toy_text environment)
        state_dim = tuple(map(lambda x: int(x.n), observation_space))

    else:
        raise TypeError(f"{observation_space} observation_space is currently not supported.")

    return state_dim


def find_action_space(action_space: gym.spaces.Space) -> Tuple[int, bool, float, float]:
    """
    Determine the action space dimensions and properties from a Gymnasium action space.

    :param action_space: Gymnasium action space of the environment.
        Can be a Discrete space or a continuous Box space.

    :return: A tuple containing:
        - int: The dimensionality of the actions.
        - bool: Whether the action space is discrete (`True`) or continuous (`False`).
        - float: Scale factor for the action values (1.0 for Discrete spaces).
        - float: Bias for the action values (0.0 for Discrete spaces).

    :raises TypeError: If the action space type is not supported.
    """
    # action_dim
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = int(action_space.n)
        discrete_action = True

        action_scale, action_bias = 1.0, 0.0

    elif isinstance(action_space, gym.spaces.Box):
        action_dim = int(action_space.shape[0])

        if action_space.high[0] == np.inf:
            action_scale = 1.  # (env.unwrapped.envs[0].action_space.high[0] - env.unwrapped.envs[0].action_space.low[0]) / 2
            action_bias = 0.  # (env.unwrapped.envs[0].action_space.high[0] + env.unwrapped.envs[0].action_space.low[0]) / 2
        else:
            action_scale = (action_space.high[0] - action_space.low[0]) / 2
            action_bias = (action_space.high[0] + action_space.low[0]) / 2

        discrete_action = False
    else:
        raise TypeError(f"{action_space} action_space is currently not supported.")

    return action_dim, discrete_action, action_scale, action_bias


def fix_observation(observation: Union[np.ndarray, torch.Tensor], device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Fix observation appropriate to torch neural network module.

    :param observation: The input observation
    :param device: torch device to set the observation to

    :return: The input observation in the form of two dimension tensor
    """

    if isinstance(observation, torch.Tensor):
        observation = observation.to(dtype=torch.float32)

    else:
        observation = np.asarray(observation).astype(np.float32)
        observation = torch.from_numpy(observation)

    if device is not None:
        observation = observation.to(device)

    return observation


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
    resolved_class: Any = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
    optimizer_option: Dict[str, Any] = {}

    if optimizer_kwargs is not None:
        for key, value in optimizer_kwargs.items():
            assert isinstance(
                value, (int, float, bool, str)
            ), "Value of optimizer_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                optimizer_option[key] = eval(value)
            else:
                optimizer_option[key] = value

    optimizer = resolved_class(optimizer_network, lr=optimizer_lr, **optimizer_option)

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


def pprint_config(config: Any) -> bool:
    """
    Prettily print the configuration.

    :param config: Configuration of the experiment.
    """
    print('=' * 100)
    pprint(config, expand_all=True)
    print('=' * 100)
    answer = input("Continue? [y/n]: ")
    if answer in ["y", "Y", "", " ", "ㅛ"]:
        return True
    else:
        print("Quitting...")
        return False


def set_network_configs(network_type: str, network_list: Dict[str, Any],
                        network_config: Optional[Dict[str, Any]], ) -> Tuple[str, Dict[str, Any]]:
    assert network_type in network_list.keys()
    if network_config is None:
        network_config = dict()

    assert set(network_config.keys()).issubset(network_list[network_type].keys())
    for network in network_list[network_type].keys():
        if network not in network_config.keys():
            network_config[network] = dict()

    return network_type, network_config
