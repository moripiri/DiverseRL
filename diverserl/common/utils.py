import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch.optim
from gymnasium import Env
from gymnasium.wrappers import (AtariPreprocessing, FlattenObservation,
                                FrameStack, TransformObservation)
from torch import nn


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


def env_namespace(env_spec: gym.envs.registration.EnvSpec) -> str:
    """
    Return namespace (classic_control, mujoco, atari_env, etc..) of an environment.

    source: gym.envs.registration.pprint_registry()

    :param env_spec: env_specification of gymnasium environment
    :return: namespace
    """
    env_entry_point = re.sub(r":\w+", "", env_spec.entry_point)
    split_entry_point = env_entry_point.split(".")

    if len(split_entry_point) >= 3:
        # If namespace is of the format:
        #  - gymnasium.envs.mujoco.ant_v4:AntEnv
        #  - gymnasium.envs.mujoco:HumanoidEnv
        ns = split_entry_point[2]
    elif len(split_entry_point) > 1:
        # If namespace is of the format - shimmy.atari_env
        ns = split_entry_point[1]
    else:
        # If namespace cannot be found, default to env name
        ns = env_spec.name

    return ns


def make_env(env_id: str, env_option: Dict[str, Any], seed: int = 0,
             image_size: int = 84, noop_max: int = 30, frame_skip: int = 4, frame_stack: int = 4,
             terminal_on_life_loss: bool = True, grayscale_obs: bool = True, repeat_action_probability: float = 0., **kwargs: Optional[Dict[str, Any]]
) -> \
Tuple[Env, Env]:
    """
    Creates gym environments for deep RL training.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param seed: random seed.
    :param image_size: size of the image_type observation (image_size, image_size)
    :param noop_max: used for atari image type observation
    :param frame_skip:
    :param frame_stack:
    :param terminal_on_life_loss:
    :param grayscale_obs:
    :param repeat_action_probability:

    :return: environments for training and evaluation
    """

    namespace = env_namespace(gym.make(env_id).spec)
    if namespace == 'atari_env':
        env_option[
            'repeat_action_probability'] = repeat_action_probability  # add repeat_action_probability in env_option

        if '-ram' in env_id:
            env_option['frameskip'] = frame_skip
            env = TransformObservation(
                FlattenObservation(FrameStack(gym.make(env_id, **env_option), num_stack=frame_stack)),
                lambda obs: obs / 255.)
        else:
            env_option['frameskip'] = 1  # fixed to 1 for AtariPreprocessing
            env = gym.make(env_id, **env_option)

            env = FrameStack(AtariPreprocessing(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=image_size,
                                                terminal_on_life_loss=terminal_on_life_loss,
                                                grayscale_obs=grayscale_obs, scale_obs=True), num_stack=frame_stack)

    else:
        env = gym.make(env_id, **env_option)

    env.action_space.seed(seed)

    eval_env = deepcopy(env)
    eval_env.action_space.seed(seed - 1)

    return env, eval_env
