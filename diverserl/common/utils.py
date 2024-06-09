import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch.optim
from gymnasium import Env
from gymnasium.wrappers import (AtariPreprocessing, FlattenObservation,
                                FrameStack)
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


def get_wrapper(wrapper_name: str, wrapper_kwargs: Optional[Dict[str, Any]]) -> Tuple[
    Type[gym.Wrapper], Dict[str, Any]]:
    """
    Return Gymnasium's wrapper class and wrapper arguments.

    :param wrapper_name: Name of the gymnasium wrapper.
    :param wrapper_kwargs: Additional arguments to be passed to the gymnasium wrapper.

    :return: Gymnasium's wrapper class and wrapper arguments.
    """
    wrapper_class = getattr(gym.wrappers, wrapper_name)
    wrapper_option = {}

    if wrapper_kwargs is not None:
        for key, value in wrapper_kwargs.items():
            assert isinstance(value, Union[
                int, float, bool, str]), "Value of wrapper_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                wrapper_option[key] = eval(value)
            else:
                wrapper_option[key] = value

    return wrapper_class, wrapper_option


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


class ScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Scale Atari Ram's uint8 observations([0, 255]) to float32 observations([0, 1])
        :param env: gym environment
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return obs / 255.


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


def make_atari_ram_env(env_id: str, env_option: Dict[str, Any], frame_skip: int = 4, frame_stack: int = 4,
                       repeat_action_probability: float = 0.):
    """
    Return Gymnasium's Atari-Ram enviornment with appropriate wrappers.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param frame_skip: number of frames to skip before observation.
    :param frame_stack: number of frames to stack before observation.
    :param repeat_action_probability: probability of taking previous actions instead of taking current action.

    :return: Gymnasium's atari_ram environment.
    """
    env_option['repeat_action_probability'] = repeat_action_probability
    env_option['frameskip'] = frame_skip

    env = gym.make(env_id, **env_option)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ScaleObservation(FlattenObservation(FrameStack(env, num_stack=frame_stack)))

    return env


def make_atari_env(env_id: str, env_option: Dict[str, Any], image_size: int = 84, noop_max: int = 30,
                   frame_skip: int = 4, frame_stack: int = 4,
                   terminal_on_life_loss: bool = True, grayscale_obs: bool = True,
                   repeat_action_probability: float = 0., ):
    """
    Return Gymnasium's Atari environment.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param image_size: size of the image_type observation (image_size, image_size)
    :param noop_max: For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
    :param frame_skip: number of frames to skip before observation.
    :param frame_stack: number of frames to stack before observation.
    :param terminal_on_life_loss: `if True`, then :meth:`step()` returns `terminated=True` whenever a life is lost.
    :param grayscale_obs: Whether to use grayscale observation.
    :param repeat_action_probability: probability of taking previous actions instead of taking current action.
    :return:
    """
    env_option['repeat_action_probability'] = repeat_action_probability
    env_option['frameskip'] = 1  # fixed to 1 for AtariPreprocessing

    env = gym.make(env_id, **env_option)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = FrameStack(
        AtariPreprocessing(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=image_size,
                           terminal_on_life_loss=terminal_on_life_loss,
                           grayscale_obs=grayscale_obs, scale_obs=True), num_stack=frame_stack)

    return env


def make_envs(env_id: str, env_option: Optional[Dict[str, Any]] = None, wrapper_option: Optional[Dict[str, Any]] = None,
              seed: int = 0, num_envs: int = 1, vector_env: bool = True,
              **kwargs: Optional[Dict[str, Any]]
              ) -> \
        Union[gym.Env, gym.vector.SyncVectorEnv]:
    """
    Creates gymnasium environments for training or evaluation.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param wrapper_option: additional arguments for wrapper creation.
    :param seed: random seed.
    :param num_envs: number of environments to generate if sync_vector_env is True.
    :param vector_env: whether to return env as sync_vector_env.

    :return: generated gymnasium environment
    """

    namespace = env_namespace(gym.make(env_id).spec)

    env_option = {} if env_option is None else dict(env_option)
    wrapper_option = {} if wrapper_option is None else dict(wrapper_option)

    def make_env(random_seed: int = 0):
        """
        Create a gymnasium environment generating function.

        :param random_seed: random seed to apply to the environment.
        :return: Function that returns Gymnasium environment.
        """
        def thunk() -> Env:
            """
            Create a Gymnasium environment.

            :return: Gymnasium environment.
            """
            nonlocal env_option
            nonlocal wrapper_option

            env_option = deepcopy(env_option)
            wrapper_option = deepcopy(wrapper_option)

            if namespace == 'atari_env':
                atari_option = {}
                for key in ['image_size', 'noop_max', 'frame_skip', 'frame_stack', 'repeat_action_probability',
                            'terminal_on_life_loss', 'grayscale_obs', 'repeat_action_probability']:
                    try:
                        atari_option[key] = env_option[key]
                        del env_option[key]

                    except:
                        continue

                if '-ram' in env_id:
                    env = make_atari_ram_env(env_id, env_option, **atari_option)

                else:
                    env = make_atari_env(env_id, env_option, **atari_option)

            else:
                env = gym.make(env_id, **env_option)
                env = gym.wrappers.RecordEpisodeStatistics(env)

            for wrapper_name, wrapper_kwargs in wrapper_option.items():
                wrapper_class, wrapper_kwargs = get_wrapper(wrapper_name, wrapper_kwargs)
                env = wrapper_class(env, **wrapper_kwargs)

            env.action_space.seed(random_seed)

            return env

        return thunk

    if vector_env:
        env = gym.vector.SyncVectorEnv([make_env(seed + i) for i in
                                        range(num_envs)])
    else:
        env = make_env(seed)()

    return env
