import re
from typing import Optional, Union

from gymnasium import Env
from gymnasium.wrappers import (AtariPreprocessing, FlattenObservation,
                                FrameStack)

from diverserl.common.utils import get_wrapper
from diverserl.common.wrappers import *


def env_namespace(env_spec: gym.envs.registration.EnvSpec) -> str:
    """
    Return namespace (classic_control, mujoco, atari_env, etc..) of an environment.

    source: gym.envs.registration.pprint_registry()

    :param env_spec: env_specification of gymnasium environment
    :return: namespace
    """
    if env_spec.namespace is None: #pure gymnasium env
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
    else:
        ns = env_spec.namespace

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
                           grayscale_obs=grayscale_obs, scale_obs=False), num_stack=frame_stack)

    return env


def make_envs(env_id: str, env_option: Optional[Dict[str, Any]] = None, wrapper_option: Optional[Dict[str, Any]] = None,
              seed: int = 1234, num_envs: int = 1, vector_env: bool = True, render: bool = False, record: bool = False,
              ) -> \
        Tuple[Union[gym.Env, gym.vector.SyncVectorEnv], gym.Env]:
    """
    Creates gymnasium environments for training or evaluation.

    :param env_id: name of the gymnasium environment.
    :param env_option: additional arguments for environment creation.
    :param wrapper_option: additional arguments for wrapper creation.
    :param seed: random seed.
    :param num_envs: number of environments to generate if sync_vector_env is True.
    :param vector_env: whether to return env as sync_vector_env.
    :param record: record the evaluation environment
    :param render: render the evaluation environment

    :return: generated gymnasium environment
    """

    namespace = env_namespace(gym.make(env_id).spec)

    env_option = {} if env_option is None else dict(env_option)
    wrapper_option = {} if wrapper_option is None else dict(wrapper_option)

    def make_env(random_seed: int = 0, render_env: bool = False, record_env: bool = False):
        """
        Create a gymnasium environment generating function.

        :param random_seed: random seed to apply to the environment.
        :param render_env: whether to render the environment.
        :param record_env: whether to record the environment.

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

            assert not (render_env and record_env), ValueError("Cannot specify both render_env and record")
            if render_env and not record_env:
                env_option['render_mode'] = 'human'
            elif not render_env and record_env:
                env_option['render_mode'] = 'rgb_array'

            env = gym.make(env_id, **env_option)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            for wrapper_name, wrapper_kwargs in wrapper_option.items():
                wrapper_class, wrapper_kwargs = get_wrapper(wrapper_name, wrapper_kwargs)
                env = wrapper_class(env, **wrapper_kwargs)

            env.action_space.seed(random_seed)

            return env

        return thunk

    if vector_env:
        env = gym.vector.SyncVectorEnv([make_env(seed + i, False, False) for i in
                                        range(num_envs)])
        eval_env = make_env(seed - 1, render, record)()

    else:
        env = make_env(seed, False, False)()
        eval_env = make_env(seed - 1, render, record)()

    return env, eval_env
