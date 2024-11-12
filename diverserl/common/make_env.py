import re
from typing import Any, Dict, Optional, Tuple, Type, Union

import ale_py
import shimmy
from gymnasium import Env

import diverserl
from diverserl.common.wrappers import *

gym.register_envs(ale_py)
gym.register_envs(shimmy)


def env_namespace(env_spec: gym.envs.registration.EnvSpec) -> str:
    """
    Return namespace (classic_control, mujoco, atari_env, etc..) of an environment.

    source: gym.envs.registration.pprint_registry()

    :param env_spec: env_specification of gymnasium environment
    :return: namespace
    """
    if env_spec.namespace is None:  #pure gymnasium env
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


def get_wrapper(wrapper_name: str, wrapper_kwargs: Optional[Dict[str, Any]], env: gym.Env) -> Tuple[
    Type[gym.Wrapper], Dict[str, Any]]:
    """
    Return Gymnasium's wrapper class and wrapper arguments.

    :param wrapper_name: Name of the gymnasium wrapper.
    :param wrapper_kwargs: Additional arguments to be passed to the gymnasium wrapper.

    :return: Gymnasium's wrapper class and wrapper arguments.
    """
    try:
        wrapper_class = getattr(gym.wrappers, wrapper_name)
    except:
        wrapper_class = getattr(diverserl.common.wrappers, wrapper_name)

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
            nonlocal env_id
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
                wrapper_class, wrapper_kwargs = get_wrapper(wrapper_name, wrapper_kwargs, env)
                env = wrapper_class(env, **wrapper_kwargs)

            env.action_space.seed(random_seed)

            return env

        return thunk

    if vector_env:
        env = gym.vector.SyncVectorEnv([make_env(seed + i, False, False) for i in
                                        range(num_envs)])
        eval_env = gym.vector.SyncVectorEnv([make_env(seed - 1, render, record) for i in range(1)])

    else:
        env = make_env(seed, False, False)()

        eval_env = make_env(seed - 1, render, record)()

    return env, eval_env
