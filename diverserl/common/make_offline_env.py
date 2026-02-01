from collections.abc import Callable
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type, cast

import gymnasium as gym
import minari
from gymnasium import Env

import diverserl
import diverserl.common.buffer as buffer_module
import diverserl.common.filters as filters_module
from diverserl.common.buffer import DatasetBuffer, SequenceDatasetBuffer
from diverserl.common.make_env import get_wrapper


def get_dataset_buffer(
    buffer_name: str, buffer_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Type[DatasetBuffer], Dict[str, Any]]:
    buffer_class = getattr(buffer_module, buffer_name)
    buffer_option = {}

    if buffer_kwargs is not None:
        for key, value in buffer_kwargs.items():
            assert isinstance(
                value, (int, float, bool, str)
            ), "Value of buffer_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                buffer_option[key] = eval(value)
            else:
                buffer_option[key] = value

    return buffer_class, buffer_option


def get_filter(filter_name: str, filter_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[
    Callable[..., Tuple[DatasetBuffer, Dict[str, Any]]], Dict[str, Any]]:
    """
    Return filter class and arguments for offline dataset.

    :param filter_name: Name of the dataset filter.
    :param filter_kwargs: Additional arguments to be passed to the dataset filter.

    :return: filter class and arguments.
    """
    filter_class = getattr(filters_module, filter_name)
    filter_option = {}

    if filter_kwargs is not None:
        for key, value in filter_kwargs.items():
            assert isinstance(
                value, (int, float, bool, str)
            ), "Value of wrapper_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                filter_option[key] = eval(value)
            else:
                filter_option[key] = value

    return filter_class, filter_option


def make_offline_envs(
    dataset_id: str,
    buffer_name: str = "DatasetBuffer",
    filter_option: Optional[Dict[str, Any]] = None,
    eval_env_option: Optional[Dict[str, Any]] = None,
    eval_wrapper_option: Optional[Dict[str, Any]] = None,
    seed: int = 1234,
    vector_env: bool = True,
    render: bool = False,
    record: bool = False,
) -> Dict[str, Any]:
    """
    Creates dataset and gymnasium environments for offline training or evaluation.

    :param dataset_id: name of the minari dataset.
    :param buffer_type: Type of DatasetBuffer to use.
    :param filter_option: filter arguments to apply for offline dataset.
    :param eval_env_option: additional arguments for evaluation environment creation.
    :param eval_wrapper_option: additional arguments for evaluation environment wrapper creation.
    :param seed: random seed.
    :param vector_env: whether to return eval_env as sync_vector_env.
    :param record: record the evaluation environment
    :param render: render the evaluation environment

    :return: generated gymnasium environment
    """

    filter_option_dict_raw: Dict[str, Any] = {} if filter_option is None else dict(filter_option)
    base_eval_env_option: Dict[str, Any] = {} if eval_env_option is None else dict(eval_env_option)
    eval_wrapper_option_dict: Dict[str, Any] = {} if eval_wrapper_option is None else dict(eval_wrapper_option)

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
            eval_env_option_local = deepcopy(base_eval_env_option)
            eval_wrapper_option_local = deepcopy(eval_wrapper_option_dict)

            assert not (render_env and record_env), ValueError("Cannot specify both render_env and record")
            if render_env and not record_env:
                eval_env_option_local['render_mode'] = 'human'
            elif not render_env and record_env:
                eval_env_option_local['render_mode'] = 'rgb_array'

            env = dataset.recover_environment(**eval_env_option_local)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            for wrapper_name, wrapper_kwargs in eval_wrapper_option_local.items():
                wrapper_class, wrapper_kwargs = get_wrapper(wrapper_name, wrapper_kwargs, env)
                env = wrapper_class(env, **wrapper_kwargs)

            env.action_space.seed(random_seed)

            return env

        return thunk

    dataset = minari.load_dataset(dataset_id)

    buffer_class, buffer_kwargs = get_dataset_buffer(buffer_name)
    buffer = buffer_class(dataset, **buffer_kwargs)

    for filter_name, filter_kwargs_any in filter_option_dict_raw.items():
        filter_kwargs = {} if filter_kwargs_any is None else cast(Dict[str, Any], filter_kwargs_any)
        filter_func, filter_kwargs = get_filter(filter_name, filter_kwargs)
        buffer, filters_eval_env_option = filter_func(buffer, **filter_kwargs)

        for key, value in filters_eval_env_option.items():
            eval_wrapper_option_dict[key] = value

    buffer.init_buffer()

    if vector_env:
        eval_env = gym.vector.SyncVectorEnv([make_env(seed - 1, render, record) for i in range(1)])

    else:
        eval_env = make_env(seed - 1, render, record)()

    return {'buffer': buffer, 'eval_env': eval_env}
