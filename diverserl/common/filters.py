# Filters for Offline Dataset

from functools import reduce
from typing import Any, Dict, Tuple

import numpy as np

from diverserl.common.buffer import DatasetBuffer


def filter_reward(dataset: DatasetBuffer, dataset_frac: float = 0.7, gamma: float = 0.99) -> Tuple[DatasetBuffer, Dict[str, Any]]:
    """
    Filters rewards from the given dataset based on the provided fraction and discount factor. The rewards are discounted
    and summed for each episode, followed by selecting episodes that meet the reward fraction. The function returns the
    filtered dataset and an empty dictionary.

    :param dataset: The dataset of type MinariDataset to process and filter based on rewards.
    :param dataset_frac: A fraction (float) indicating the proportion of episodes with higher rewards to retain.
    :param gamma: Discount factor (float) to be applied to future rewards in the calculation.

    :return: A tuple where the first element is the filtered MinariDataset.
    """

    assert dataset.dataset is not None, "Filters must be applied before DatasetBuffer.init_buffer()."

    reward_dict = dict(zip(list(map(lambda x: int(x), dataset.dataset['id'])), list(map(lambda x: reduce(
        lambda x, y: (x[0] + y * x[1], x[1] * gamma),
        x,
        (0, 1))[0], dataset.dataset['rewards']))))

    reward_dict = dict(sorted(reward_dict.items(), key=lambda item: item[1]))
    idxs = list(reward_dict.keys())[:int(len(reward_dict) * (1 - dataset_frac))]

    dataset.filter_episodes(idxs)

    return dataset, {}


def normalize_observation(buffer: DatasetBuffer) -> Tuple[DatasetBuffer, Dict[str, Any]]:
    assert buffer.dataset is not None, "Filters must be applied before DatasetBuffer.init_buffer()."
    obs = np.vstack(buffer.dataset['observations'])
    obs_mean, obs_std = obs.mean(axis=0), obs.std(axis=0)
    normalized_observation = []
    for observation in buffer.dataset['observations']:
        normalized_observation.append((observation - obs_mean) / (obs_std + 1e-8))

    buffer.dataset['observations'] = normalized_observation

    return buffer, {'TransformObservation': {'func': lambda obs: (obs-obs_mean)/(obs_std + 1e-8), "observation_space": "env.observation_space"}}


def scale_reward(buffer: DatasetBuffer, scale: float = 0.001) -> Tuple[DatasetBuffer, Dict[str, Any]]:
    assert buffer.dataset is not None, "Filters must be applied before DatasetBuffer.init_buffer()."
    scaled_rewards = []
    for reward in buffer.dataset['rewards']:
        scaled_rewards.append(reward * scale)

    buffer.dataset['rewards'] = scaled_rewards

    return buffer, {'TransformReward': {'func': lambda reward: reward * scale}}
