from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from minari import MinariDataset
from torch import Tensor

from diverserl.common.utils import find_action_space, find_observation_space


class ReplayBuffer:
    """
    Simple Buffer to store environment transitions and trajectories for Deep RL training.
    """

    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, max_size: int = 10 ** 6,
                 save_log_prob: bool = False, optimize_memory_usage: bool = False, num_envs: int = 1,
                 device: str = "cpu") -> None:
        """
        Initialize buffer with given dimensions and settings.

        :param state_dim: state dimension of the environment
        :param action_dim: action dimension of the environment
        :param max_size: max size of the ReplayBuffer
        :param save_log_prob: Whether to save the log probabilities of the trajectories
        :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer(source: https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274)
        :param num_envs: Number of the parallel environments
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_size = max_size
        self.device = device

        self.save_log_prob = save_log_prob
        self.optimize_memory_usage = optimize_memory_usage
        self.num_envs = num_envs

        if self.optimize_memory_usage:
            max_s_size = self.max_size + 1
        else:
            max_s_size = self.max_size

        self.s_size = (max_s_size, num_envs, state_dim) if isinstance(state_dim, int) else (
            max_s_size, num_envs, *state_dim)

        self.s_dtype = np.float32 if isinstance(state_dim, int) else np.uint8

        self.a_size = (self.max_size, num_envs, action_dim)

        self.reset()

    def __len__(self) -> int:
        return self.idx

    @property
    def size(self) -> int:
        """
        Size of the stored transitions. Not equal to the buffer's length.
        :return: Current ReplayBuffer size.
        """
        return self.max_size if self.full else self.idx

    def reset(self):
        """
        Resets the ReplayBuffer to the initial state.
        """
        self.s = np.empty(self.s_size, dtype=self.s_dtype)
        self.a = np.empty(self.a_size, dtype=np.float32)
        self.r = np.empty((self.max_size, self.num_envs, 1), dtype=np.float32)

        if not self.optimize_memory_usage:
            self.ns = np.empty(self.s_size, dtype=self.s_dtype)

        self.d = np.empty((self.max_size, self.num_envs, 1), dtype=np.float32)
        self.t = np.empty((self.max_size, self.num_envs, 1), dtype=np.float32)

        if self.save_log_prob:
            self.log_prob = np.empty((self.max_size, self.num_envs, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            ns: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
            log_prob: Optional[np.ndarray] = None,
            ) -> None:
        """
        Add a single transition to the ReplayBuffer.

        :param s: state
        :param a: action
        :param r: reward
        :param ns: next state
        :param d: terminated
        :param t: truncated
        :param log_prob: log probability of the action
        """
        if isinstance(self.state_dim, tuple):
            s = s.reshape((self.num_envs, *self.state_dim))
            ns = ns.reshape((self.num_envs, *self.state_dim))
        else:
            s = s.reshape((self.num_envs, self.state_dim))
            ns = ns.reshape((self.num_envs, self.state_dim))

        self.s[self.idx] = s
        self.a[self.idx] = a.reshape((self.num_envs, self.action_dim))
        self.r[self.idx] = r.reshape((self.num_envs, 1))

        if not self.optimize_memory_usage:
            self.ns[self.idx] = ns
        else:
            self.s[self.idx + 1] = ns

        self.d[self.idx] = d.reshape((self.num_envs, 1))
        self.t[self.idx] = t.reshape((self.num_envs, 1))

        if self.save_log_prob:
            log_prob = log_prob.reshape((self.num_envs, 1))
            self.log_prob[self.idx] = log_prob

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Randomly sample a batch of transitions.
        :param batch_size: Size of the sampled batch.

        :return: Sampled states, actions, rewards, next states, terminateds, truncateds, (log_probabilities)
        """
        if self.optimize_memory_usage and self.full:
            # don't sample idx in batch_ids
            batch_ids = (np.random.randint(1, self.max_size, size=batch_size) + self.idx) % self.max_size
        else:
            batch_ids = np.random.randint(0, self.size, size=batch_size)

        env_ids = np.random.randint(0, high=self.num_envs, size=(batch_size,))

        states = torch.from_numpy(self.s[batch_ids, env_ids, :]).to(self.device)
        actions = torch.from_numpy(self.a[batch_ids, env_ids, :]).to(self.device)
        rewards = torch.from_numpy(self.r[batch_ids, env_ids, :]).to(self.device)
        if not self.optimize_memory_usage:
            next_states = torch.from_numpy(self.ns[batch_ids, env_ids, :]).to(self.device)
        else:
            # for trajectory sampling, it's not (batch_ids + 1) % self.max_size
            next_states = torch.from_numpy(self.s[batch_ids + 1, env_ids, :]).to(self.device)

        dones = torch.from_numpy(self.d[batch_ids, env_ids, :]).to(self.device)
        terminates = torch.from_numpy(self.t[batch_ids, env_ids, :]).to(self.device)

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[batch_ids, env_ids, :]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs

        return states, actions, rewards, next_states, dones, terminates

    def all_sample(self) -> Tuple[Tensor, ...]:
        """
        Return all records from buffer.

        :return: Tuples of RL records
        """
        ids = np.arange(self.size)

        states = torch.from_numpy(self.s[ids]).to(self.device)
        actions = torch.from_numpy(self.a[ids]).to(self.device)
        rewards = torch.from_numpy(self.r[ids]).to(self.device)

        if not self.optimize_memory_usage:
            next_states = torch.from_numpy(self.ns[ids]).to(self.device)
        else:
            next_states = torch.from_numpy(self.s[ids + 1]).to(self.device)

        dones = torch.from_numpy(self.d[ids]).to(self.device)
        terminates = torch.from_numpy(self.t[ids]).to(self.device)

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[ids]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs

        return states, actions, rewards, next_states, dones, terminates


class NstepReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, max_size: int = 10 ** 6,
                 n_step: int = 3, discount: float = 0.99, save_log_prob: bool = False,
                 optimize_memory_usage: bool = False, num_envs: int = 1,
                 device: str = "cpu") -> None:
        super().__init__(state_dim, action_dim, max_size, save_log_prob, optimize_memory_usage, num_envs, device)

        self.discount = discount
        self.n_step = n_step

    def __len__(self) -> int:
        return self.idx - self.n_step

    @property
    def size(self) -> int:
        """
        Size of the stored transitions. Not equal to the buffer's length.
        :return: Current ReplayBuffer size.
        """
        return (self.max_size if self.full else self.idx) - self.n_step

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Randomly sample a batch of transitions.
       :param batch_size: Size of the sampled batch.

       :return: Sampled states, actions, rewards, next states, terminateds, truncateds, (log_probabilities)
       """
        if self.optimize_memory_usage and self.full:
            # don't sample idx in batch_ids
            batch_ids = (np.random.randint(1, self.max_size, size=batch_size) + self.idx) % self.max_size
        else:
            batch_ids = np.random.randint(0, self.size, size=batch_size)

        env_ids = np.random.randint(0, high=self.num_envs, size=(batch_size,))

        states = torch.from_numpy(self.s[batch_ids, env_ids, :]).to(self.device)
        actions = torch.from_numpy(self.a[batch_ids, env_ids, :]).to(self.device)

        if not self.optimize_memory_usage:
            next_states = torch.from_numpy(self.ns[batch_ids + self.n_step, env_ids, :]).to(self.device)
        else:
            # for trajectory sampling, it's not (batch_ids + 1) % self.max_size
            next_states = torch.from_numpy(self.s[batch_ids + 1 + self.n_step, env_ids, :]).to(self.device)

        dones = torch.from_numpy(self.d[batch_ids, env_ids, :]).to(self.device)
        terminates = torch.from_numpy(self.t[batch_ids, env_ids, :]).to(self.device)

        rewards = torch.zeros_like(torch.from_numpy(self.r[batch_ids, env_ids, :]).to(self.device))
        discounts = torch.ones_like(rewards)

        for i in range(self.n_step):
            step_reward = torch.from_numpy(self.r[batch_ids + i, env_ids, :]).to(self.device)
            rewards += discounts * step_reward

            step_dones = torch.from_numpy(self.d[batch_ids + i, env_ids, :]).to(self.device)
            step_terminates = torch.from_numpy(self.t[batch_ids + i, env_ids, :]).to(self.device)

            masks = torch.logical_or(step_dones, step_terminates).to(dtype=torch.float32)

            discounts *= (1 - step_dones) * self.discount

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[batch_ids, env_ids, :]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs, discounts

        return states, actions, rewards, next_states, dones, terminates, discounts


class DatasetBuffer:
    def __init__(self, dataset: MinariDataset, device: str = "cpu") -> None:
        self.state_dim = find_observation_space(dataset.spec.observation_space)
        self.action_dim, _, _, _ = find_action_space(dataset.spec.action_space)

        self.dataset = self.load_dataset(dataset)
        self.dataset_metadata = dataset.storage.metadata
        self.ref_min_score, self.ref_max_score = self.dataset_metadata.get('ref_min_score'), self.dataset_metadata.get('ref_max_score')
        self.device = device

    def __len__(self) -> int:
        return len(self.s)

    @property
    def size(self) -> int:
        """
        Size of the stored transitions. Not equal to the buffer's length.
        :return: Current ReplayBuffer size.
        """
        return len(self.s)

    def load_dataset(self, dataset: MinariDataset) -> Dict[str, List[Any]]:
        temp_dataset = {'id': [], 'observations': [], 'actions': [], 'rewards': [], 'terminations': [], 'truncations': [], 'infos': [] }
        for episode in dataset:
            temp_dataset['id'].append(episode.id)
            temp_dataset['observations'].append(episode.observations)
            temp_dataset['actions'].append(episode.actions)
            temp_dataset['rewards'].append(episode.rewards.reshape(-1, 1))
            temp_dataset['terminations'].append(episode.terminations.reshape(-1, 1))
            temp_dataset['truncations'].append(episode.truncations.reshape(-1, 1))
            temp_dataset['infos'].append(episode.infos)

        return temp_dataset

    def filter_episodes(self, ids: List[int]) -> None:
        dataset_ids = [self.dataset['id'].index(id) for id in ids]
        for key in self.dataset.keys():
            self.dataset[key] = [item for i, item in enumerate(self.dataset[key]) if i not in [dataset_ids]]


    def init_buffer(self):
        print(self.dataset['observations'])
        self.s = np.concatenate(list(map(lambda x: x[:-1], self.dataset['observations'])), axis=0, dtype=np.float32)
        self.a = np.concatenate(self.dataset['actions'], axis=0, dtype=np.float32)
        self.r = np.concatenate(self.dataset['rewards'], axis=0, dtype=np.float32)
        self.ns = np.concatenate(list(map(lambda x: x[1:], self.dataset['observations'])), axis=0, dtype=np.float32)
        self.d = np.concatenate(self.dataset['terminations'], axis=0, dtype=np.float32)
        self.t = np.concatenate(self.dataset['truncations'], axis=0, dtype=np.float32)

        del self.dataset

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        batch_ids = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.s[batch_ids,:]).to(self.device)
        actions = torch.from_numpy(self.a[batch_ids,:]).to(self.device)
        rewards = torch.from_numpy(self.r[batch_ids,:]).to(self.device)
        next_states = torch.from_numpy(self.ns[batch_ids,:]).to(self.device)
        dones = torch.from_numpy(self.d[batch_ids,:]).to(self.device)
        terminates = torch.from_numpy(self.t[batch_ids,:]).to(self.device)

        return states, actions, rewards, next_states, dones, terminates


class SequenceDatasetBuffer(DatasetBuffer):
    def __init__(self, dataset: MinariDataset, sequence_length: int = 10, gamma: float = 0.99, device: str = "cpu") -> None:
        super().__init__(dataset, device)
        self.sequence_length = sequence_length
        self.gamma = gamma

    def __len__(self) -> int:
        return len(self.s)

    def init_buffer(self):
        self.sample_prob = np.array([len(obs) for obs in self.dataset['observations']], dtype=np.float64)
        self.sample_prob /= sum(self.sample_prob)
        self.s = self.dataset['observations']
        self.a = self.dataset['actions']
        self.r = list(map(lambda x: self.discounted_cumsum(x, self.gamma), self.dataset['rewards']))
        self.d = self.dataset['terminations']
        self.t = self.dataset['truncations']

        del self.dataset

    def discounted_cumsum(self, x: np.ndarray, gamma: float) -> np.ndarray:
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            cumsum[t] = x[t] + gamma * cumsum[t + 1]
        return cumsum

    def pad_along_axis(
            self, arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0) -> np.ndarray:
        pad_size = pad_to - arr.shape[axis]
        if pad_size <= 0:
            return arr

        npad = [(0, 0)] * arr.ndim
        npad[axis] = (0, pad_size)
        return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:

        traj_idx = np.random.choice(len(self), size=batch_size, p=self.sample_prob)
        start_idx = np.random.randint(0, list(map(lambda x: self.r[x].shape[0] - 1, traj_idx)) , batch_size)

        states = np.asarray(list(map(lambda x, y: self.pad_along_axis(self.s[x][y:y + self.sequence_length], self.sequence_length), traj_idx, start_idx)))
        actions = np.asarray(list(map(lambda x, y: self.pad_along_axis(self.a[x][y:y + self.sequence_length], self.sequence_length), traj_idx, start_idx)))
        returns = np.asarray(list(map(lambda x, y: self.pad_along_axis(self.r[x][y:y + self.sequence_length], self.sequence_length), traj_idx, start_idx)))
        time_steps = np.asarray(list(map(lambda x: np.arange(x, x + self.sequence_length), start_idx)))

        states_lengths = list(map(lambda x, y: len(self.s[x][y:y + self.sequence_length]), traj_idx, start_idx))

        mask = np.vstack(list(map(lambda x: np.hstack([np.ones(x), np.zeros(self.sequence_length - x)]), states_lengths)))

        states = torch.from_numpy(states).to(self.device, torch.float32)
        actions = torch.from_numpy(actions).to(self.device, torch.float32)
        returns = torch.from_numpy(returns).to(self.device, torch.float32)
        time_steps = torch.from_numpy(time_steps).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        return states, actions, returns, time_steps, mask
