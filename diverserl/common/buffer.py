import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_size=10**6,
        save_log_prob=False,
        device="cpu",
    ) -> None:
        """
        Buffer to record and save the RL agent trajectories.

        :param state_dim: Length of the state
        :param action_dim: Length of the action
        :param max_size: Maximum length of the ReplayBuffer
        """
        self.device = device
        self.max_size = max_size

        self.s_size = (self.max_size, state_dim) if isinstance(state_dim, int) else (self.max_size, *state_dim)
        self.a_size = (self.max_size, action_dim)

        self.save_log_prob = save_log_prob

        self.s = np.empty(self.s_size, dtype=np.float32)
        self.a = np.empty(self.a_size, dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.ns = np.empty(self.s_size, dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)
        self.t = np.empty((self.max_size, 1), dtype=np.float32)

        if save_log_prob:
            self.log_prob = np.empty((self.max_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.idx

    def add(
        self,
        s: np.ndarray,
        a: Union[int, np.ndarray],
        r: float,
        ns: np.ndarray,
        d: bool,
        t: bool,
        log_prob: Optional[float] = None,
    ) -> None:
        """
        Add the one-step result to the buffer.

        :param s: state
        :param a: action
        :param r: reward
        :param ns: next state
        :param d: done
        :param t: truncated
        :param log_prob: log_probability
        """

        np.copyto(self.s[self.idx], s)
        np.copyto(self.a[self.idx], a)
        np.copyto(self.r[self.idx], r)
        np.copyto(self.ns[self.idx], ns)
        np.copyto(self.d[self.idx], d)
        np.copyto(self.t[self.idx], t)

        if self.save_log_prob:
            np.copyto(self.log_prob[self.idx], log_prob)

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    @property
    def size(self) -> int:
        return self.max_size if self.full else self.idx

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Randomly sample wanted size of mini batch from buffer.

        :param batch_size: Length of the mini-batch
        :return: Sampled mini batch
        """
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = torch.from_numpy(self.s[ids]).to(self.device)
        actions = torch.from_numpy(self.a[ids]).to(self.device)
        rewards = torch.from_numpy(self.r[ids]).to(self.device)
        next_states = torch.from_numpy(self.ns[ids]).to(self.device)
        dones = torch.from_numpy(self.d[ids]).to(self.device)
        terminates = torch.from_numpy(self.t[ids]).to(self.device)

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[ids]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs

        return states, actions, rewards, next_states, dones, terminates

    def delete(self):
        self.s = np.empty(self.s_size, dtype=np.float32)
        self.a = np.empty(self.a_size, dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.ns = np.empty(self.s_size, dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)
        self.t = np.empty((self.max_size, 1), dtype=np.float32)

        if self.save_log_prob:
            self.log_prob = np.empty((self.max_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
