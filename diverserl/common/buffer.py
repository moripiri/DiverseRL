import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, max_size: int = 10 ** 6,
                 save_log_prob: bool = False, num_envs: int = 1, device: str = "cpu") -> None:

        # super().__init__(state_dim, action_dim, max_size, save_log_prob, device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_size = max_size
        self.device = device

        self.save_log_prob = save_log_prob
        self.num_envs = num_envs

        self.s_size = (self.max_size, num_envs, state_dim) if isinstance(state_dim, int) else (
            self.max_size, num_envs, *state_dim)

        self.a_size = (self.max_size, num_envs, action_dim)

        self.reset()

    def __len__(self) -> int:
        return self.idx

    @property
    def size(self) -> int:
        return self.max_size if self.full else self.idx

    def reset(self):
        self.s = np.empty(self.s_size, dtype=np.float32)
        self.a = np.empty(self.a_size, dtype=np.float32)
        self.r = np.empty((self.max_size, self.num_envs, 1), dtype=np.float32)
        self.ns = np.empty(self.s_size, dtype=np.float32)
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

        if isinstance(self.state_dim, tuple):
            s = s.reshape((self.num_envs, *self.state_dim))
            ns = ns.reshape((self.num_envs, *self.state_dim))
        else:
            s = s.reshape((self.num_envs, self.state_dim))
            ns = ns.reshape((self.num_envs, self.state_dim))

        self.s[self.idx] = s
        self.a[self.idx] = a.reshape((self.num_envs, self.action_dim))
        self.r[self.idx] = r.reshape((self.num_envs, 1))
        self.ns[self.idx] = ns
        self.d[self.idx] = d.reshape((self.num_envs, 1))
        self.t[self.idx] = t.reshape((self.num_envs, 1))

        if self.save_log_prob:

            log_prob = log_prob.reshape((self.num_envs, 1))
            self.log_prob[self.idx] = log_prob

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        batch_ids = np.random.randint(0, self.size, size=batch_size)
        env_ids = np.random.randint(0, high=self.num_envs, size=(batch_size,))

        states = torch.from_numpy(self.s[batch_ids, env_ids, :]).to(self.device)
        actions = torch.from_numpy(self.a[batch_ids, env_ids, :]).to(self.device)
        rewards = torch.from_numpy(self.r[batch_ids, env_ids, :]).to(self.device)
        next_states = torch.from_numpy(self.ns[batch_ids, env_ids, :]).to(self.device)
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
        next_states = torch.from_numpy(self.ns[ids]).to(self.device)
        dones = torch.from_numpy(self.d[ids]).to(self.device)
        terminates = torch.from_numpy(self.t[ids]).to(self.device)

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[ids]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs

        return states, actions, rewards, next_states, dones, terminates
