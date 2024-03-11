import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from diverserl.misc.sumtree import SumTree


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
        self.real_size = 0
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
        self.real_size = self.max_size if self.full else self.idx

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

    def all_sample(self) -> Tuple[Tensor, ...]:
        """
        Return all records from buffer.

        :return: Tuples of RL records
        """
        ids = np.arange(self.max_size if self.full else self.idx)

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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 state_dim: Union[int, Tuple[int, ...]],
                 action_dim: int,
                 max_size: int = 10 ** 6,
                 eps: float = 0.01,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 device="cpu",
                 ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, max_size=max_size, save_log_prob=False,
                         device=device)

        self.tree = SumTree(size=max_size)

        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def sample(self, batch_size) -> Tuple[Any, ...]:
        sample_ids, tree_ids = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_ids.append(tree_idx)
            sample_ids.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        states = torch.from_numpy(self.s[sample_ids]).to(self.device)
        actions = torch.from_numpy(self.a[sample_ids]).to(self.device)
        rewards = torch.from_numpy(self.r[sample_ids]).to(self.device)
        next_states = torch.from_numpy(self.ns[sample_ids]).to(self.device)
        dones = torch.from_numpy(self.d[sample_ids]).to(self.device)
        terminates = torch.from_numpy(self.t[sample_ids]).to(self.device)

        if self.save_log_prob:
            log_probs = torch.from_numpy(self.log_prob[sample_ids]).to(self.device)

            return states, actions, rewards, next_states, dones, terminates, log_probs, weights, tree_ids

        return states, actions, rewards, next_states, dones, terminates, weights, tree_ids

    def add(self,
        s: np.ndarray,
        a: Union[int, np.ndarray],
        r: float,
        ns: np.ndarray,
        d: bool,
        t: bool,
        log_prob: Optional[float] = None,
    ) -> None:
        self.tree.add(self.max_priority, self.idx)

        super().add(s, a, r, ns, d, t, log_prob)

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
