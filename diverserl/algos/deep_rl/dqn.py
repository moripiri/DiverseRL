import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.networks import MLP


class DQN(DeepRL):
    def __init__(
        self,
        env: gym.Env,
        eps: float = 0.05,
        gamma: float = 0.9,
        batch_size: int = 256,
        buffer_size: int = 10**6,
        learning_rate: float = 0.001,
        target_copy_freq: int = 5,
        device: str = "cpu",
    ):
        super().__init__(device)

        assert isinstance(env.observation_space, spaces.Box) and isinstance(
            env.action_space, spaces.Discrete
        ), f"{self} supports only Box type observation space and Discrete type action space."

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.q_network = MLP(self.state_dim, self.action_dim, device=device).train()
        self.target_q_network = MLP(self.state_dim, self.action_dim, device=device).eval()

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.buffer = ReplayBuffer(self.state_dim, 1, max_size=buffer_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.training_step = 0

        self.eps = eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_copy_freq = target_copy_freq

    def __repr__(self):
        return "DQN"

    def get_action(self, observation: np.ndarray | torch.Tensor) -> int | list[int]:
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        else:
            observation = super()._fix_ob_shape(observation)

            self.q_network.train()
            with torch.no_grad():
                return self.q_network(observation).argmax(1).numpy()[0]

    def eval_action(self, observation: np.ndarray | torch.Tensor) -> int | list[int]:
        observation = super()._fix_ob_shape(observation)
        self.q_network.eval()

        with torch.no_grad():
            return self.q_network(observation).argmax(1).numpy()[0]

    def train(self):
        self.training_step += 1
        self.q_network.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            target_value = r + self.gamma * (1 - d) * self.target_q_network(ns).max(1, keepdims=True)[0]

        selected_value = self.q_network(s).gather(1, a.to(torch.int64))

        train_loss = F.huber_loss(selected_value, target_value)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        if self.training_step % self.target_copy_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {"train_loss": train_loss.detach().numpy()}
