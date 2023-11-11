import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.networks.basic_networks import MLP


class DDPG(DeepRL):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        tau: float = 0.05,
        noise_scale: float = 0.1,
        batch_size: int = 256,
        buffer_size: int = 10**6,
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        device="cpu",
    ):
        super().__init__(device)

        assert isinstance(env.observation_space, spaces.Box) and isinstance(
            env.action_space, spaces.Box
        ), f"{self} supports only Box type observation space and action space."

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_scale = (env.action_space.high[0] - env.action_space.low[0]) / 2
        self.action_bias = (env.action_space.high[0] + env.action_space.low[0]) / 2

        self.actor = MLP(
            self.state_dim,
            self.action_dim,
            last_activation="Tanh",
            output_scale=self.action_scale,
            output_bias=self.action_bias,
            device=device,
        ).train()
        self.target_actor = MLP(
            self.state_dim,
            self.action_dim,
            last_activation="Tanh",
            output_scale=self.action_scale,
            output_bias=self.action_bias,
            device=device,
        ).eval()
        self.critic = MLP(self.state_dim + self.action_dim, 1, device=device).train()
        self.target_critic = MLP(self.state_dim + self.action_dim, 1, device=device).eval()

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale

        self.training_step = 0

    def __repr__(self):
        return "DDPG"

    def get_action(self, observation: np.ndarray | torch.Tensor) -> list[float]:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()
        with torch.no_grad():
            action = self.actor(observation).numpy()[0]
            noise = np.random.normal(loc=0, scale=self.noise_scale, size=self.action_dim)

        return np.clip(action + noise, -self.action_scale + self.action_bias, self.action_scale + self.action_bias)

    def eval_action(self, observation: np.ndarray | torch.Tensor) -> list[float]:
        observation = super()._fix_ob_shape(observation)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(observation).numpy()[0]

        return action

    def train(self) -> dict:
        self.training_step += 1
        self.actor.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_value = r + self.gamma * (1 - d) * self.target_critic((ns, self.target_actor(ns)))

        critic_loss = F.mse_loss(self.critic((s, a)), target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic((s, self.actor(s))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"actor_loss": actor_loss.detach().numpy(), "critic_loss": critic_loss.detach().numpy()}
