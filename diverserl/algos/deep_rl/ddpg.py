from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import soft_update
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
        actor_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_lr: float = 0.001,
        critic_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device="cpu",
    ) -> None:
        """
        DDPG(Deep Deterministic Policy Gradients)

        Paper: Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.

        :param env: The environment for RL agent to learn from
        :param gamma: The discount factor
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param noise_scale: Stddev for Gaussian noise added to policy action at training time.
        :param batch_size: Minibatch size for optimizer.
        :param buffer_size: Maximum length of replay buffer.
        :param actor_lr: Learning rate for actor.
        :param actor_optimizer: Optimizer class (or name) for actor.
        :param actor_optimizer_kwargs: Parameter dict for actor optimizer.
        :param critic_lr: Learning rate of the critic
        :param critic_optimizer: Optimizer class (or str) for the critic
        :param critic_optimizer_kwargs: Parameter dict for the critic optimizer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
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
        self.target_actor = deepcopy(self.actor).eval()

        self.critic = MLP(self.state_dim + self.action_dim, 1, device=device).train()
        self.target_critic = deepcopy(self.critic).eval()

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        actor_optimizer = getattr(torch.optim, actor_optimizer) if isinstance(actor_optimizer, str) else actor_optimizer
        if actor_optimizer_kwargs is None:
            actor_optimizer_kwargs = {}

        critic_optimizer = (
            getattr(torch.optim, critic_optimizer) if isinstance(critic_optimizer, str) else critic_optimizer
        )
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {}

        self.actor_optimizer = actor_optimizer(self.actor.parameters(), lr=actor_lr, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale

        self.training_step = 0

    def __repr__(self):
        return "DDPG"

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the DDPG action from an observation (in training mode)

        :param observation: The input observation
        :return: The DDPG agent's action
        """
        observation = super()._fix_ob_shape(observation)

        self.actor.train()
        with torch.no_grad():
            action = self.actor(observation).numpy()[0]
            noise = np.random.normal(loc=0, scale=self.noise_scale, size=self.action_dim)

        return np.clip(action + noise, -self.action_scale + self.action_bias, self.action_scale + self.action_bias)

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the DDPG action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The DDPG agent's action (in evaluation mode)
        """
        observation = super()._fix_ob_shape(observation)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(observation).numpy()[0]

        return action

    def train(self) -> Dict[str, Any]:
        """
        Train the DDPG policy.

        :return: Training result (actor_loss, critic_loss)
        """
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

        soft_update(self.actor, self.target_actor, self.tau)
        soft_update(self.critic, self.target_critic, self.tau)

        return {"actor_loss": actor_loss.detach().numpy(), "critic_loss": critic_loss.detach().numpy()}
