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
from diverserl.networks import MLP, GaussianActor


class SACv2(DeepRL):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.1,
        train_alpha: bool = True,
        target_alpha: Optional[float] = None,
        tau: float = 0.05,
        critic_update: int = 2,
        batch_size: int = 256,
        buffer_size: int = 10**6,
        actor_lr: float = 0.001,
        actor_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_lr: float = 0.001,
        critic_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        alpha_lr: float = 0.001,
        alpha_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        alpha_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)

        assert isinstance(env.observation_space, spaces.Box) and isinstance(
            env.action_space, spaces.Box
        ), f"{self} supports only Box type observation space and action space."

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_scale = (env.action_space.high[0] - env.action_space.low[0]) / 2
        self.action_bias = (env.action_space.high[0] + env.action_space.low[0]) / 2

        self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=train_alpha)
        self.target_alpha = -self.action_dim if target_alpha is None else target_alpha
        self.train_alpha = train_alpha

        self.critic_update = critic_update
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            output_scale=self.action_scale,
            output_bias=self.action_bias,
            device=self.device,
        ).train()
        self.critic = MLP(self.state_dim + self.action_dim, 1, device=self.device).train()
        self.critic2 = MLP(self.state_dim + self.action_dim, 1, device=self.device).train()

        self.target_critic = deepcopy(self.critic).eval()
        self.target_critic2 = deepcopy(self.critic2).eval()

        actor_optimizer = getattr(torch.optim, actor_optimizer) if isinstance(actor_optimizer, str) else actor_optimizer
        if actor_optimizer_kwargs is None:
            actor_optimizer_kwargs = {}

        critic_optimizer = (
            getattr(torch.optim, critic_optimizer) if isinstance(critic_optimizer, str) else critic_optimizer
        )
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {}

        if self.train_alpha:
            alpha_optimizer = (
                getattr(torch.optim, alpha_optimizer) if isinstance(alpha_optimizer, str) else alpha_optimizer
            )
            if alpha_optimizer_kwargs is None:
                alpha_optimizer_kwargs = {}

            self.alpha_optimizer = alpha_optimizer([self.log_alpha], lr=alpha_lr, **alpha_optimizer_kwargs)

        self.actor_optimizer = actor_optimizer(self.actor.parameters(), lr=actor_lr, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), lr=critic_lr, **critic_optimizer_kwargs)
        self.critic2_optimizer = critic_optimizer(self.critic2.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.training_step = 0

    def __repr__(self):
        return "SACv2"

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        observation = super()._fix_ob_shape(observation)

        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.numpy()[0]

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def train(self) -> Dict[str, Any]:
        self.training_step += 1
        self.actor.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        # critic network training
        with torch.no_grad():
            ns_action, ns_logpi = self.actor(ns)

            target_min_aq = torch.minimum(self.target_critic((ns, ns_action)), self.target_critic2((ns, ns_action)))

            target_q = r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi)

        critic_loss = F.mse_loss(self.critic((s, a)), target_q)
        critic2_loss = F.mse_loss(self.critic2((s, a)), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # actor training
        s_action, s_logpi = self.actor(s)
        min_aq_rep = torch.minimum(self.critic((s, s_action)), self.critic2((s, s_action)))
        actor_loss = (self.alpha * s_logpi - min_aq_rep).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.train_alpha:
            alpha_loss = -(self.log_alpha.exp() * (s_logpi + self.target_alpha).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        if self.training_step % self.critic_update == 0:
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

        return {}
