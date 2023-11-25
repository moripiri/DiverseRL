from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, soft_update
from diverserl.networks import GaussianActor, QNetwork


class SACv2(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
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
        """
        SAC (Soft Actor-Critic)

        Paper: Soft Actor-Critic Algorithm and Applications, Haarnoja et al, 2018

        :param observation_space: The observation space of the environment.
        :param action_space: The action space of the environment.
        :param gamma: The discount factor.
        :param alpha: The entropy temperature parameter.
        :param train_alpha: Whether to train the parameter alpha.
        :param target_alpha: Target entropy value (usually set as -|action_dim|).
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param critic_update: Critic will only be updated once for every critic_update steps.
        :param batch_size: Minibatch size for optimizer.
        :param buffer_size: Maximum length of replay buffer.
        :param actor_lr: Learning rate for actor.
        :param actor_optimizer: Optimizer class (or name) for actor.
        :param actor_optimizer_kwargs: Parameter dict for actor optimizer.
        :param critic_lr: Learning rate of the critic
        :param critic_optimizer: Optimizer class (or str) for the critic
        :param critic_optimizer_kwargs: Parameter dict for the critic optimizer
        :param alpha_lr: Learning rate for alpha.
        :param alpha_optimizer: Optimizer class (or str) for the alpha
        :param alpha_optimizer_kwargs: Parameter dict for the alpha optimizer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__()

        assert isinstance(observation_space, spaces.Box) and isinstance(
            action_space, spaces.Box
        ), f"{self} supports only Box type observation space and action space."

        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.action_scale = (action_space.high[0] - action_space.low[0]) / 2
        self.action_bias = (action_space.high[0] + action_space.low[0]) / 2

        self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=train_alpha)
        self.target_alpha = -self.action_dim if target_alpha is None else target_alpha
        self.train_alpha = train_alpha

        self.critic_update = critic_update
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        self.actor = GaussianActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            output_scale=self.action_scale,
            output_bias=self.action_bias,
            device=device,
        ).train()

        self.critic = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim, device=device).train()
        self.critic2 = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim, device=device).train()

        self.target_critic = deepcopy(self.critic).eval()
        self.target_critic2 = deepcopy(self.critic2).eval()

        actor_optimizer, actor_optimizer_kwargs = get_optimizer(actor_optimizer, actor_optimizer_kwargs)
        critic_optimizer, critic_optimizer_kwargs = get_optimizer(critic_optimizer, critic_optimizer_kwargs)

        self.actor_optimizer = actor_optimizer(self.actor.parameters(), lr=actor_lr, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), lr=critic_lr, **critic_optimizer_kwargs)
        self.critic2_optimizer = critic_optimizer(self.critic2.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        if self.train_alpha:
            alpha_optimizer, alpha_optimizer_kwargs = get_optimizer(alpha_optimizer, alpha_optimizer_kwargs)
            self.alpha_optimizer = alpha_optimizer([self.log_alpha], lr=alpha_lr, **alpha_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def __repr__(self):
        return "SACv2"

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the SACv2 action from an observation (in training mode)

        :param observation: The input observation
        :return: The SACv2 agent's action
        """
        observation = super()._fix_ob_shape(observation)

        self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the SACv2 action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The SACv2 agent's action (in evaluation mode)
        """
        observation = super()._fix_ob_shape(observation)

        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.numpy()[0]

    @property
    def alpha(self):
        """
        :return: Return current alpha value as float
        """
        return self.log_alpha.exp().detach()

    def train(self) -> Dict[str, Any]:
        """
        Train SACv2 policy.
        :return: training results
        """
        self.training_step += 1
        result_dict = {}
        self.actor.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        # critic network training
        with torch.no_grad():
            ns_action, ns_logprob = self.actor(ns)

            target_min_aq = torch.minimum(self.target_critic((ns, ns_action)), self.target_critic2((ns, ns_action)))

            target_q = r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logprob)

        critic_loss = F.mse_loss(self.critic((s, a)), target_q)
        critic2_loss = F.mse_loss(self.critic2((s, a)), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # actor training
        s_action, s_logprob = self.actor(s)
        min_aq_rep = torch.minimum(self.critic((s, s_action)), self.critic2((s, s_action)))
        actor_loss = (self.alpha * s_logprob - min_aq_rep).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha training
        if self.train_alpha:
            alpha_loss = -(self.log_alpha.exp() * (s_logprob + self.target_alpha).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            result_dict["alpha_loss"] = alpha_loss.detach().cpu().numpy()

        # critic update
        if self.training_step % self.critic_update == 0:
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

        result_dict["actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict["critic_loss"] = critic_loss.detach().cpu().numpy()
        result_dict["critic2_loss"] = critic2_loss.detach().cpu().numpy()

        return result_dict
