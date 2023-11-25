from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import torch
import torch.nn.functional as F

from diverserl.algos.deep_rl import DDPG
from diverserl.common.utils import soft_update
from diverserl.networks.basic_networks import MLP


class TD3(DDPG):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        tau: float = 0.05,
        noise_scale: float = 0.1,
        target_noise_scale: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        batch_size: int = 256,
        buffer_size: int = 10**6,
        actor_lr: float = 0.001,
        actor_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_lr: float = 0.001,
        critic_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """
        TD3(Twin Delayed Deep Deterministic Policy Gradients)

        Paper: Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018

        :param env: The environment for RL agent to learn from
        :param gamma: Discount factor.
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param noise_scale: Stddev for Gaussian noise added to policy action at training time.
        :param target_noise_scale: Stddev for smoothing noise added to target policy action.
        :param noise_clip: Limit for absolute value of target policy action noise.
        :param policy_delay: Policy will only be updated once every policy_delay times for each update of the critics.
        :param batch_size: Minibatch size for optimizer.
        :param buffer_size: Maximum length of replay buffer.
        :param actor_lr: Learning rate for actor.
        :param actor_optimizer: Optimizer class (or name) for actor.
        :param actor_optimizer_kwargs: Parameter dict for actor optimizer.
        :param critic_lr: Learning rate for critics.
        :param critic_optimizer: Optimizer class (or name) for critics.
        :param critic_optimizer_kwargs: Parameter dict for critic optimizer.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """

        super().__init__(
            env,
            gamma,
            tau,
            noise_scale,
            batch_size,
            buffer_size,
            actor_lr,
            actor_optimizer,
            actor_optimizer_kwargs,
            critic_lr,
            critic_optimizer,
            critic_optimizer_kwargs,
            device,
        )

        self.critic2 = MLP(self.state_dim + self.action_dim, 1, device=device).train()
        self.target_critic2 = deepcopy(self.critic2).eval()

        critic2_optimizer = (
            getattr(torch.optim, critic_optimizer) if isinstance(critic_optimizer, str) else critic_optimizer
        )
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {}

        self.critic2_optimizer = critic2_optimizer(self.critic2.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.target_noise_scale = target_noise_scale
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip

    def __repr__(self) -> str:
        return "TD3"

    def train(self) -> Dict[str, Any]:
        """
        Train the TD3 policy.

        :return: Training result (actor_loss, critic_loss, critic2_loss)
        """
        self.training_step += 1
        self.actor.train()

        result_dict = dict()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_action = (
                self.target_actor(ns)
                + torch.normal(0, self.target_noise_scale, (self.batch_size, self.action_dim)).clamp(
                    -self.noise_clip, self.noise_clip
                )
            ).clamp(-self.action_scale + self.action_bias, self.action_scale + self.action_bias)
            target_value = r + self.gamma * (1 - d) * torch.minimum(
                self.target_critic((ns, target_action)), self.target_critic2((ns, target_action))
            )

        critic_loss = F.mse_loss(self.critic((s, a)), target_value)
        critic2_loss = F.mse_loss(self.critic2((s, a)), target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.training_step % self.policy_delay == 0:
            actor_loss = -self.critic((s, self.actor(s))).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

            result_dict["actor_loss"] = actor_loss.detach().numpy()
        result_dict["critic_loss"] = critic_loss.detach().numpy()
        result_dict["critic2_loss"] = critic2_loss.detach().numpy()

        return result_dict
