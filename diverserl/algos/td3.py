from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, soft_update
from diverserl.networks.basic_networks import DeterministicActor, QNetwork


class TD3(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "Default",
        network_config: Optional[Dict[str, Any]] = None,
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
            **kwargs: Optional[Dict[str, Any]]

    ) -> None:
        """
        TD3(Twin Delayed Deep Deterministic Policy Gradients)

        Paper: Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018

        :param observation_space: Observation space of the environment for RL agent to learn from
        :param action_space: Action space of the environment for RL agent to learn from
        :param network_type: Type of the TD3 networks to be used.
        :param network_config: Configurations of the TD3 networks.
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
            network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )

        assert isinstance(observation_space, spaces.Box) and isinstance(
            action_space, spaces.Box
        ), f"{self} supports only Box type observation space and action space."

        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.action_scale = (action_space.high[0] - action_space.low[0]) / 2
        self.action_bias = (action_space.high[0] + action_space.low[0]) / 2

        self.buffer_size = buffer_size
        self._build_network()


        self.actor_optimizer = get_optimizer(self.actor.parameters(), actor_lr, actor_optimizer, actor_optimizer_kwargs)
        self.critic_optimizer = get_optimizer(
            self.critic.parameters(), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )
        self.critic2_optimizer = get_optimizer(
            self.critic2.parameters(), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.target_noise_scale = target_noise_scale
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

    def __repr__(self) -> str:
        return "TD3"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Actor": DeterministicActor, "Critic": QNetwork, "Buffer": ReplayBuffer}}

    def _build_network(self) -> None:
        actor_class = self.network_list()[self.network_type]["Actor"]
        critic_class = self.network_list()[self.network_type]["Critic"]

        actor_config = self.network_config["Actor"]
        critic_config = self.network_config["Critic"]

        self.actor = actor_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            last_activation="Tanh",
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            device=self.device,
            **actor_config,
        ).train()
        self.target_actor = deepcopy(self.actor).eval()

        self.critic = critic_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **critic_config
        ).train()
        self.target_critic = deepcopy(self.critic).eval()

        self.critic2 = critic_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **critic_config
        ).train()
        self.target_critic2 = deepcopy(self.critic2).eval()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]
        self.buffer = buffer_class(self.state_dim, self.action_dim, self.buffer_size, device=self.device, **buffer_config)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the DDPG action from an observation (in training mode)

        :param observation: The input observation
        :return: The DDPG agent's action
        """
        observation = self._fix_ob_shape(observation)

        self.actor.train()
        with torch.no_grad():
            action = self.actor(observation).cpu().numpy()[0]
            noise = np.random.normal(loc=0, scale=self.noise_scale, size=self.action_dim)

        return np.clip(action + noise, -self.action_scale + self.action_bias, self.action_scale + self.action_bias)

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the DDPG action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The DDPG agent's action (in evaluation mode)
        """
        observation = self._fix_ob_shape(observation)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(observation).cpu().numpy()[0]

        return action

    def train(self) -> Dict[str, Any]:
        """
        Train the TD3 policy.

        :return: Training result (actor_loss, critic_loss, critic2_loss)
        """
        self.training_count += 1
        self.actor.train()

        result_dict = dict()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_action = (
                self.target_actor(ns)
                + torch.normal(0, self.target_noise_scale, (self.batch_size, self.action_dim))
                .clamp(-self.noise_clip, self.noise_clip)
                .to(self.device)
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

        if self.training_count % self.policy_delay == 0:
            actor_loss = -self.critic((s, self.actor(s))).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

            result_dict["loss/actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict["loss/critic_loss"] = critic_loss.detach().cpu().numpy()
        result_dict["loss/critic2_loss"] = critic2_loss.detach().cpu().numpy()

        return result_dict
