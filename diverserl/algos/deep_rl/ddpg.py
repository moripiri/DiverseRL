from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, soft_update
from diverserl.networks.basic_networks import DeterministicActor, QNetwork


class DDPG(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "MLP",
        network_config: Optional[Dict[str, Any]] = None,
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
        device: str = "cpu",
    ) -> None:
        """
        DDPG(Deep Deterministic Policy Gradients)

        Paper: Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.

        :param env: The environment for RL agent to learn from
        :param gamma: The discount factor
        :param network_type: Type of the DDPG networks to be used.
        :param network_config: Configurations of the DDPG networks.
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

        self._build_network()

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        actor_optimizer, actor_optimizer_kwargs = get_optimizer(actor_optimizer, actor_optimizer_kwargs)
        critic_optimizer, critic_optimizer_kwargs = get_optimizer(critic_optimizer, critic_optimizer_kwargs)

        self.actor_optimizer = actor_optimizer(self.actor.parameters(), lr=actor_lr, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale

    def __repr__(self) -> str:
        return "DDPG"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"MLP": {"actor": DeterministicActor, "critic": QNetwork}}

    def _build_network(self) -> None:
        actor_class = self.network_list()[self.network_type]["actor"]
        critic_class = self.network_list()[self.network_type]["critic"]

        actor_config = self.network_config["actor"]
        critic_config = self.network_config["critic"]

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
        self.training_count += 1
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

        return {"actor_loss": actor_loss.detach().cpu().numpy(), "critic_loss": critic_loss.detach().cpu().numpy()}
