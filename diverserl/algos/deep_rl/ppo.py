from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import CategoricalActor, GaussianActor, VNetwork


class PPO(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "MLP",
        network_config: Optional[Dict[str, Any]] = None,
        mode: str = "clip",
        clip: float = 0.2,
        target_dist: float = 0.01,
        beta: float = 3.0,
        gamma: float = 0.99,
        lambda_gae: float = 0.96,
        horizon: int = 128,
        num_epochs: int = 4,
        batch_size: int = 64,
        buffer_size: int = 10**6,
        actor_lr: float = 0.001,
        actor_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_lr: float = 0.001,
        critic_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )
        assert mode.lower() in ["clip", "adaptive_kl", "kl"]
        assert isinstance(observation_space, spaces.Box), f"{self} supports only Box type observation space."

        self.state_dim = observation_space.shape[0]

        # REINFORCE supports both discrete and continuous action space.
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.discrete = True
        elif isinstance(action_space, spaces.Box):
            self.action_dim = action_space.shape[0]
            self.discrete = False
        else:
            raise TypeError

        self._build_network()
        self.buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=1 if self.discrete else self.action_dim,
            save_log_prob=True,
            max_size=buffer_size,
            device=self.device,
        )

        actor_optimizer, actor_optimizer_kwargs = get_optimizer(actor_optimizer, actor_optimizer_kwargs)
        critic_optimizer, critic_optimizer_kwargs = get_optimizer(critic_optimizer, critic_optimizer_kwargs)

        self.actor_optimizer = actor_optimizer(self.actor.parameters(), lr=actor_lr, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), lr=critic_lr, **critic_optimizer_kwargs)

        self.mode = mode.lower()

        self.clip = clip
        self.target_dist = target_dist
        self.beta = beta

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.batch_size = batch_size

        self.horizon = horizon
        self.num_epochs = num_epochs

    def __repr__(self) -> str:
        return "PPO"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        ppo_network_list = {
            "MLP": {"Actor": {"Discrete": CategoricalActor, "Continuous": GaussianActor}, "Critic": VNetwork}
        }
        return ppo_network_list

    def _build_network(self) -> None:
        actor_class = self.network_list()[self.network_type]["Actor"]["Discrete" if self.discrete else "Continuous"]
        actor_config = self.network_config["Actor"]

        critic_class = self.network_list()[self.network_type]["Critic"]
        critic_config = self.network_config["Critic"]

        self.actor = actor_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **actor_config
        ).train()
        self.critic = critic_class(state_dim=self.state_dim, device=self.device, **critic_config).train()

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()

        with torch.no_grad():
            action, log_prob = self.actor(observation)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()

        with torch.no_grad():
            action, log_prob = self.actor(observation, deterministic=True)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def train(self) -> Dict[str, Any]:
        s, a, r, ns, d, t, log_prob = self.buffer.all_sample()

        old_values = self.critic(s).detach()
        returns = torch.zeros_like(r)
        advantages = torch.zeros_like(r)

        running_return = torch.zeros(1)
        previous_value = torch.zeros(1)
        running_advantage = torch.zeros(1)

        # GAE
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - d[t])
            running_tderror = r[t] + self.gamma * previous_value * (1 - d[t]) - old_values[t]
            running_advantage = running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t])

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std())
        returns = (returns - returns.mean()) / (returns.std())

        for epoch in range(self.num_epochs):
            log_policy = self.actor.log_prob(s, a)
            ratio = (log_policy - log_prob).exp()

            surrogate = ratio * advantages
            clipped_surrogate = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            actor_loss = -torch.minimum(clipped_surrogate, surrogate).mean()
            critic_loss = F.mse_loss(self.critic(s), returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer.delete()

        return {"actor_loss": actor_loss.detach().cpu().numpy(), "critic_loss": critic_loss.detach().cpu().numpy()}
