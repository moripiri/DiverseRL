from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
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
        actor_config = self.network_config["Actor"]["Discrete" if self.discrete else "Continuous"]

        critic_class = self.network_list()[self.network_type]["Critic"]
        critic_config = self.network_config["Critic"]

        self.actor = actor_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **actor_config
        ).train()
        self.critic = critic_class(state_dim=self.state_dim, device=self.device, **critic_config).train()

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()

        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.cpu().numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        observation = super()._fix_ob_shape(observation)

        self.actor.train()

        with torch.no_grad():
            action, _ = self.actor(observation, deterministic=True)

        return action.cpu().numpy()[0]

    def train(self) -> Dict[str, Any]:
        pass
