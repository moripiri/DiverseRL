from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.pixel_rl.base import PixelRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.image_augmentation import center_crop, random_crop
from diverserl.common.utils import (fix_observation, get_optimizer,
                                    hard_update, soft_update)
from diverserl.networks import GaussianActor, PixelEncoder, QNetwork
from diverserl.networks.d2rl_networks import D2RLGaussianActor, D2RLQNetwork


class RAD(PixelRL):
    def __init__(self,
                 env: gym.vector.VectorEnv,
                 eval_env: gym.vector.VectorEnv,
                 network_type: str = "Default",
                 network_config: Optional[Dict[str, Any]] = None,
                 pre_image_size: int = 100,
                 image_size: int = 84,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 train_alpha: bool = True,
                 target_alpha: Optional[float] = None,
                 tau: float = 0.01,
                 encoder_tau: float = 0.05,
                 target_critic_update: int = 2,
                 batch_size: int = 256,
                 buffer_size: int = 10 ** 6,
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
        RAD (Reinforcement learning with Augmented Data)

        Paper: Reinforcement Learning with Augmented Data, Laskin et al., 2020

        :param env: Gymnasium environment to train the RAD algorithm
        :param eval_env: Gymnasium environment to evaluate the RAD algorithm
        :param network_type: Type of the RAD networks to be used.
        :param network_config: Configurations of the RAD networks.
        :param pre_image_size: Size of images provided by environment.
        :param image_size: Image size for random crop.
        :param gamma: The discount factor.
        :param alpha: The entropy temperature parameter.
        :param train_alpha: Whether to train the parameter alpha.
        :param target_alpha: Target entropy value (usually set as -|action_dim|).
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param encoder_tau: Interpolation factor in polyak averaging for target encoder network.
        :param target_critic_update: Critic will only be updated once for every target_critic_update steps.
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
        super().__init__(
            env=env, eval_env=eval_env, network_type=network_type, network_list=self.network_list(),
            network_config=network_config, device=device
        )
        assert pre_image_size > image_size, "Pre-image size must be greater than image size"
        self.pre_image_size = pre_image_size
        self.image_size = image_size

        assert self.state_dim[1:] == (self.pre_image_size, self.pre_image_size)

        self.pre_state_dim = self.state_dim
        self.state_dim = (self.state_dim[0], self.image_size, self.image_size)

        self.buffer_size = buffer_size

        self._build_network()

        hard_update(self.critic, self.target_critic)
        hard_update(self.critic2, self.target_critic2)
        hard_update(self.encoder, self.target_encoder)

        self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=train_alpha)
        self.target_alpha = -self.action_dim if target_alpha is None else target_alpha
        self.train_alpha = train_alpha

        self.target_critic_update = target_critic_update

        self.actor_optimizer = get_optimizer(self.actor.parameters(), actor_lr, actor_optimizer, actor_optimizer_kwargs)
        self.critic_optimizer = get_optimizer(
            list(chain(*[self.critic.parameters(), self.critic2.parameters(), self.encoder.parameters()])), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )

        if self.train_alpha:
            self.alpha_optimizer = get_optimizer([self.log_alpha], alpha_lr, alpha_optimizer, alpha_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.encoder_tau = encoder_tau

        self.batch_size = batch_size

    def __repr__(self) -> str:
        return "RAD"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {
            "Default": {"Actor": GaussianActor, "Critic": QNetwork, "Encoder": PixelEncoder, "Buffer": ReplayBuffer},
            "D2RL": {"Actor": D2RLGaussianActor, "Critic": D2RLQNetwork, "Encoder": PixelEncoder,
                     "Buffer": ReplayBuffer}, }

    def _build_network(self) -> None:
        encoder_class = self.network_list()[self.network_type]["Encoder"]
        encoder_config = self.network_config["Encoder"]
        self.encoder = encoder_class(state_dim=self.state_dim, device=self.device, **encoder_config)
        self.target_encoder = deepcopy(self.encoder).eval()

        feature_dim = self.encoder.feature_dim

        actor_class = self.network_list()[self.network_type]["Actor"]
        critic_class = self.network_list()[self.network_type]["Critic"]

        actor_config = self.network_config["Actor"]
        critic_config = self.network_config["Critic"]

        # Fix GaussianActor setting for RAD
        actor_config["squash"] = True
        actor_config["independent_std"] = False

        self.actor = actor_class(
            state_dim=feature_dim,
            action_dim=self.action_dim,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            device=self.device,
            **actor_config,
        ).train()

        self.critic = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device,
            **critic_config
        ).train()
        self.target_critic = deepcopy(self.critic).eval()

        self.critic2 = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device,
            **critic_config
        ).train()
        self.target_critic2 = deepcopy(self.critic2).eval()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]
        self.buffer = buffer_class(self.pre_state_dim, self.action_dim, self.buffer_size, device=self.device,
                                   **buffer_config)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the RAD action from an observation (in training mode)

        :param observation: The input observation
        :return: The RAD agent's action
        """
        observation = fix_observation(observation, self.device)

        self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(self.encoder(center_crop(observation, output_size=self.image_size)))

        return action.cpu().numpy()

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the  action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The RAD agent's action (in evaluation mode)
        """
        observation = fix_observation(observation, self.device)

        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(self.encoder(center_crop(observation, output_size=self.image_size)))

        return action.cpu().numpy()

    @property
    def alpha(self):
        """
        :return: Return current alpha value as float
        """
        return self.log_alpha.exp().detach()

    def train(self, total_step: int, max_step: int) -> Dict[str, Any]:
        """
        Train RAD policy.
        :return: training results
        """
        self.training_count += 1
        result_dict = {}
        self.actor.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        s = random_crop(s, self.image_size).to(torch.float32)
        ns = random_crop(ns, self.image_size).to(torch.float32)

        # critic network training
        with torch.no_grad():
            ns_action, ns_logprob = self.actor(self.encoder(ns))
            target_feature_ns = self.target_encoder(ns)
            target_min_aq = torch.minimum(self.target_critic((target_feature_ns, ns_action)), self.target_critic2((target_feature_ns, ns_action)))

            target_q = r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logprob)

        feature_s = self.encoder(s)
        critic_loss = F.mse_loss(self.critic((feature_s, a)), target_q) + F.mse_loss(self.critic2((feature_s, a)), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor training
        with torch.no_grad():
            feature_s = self.encoder(s)

        s_action, s_logprob = self.actor(feature_s)
        min_aq_rep = torch.minimum(self.critic((feature_s, s_action)), self.critic2((feature_s, s_action)))

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

            result_dict["loss/alpha_loss"] = alpha_loss.detach().cpu().numpy()

        # critic update
        if self.training_count % self.target_critic_update == 0:
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        result_dict["loss/actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict["loss/critic_loss"] = critic_loss.detach().cpu().numpy()
        result_dict["value/alpha"] = self.alpha.cpu().numpy()

        return result_dict
