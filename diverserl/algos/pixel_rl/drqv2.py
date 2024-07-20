from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.pixel_rl.base import PixelRL
from diverserl.common.buffer import NstepReplayBuffer
from diverserl.common.image_augmentation import random_shift_aug
from diverserl.common.utils import get_optimizer, hard_update, soft_update
from diverserl.networks import DeterministicActor, PixelEncoder, QNetwork
from diverserl.networks.d2rl_networks import (D2RLDeterministicActor,
                                              D2RLQNetwork)


class DrQv2(PixelRL):
    def __init__(
            self,
            env: gym.vector.SyncVectorEnv,
            eval_env: gym.Env,
            network_type: str = "Default",
            network_config: Optional[Dict[str, Any]] = None,
            image_pad: int = 4,
            gamma: float = 0.99,
            tau: float = 0.05,
            encoder_tau: float = 0.05,
            noise_scale_init: float = 1.0,
            noise_scale_final: float = 0.1,
            noise_decay_horizon: int = 100000,
            target_noise_scale: float = 0.2,
            noise_clip: float = 0.5,
            policy_delay: int = 2,
            batch_size: int = 256,
            buffer_size: int = 10 ** 6,
            actor_lr: float = 0.001,
            actor_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
            actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            critic_lr: float = 0.001,
            critic_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
            critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            device: str = "cpu",
    ) -> None:
        """
        DrQ-v2

        Paper: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning, Yarats et al., 2021

        :param env: Gymnasium environment to train the TD3 algorithm
        :param network_type: Type of the TD3 networks to be used.
        :param network_config: Configurations of the TD3 networks.
        :param gamma: Discount factor.
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param encoder_tau: Interpolation factor in polyak averaging for target encoder network.
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
            env=env, eval_env=eval_env, network_type=network_type, network_list=self.network_list(),
            network_config=network_config, device=device
        )

        assert isinstance(
            self.action_space, gym.spaces.Box), f"{self} supports only Box type action space."

        self.image_pad = image_pad

        self.buffer_size = buffer_size
        self._build_network()

        self.actor_optimizer = get_optimizer(self.actor.parameters(), actor_lr, actor_optimizer, actor_optimizer_kwargs)
        self.critic_optimizer = get_optimizer(
            list(chain(*[self.critic.parameters(), self.critic2.parameters(), self.encoder.parameters()])), critic_lr, critic_optimizer,
            critic_optimizer_kwargs
        )

        self.gamma = gamma
        self.tau = tau
        self.encoder_tau = encoder_tau
        self.batch_size = batch_size

        self.noise_scale_init = noise_scale_init
        self.noise_scale_final = noise_scale_final
        self.noise_decay_horizon = noise_decay_horizon

        self.noise_scale = noise_scale_init

        self.target_noise_scale = target_noise_scale
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.action_count = 0

    def __repr__(self) -> str:
        return "DrQv2"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Actor": DeterministicActor, "Critic": QNetwork, "Encoder": PixelEncoder, "Buffer": NstepReplayBuffer},
                "D2RL": {"Actor": D2RLDeterministicActor, "Critic": D2RLQNetwork, "Encoder": PixelEncoder, "Buffer": NstepReplayBuffer}}

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

        self.actor = actor_class(
            state_dim=feature_dim,
            action_dim=self.action_dim,
            last_activation="Tanh",
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            device=self.device,
            **actor_config,
        ).train()
        self.target_actor = deepcopy(self.actor).eval()

        self.critic = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, **critic_config
        ).train()
        self.target_critic = deepcopy(self.critic).eval()

        self.critic2 = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, **critic_config
        ).train()
        self.target_critic2 = deepcopy(self.critic2).eval()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]
        self.buffer = buffer_class(self.state_dim, self.action_dim, self.buffer_size, device=self.device,
                                   **buffer_config)

    def update_noise_scale(self) -> None:
        self.noise_scale = self.noise_scale_init + \
                           (1 - min(self.action_count / self.noise_decay_horizon, 1)) * (self.noise_scale_final - self.noise_scale_init)
        self.action_count += 1


    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the TD3 action from an observation (in training mode)

        :param observation: The input observation
        :return: The TD3 agent's action
        """
        self.update_noise_scale()
        observation = self._fix_observation(observation)

        self.actor.train()
        with torch.no_grad():
            action = self.actor(self.encoder(observation)).cpu().numpy()[0]
            noise = np.random.normal(loc=0, scale=self.noise_scale, size=(action.shape[0], self.action_dim))

        return np.clip(action + noise, -self.action_scale + self.action_bias, self.action_scale + self.action_bias)

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the TD3 action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The TD3 agent's action (in evaluation mode)
        """
        observation = self._fix_observation(observation)
        observation = torch.unsqueeze(observation, dim=0)

        self.actor.eval()

        with torch.no_grad():
            action = self.actor(self.encoder(observation)).cpu().numpy()

        return action

    def train(self, total_step: int, max_step: int) -> Dict[str, Any]:
        """
        Train the TD3 policy.

        :return: Training result (actor_loss, critic_loss, critic2_loss)
        """
        self.training_count += 1
        self.actor.train()

        result_dict = dict()

        s, a, r, ns, d, t, discounts = self.buffer.sample(self.batch_size)

        s = random_shift_aug(s.to(torch.float32), image_pad=self.image_pad)
        ns = random_shift_aug(ns.to(torch.float32), image_pad=self.image_pad)

        with torch.no_grad():
            target_feature_ns = self.target_encoder(ns)
            target_action = (
                    self.target_actor(target_feature_ns)
                    + torch.normal(0, self.target_noise_scale, (self.batch_size, self.action_dim))
                    .clamp(-self.noise_clip, self.noise_clip)
                    .to(self.device)
            ).clamp(-self.action_scale + self.action_bias, self.action_scale + self.action_bias)
            target_value = r + discounts * torch.minimum(
                self.target_critic((target_feature_ns, target_action)), self.target_critic2((target_feature_ns, target_action))
            )

        feature_s = self.encoder(s)
        critic_loss = F.mse_loss(self.critic((feature_s, a)), target_value) + F.mse_loss(self.critic2((feature_s, a)), target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.training_count % self.policy_delay == 0:
            feature_s = feature_s.detach()
            actor_loss = -self.critic((feature_s, self.actor(feature_s))).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)

            result_dict["loss/actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict["loss/critic_loss"] = critic_loss.detach().cpu().numpy()

        return result_dict
