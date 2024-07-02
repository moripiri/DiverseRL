from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.pixel_rl.base import PixelRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, hard_update, soft_update
from diverserl.networks import (GaussianActor, PixelDecoder, PixelEncoder,
                                QNetwork)
from diverserl.networks.d2rl_networks import D2RLGaussianActor, D2RLQNetwork


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class SAC_AE(PixelRL):
    def __init__(self,
                 env: gym.vector.SyncVectorEnv,
                 eval_env: gym.Env,
                 network_type: str = "Default",
                 network_config: Optional[Dict[str, Any]] = None,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 train_alpha: bool = True,
                 target_alpha: Optional[float] = None,
                 tau: float = 0.01,
                 encoder_tau: float = 0.05,
                 target_critic_update: int = 2,
                 decoder_update: int = 1,
                 decoder_latent_lambda: float = 1e-6,
                 decoder_weight_lambda: float = 1e-7,
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
                 encoder_lr: float = 0.001,
                 encoder_optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
                 encoder_optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 decoder_lr: float = 0.001,
                 decoder_optimizer: Union[str, Type[torch.optim.Optimizer]] = "AdamW",
                 decoder_optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: str = "cpu",
                 ) -> None:
        super().__init__(
            env=env, eval_env=eval_env, network_type=network_type, network_list=self.network_list(),
            network_config=network_config, device=device
        )

        self.buffer_size = buffer_size

        self.buffer_size = buffer_size

        self._build_network()

        self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=train_alpha)
        self.target_alpha = -self.action_dim if target_alpha is None else target_alpha
        self.train_alpha = train_alpha

        self.target_critic_update = target_critic_update
        self.decoder_update = decoder_update
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_weight_lambda = decoder_weight_lambda

        self.actor_optimizer = get_optimizer(self.actor.layers.parameters(), actor_lr, actor_optimizer, actor_optimizer_kwargs)
        self.critic_optimizer = get_optimizer(
            self.critic.parameters(), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )
        self.critic2_optimizer = get_optimizer(
            self.critic2.parameters(), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )

        if self.train_alpha:
            self.alpha_optimizer = get_optimizer([self.log_alpha], alpha_lr, alpha_optimizer, alpha_optimizer_kwargs)

        self.encoder_optimizer = get_optimizer(self.encoder.parameters(), encoder_lr, encoder_optimizer, encoder_optimizer_kwargs)
        self.decoder_optimizer = get_optimizer(self.decoder.parameters(), decoder_lr, decoder_optimizer, decoder_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.encoder_tau = encoder_tau

        self.batch_size = batch_size

    def __repr__(self) -> str:
        return "SAC_AE"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {
            "Default": {"Actor": GaussianActor, "Critic": QNetwork, "Encoder": PixelEncoder, "Decoder": PixelDecoder, "Buffer": ReplayBuffer},
            "D2RL": {"Actor": D2RLGaussianActor, "Critic": D2RLQNetwork, "Encoder": PixelEncoder, "Decoder": PixelDecoder,
                     "Buffer": ReplayBuffer}, }

    def _build_network(self) -> None:
        assert not isinstance(self.state_dim, int), "SAC_AE only supports image-type observation."

        encoder_class = self.network_list()[self.network_type]["Encoder"]
        encoder_config = self.network_config["Encoder"]
        self.encoder = encoder_class(state_dim=self.state_dim, **encoder_config)

        self.target_encoder = deepcopy(self.encoder).eval()

        feature_dim = self.encoder.feature_dim

        decoder_class = self.network_list()[self.network_type]["Decoder"]
        decoder_config = self.network_config["Decoder"]
        self.decoder = decoder_class(state_dim=self.state_dim, feature_dim=feature_dim, **decoder_config)

        actor_class = self.network_list()[self.network_type]["Actor"]
        critic_class = self.network_list()[self.network_type]["Critic"]

        actor_config = self.network_config["Actor"]
        critic_config = self.network_config["Critic"]

        # Fix GaussianActor setting for SAC
        actor_config["squash"] = True
        actor_config["independent_std"] = False

        self.actor = actor_class(
            state_dim=feature_dim,
            action_dim=self.action_dim,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            device=self.device,
            feature_encoder=self.encoder,
            **actor_config,
        ).train()

        self.critic = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, feature_encoder=self.encoder,
            **critic_config
        ).train()
        self.target_critic = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, feature_encoder=self.target_encoder,
            **critic_config
        ).eval()

        self.critic2 = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, feature_encoder=self.encoder,
            **critic_config
        ).train()
        self.target_critic2 = critic_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, feature_encoder=self.target_encoder,
            **critic_config
        ).eval()

        hard_update(self.critic, self.target_critic)
        hard_update(self.critic2, self.target_critic2)
        hard_update(self.encoder, self.target_encoder)

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]
        self.buffer = buffer_class(self.state_dim, self.action_dim, self.buffer_size, device=self.device,
                                   **buffer_config)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the SAC action from an observation (in training mode)

        :param observation: The input observation
        :return: The SAC agent's action
        """
        observation = self._fix_observation(observation)

        self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.cpu().numpy()

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the SAC action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The SAC agent's action (in evaluation mode)
        """
        observation = self._fix_observation(observation)
        observation = torch.unsqueeze(observation, dim=0)

        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.cpu().numpy()

    @property
    def alpha(self):
        """
        :return: Return current alpha value as float
        """
        return self.log_alpha.exp().detach()

    def train(self, total_step: int, max_step: int) -> Dict[str, Any]:
        """
        Train SAC policy.
        :return: training results
        """
        self.training_count += 1
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

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()

        self.critic_optimizer.step()
        self.critic2_optimizer.step()

        # actor training
        s_action, s_logprob = self.actor(s, detach_encoder=True)
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

            result_dict["loss/alpha_loss"] = alpha_loss.detach().cpu().numpy()

        # critic update
        if self.training_count % self.target_critic_update == 0:
            soft_update(self.critic.layers, self.target_critic.layers, self.tau)
            soft_update(self.critic2.layers, self.target_critic2.layers, self.tau)

            soft_update(self.critic.feature_encoder, self.target_critic.feature_encoder, self.encoder_tau)

        # decoder update
        if self.training_count % self.target_critic_update == 0:
            feature = self.encoder(s)
            recovered_s = self.decoder(feature)
            real_s = preprocess_obs(s)

            rec_loss = F.mse_loss(recovered_s, real_s)
            latent_loss = (0.5 * torch.sum(feature ** 2, dim=1)).mean()

            ae_loss = rec_loss + self.decoder_latent_lambda * latent_loss

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            ae_loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            result_dict["loss/ae_loss"] = ae_loss.detach().cpu().numpy()

        result_dict["loss/actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict["loss/critic_loss"] = critic_loss.detach().cpu().numpy()
        result_dict["loss/critic2_loss"] = critic2_loss.detach().cpu().numpy()
        result_dict["value/alpha"] = self.alpha.cpu().numpy()

        return result_dict
