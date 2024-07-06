import itertools
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from diverserl.algos.pixel_rl.base import PixelRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import (CategoricalActor, GaussianActor, PixelEncoder,
                                VNetwork)
from diverserl.networks.d2rl_networks import (D2RLCategoricalActor,
                                              D2RLGaussianActor, D2RLVNetwork)


class PPO(PixelRL):
    def __init__(
            self,
            env: gym.vector.SyncVectorEnv,
            eval_env: gym.Env,
            network_type: str = "Default",
            network_config: Optional[Dict[str, Any]] = None,
            horizon: int = 128,
            minibatch_size: int = 64,
            num_epochs: int = 4,
            gamma: float = 0.99,
            lambda_gae: float = 0.96,
            mode: str = "clip",
            target_dist: float = 0.01,
            beta: float = 3.0,
            clip_coef: float = 0.2,
            vf_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            learning_rate: float = 0.001,
            optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            anneal_lr: bool = True,
            device: str = "cpu",
    ) -> None:
        """
        PPO(Proximal Policy Gradients)

        Paper: Proximal Policy Optimization Algorithm, Schulman et al., 2017

        :param env: Gymnasium environment to train the PPO algorithm
        :param eval_env: Gymnasium environment to evaluate the PPO algorithm
        :param network_type: Type of the DQN networks to be used.
        :param network_config: Configurations of the DQN networks.
        :param num_envs: Number of vectorized environments to run RL algorithm on.
        :param horizon: The number of steps to gather in each policy rollout
        :param minibatch_size: Minibatch size for optimizer.
        :param gamma: The discount factor
        :param lambda_gae: The lambda for the General Advantage Estimation (High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b))
        :param mode: Type of surrogate objectives (clip, adaptive_kl, fixed_kl)
        :param target_dist: Target KL divergence between the old policy and the current policy. Only used when the mode is 'adaptive_kl'.
        :param beta: Hyperparameter for the KL divergence in the surrogate.
        :param clip_coef: Surrogate clipping value. Only used when the mode is 'clip'
        :param vf_coef: Critic loss coefficient for optimization.
        :param entropy_coef: Actor Entropy coefficient for optimization.
        :param buffer_size: Maximum length of replay buffer.
        :param max_grad_norm: Maximum gradient norm of the loss. Gradient norm above this value will be clipped.
        :param learning_rate: Learning rate for actor.
        :param optimizer: Optimizer class (or name) for actor.
        :param optimizer_kwargs: Parameter dict for actor optimizer.
        :param anneal_lr: Whether to linearly decrease the learning rate during training.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(
            env=env, eval_env=eval_env, network_type=network_type, network_list=self.network_list(), network_config=network_config,
            device=device
        )
        assert mode.lower() in ["clip", "adaptive_kl", "fixed_kl"]
        assert isinstance(self.observation_space, spaces.Box), f"{self} supports only Box type observation space."

        self.mode = mode.lower()

        self.clip_coef = clip_coef
        self.target_dist = target_dist
        self.beta = beta

        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.num_envs = env.num_envs
        self.horizon = horizon
        self.num_epochs = num_epochs

        self.batch_size = self.num_envs * self.horizon
        self.minibatch_size = minibatch_size

        self.buffer_size = horizon
        self._build_network()

        self.learning_rate = learning_rate

        self.params = itertools.chain(
            *[self.actor.parameters(), self.critic.parameters(), self.encoder.parameters()])

        self.optimizer = get_optimizer(list(self.params), learning_rate, optimizer, optimizer_kwargs)

        self.anneal_lr = anneal_lr

        assert self.num_epochs > 0
        assert self.horizon >= self.minibatch_size

    def __repr__(self) -> str:
        return "PPO"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        ppo_network_list = {
            "Default": {"Actor": {"Discrete": CategoricalActor, "Continuous": GaussianActor}, "Critic": VNetwork,
                        "Encoder": PixelEncoder, "Buffer": ReplayBuffer},
            "D2RL": {"Actor": {"Discrete": D2RLCategoricalActor, "Continuous": D2RLGaussianActor}, "Critic": D2RLVNetwork,
                        "Encoder": PixelEncoder, "Buffer": ReplayBuffer}

        }
        return ppo_network_list

    def _build_network(self) -> None:

        encoder_class = self.network_list()[self.network_type]["Encoder"]
        encoder_config = self.network_config["Encoder"]

        self.encoder = encoder_class(state_dim=self.state_dim, **encoder_config)

        feature_dim = self.encoder.feature_dim

        actor_class = self.network_list()[self.network_type]["Actor"]["Discrete" if self.discrete_action else "Continuous"]
        actor_config = self.network_config["Actor"]

        if not self.discrete_action:  # Fix GaussianActor setting for PPO.
            actor_config["squash"] = False
            actor_config["independent_std"] = True
            actor_config["action_scale"] = self.action_scale
            actor_config["action_bias"] = self.action_bias

        critic_class = self.network_list()[self.network_type]["Critic"]
        critic_config = self.network_config["Critic"]

        self.actor = actor_class(
            state_dim=feature_dim, action_dim=self.action_dim, device=self.device, **actor_config
        ).train()

        self.critic = critic_class(state_dim=feature_dim, device=self.device,
                                   **critic_config).train()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]

        self.buffer = buffer_class(
            state_dim=self.state_dim,
            action_dim=1 if self.discrete_action else self.action_dim,
            save_log_prob=True,
            num_envs=self.num_envs,
            max_size=self.buffer_size,
            device=self.device,
            **buffer_config
        )

    def init_network_weight(self) -> None:
        """
        Initialize network weights for only for PPO networks in continuous action.

        source: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """

        def layer_init(m: nn.Module, std=np.sqrt(2), bias_const=0.0):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, bias_const)

            return m

        self.actor.apply(layer_init)
        self.critic.apply(layer_init)
        layer_init(self.actor.mean_layer, std=0.01)
        layer_init(self.critic.layers[-1], std=1.0)

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._fix_observation(observation)

        self.actor.train()

        with torch.no_grad():
            action, log_prob = self.actor(self.encoder(observation))

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._fix_observation(observation)
        observation = torch.unsqueeze(observation, dim=0)

        self.actor.eval()

        with torch.no_grad():
            action, log_prob = self.actor(self.encoder(observation), deterministic=False)#deterministic actor greatly diminishes the performance.

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def train(self, total_step: int, max_step: int) -> Dict[str, Any]:
        self.training_count += 1
        result_dict = {}

        if self.anneal_lr:
            self.optimizer.param_groups[0]['lr'] = (1 - total_step / max_step) * self.learning_rate

        s, a, r, ns, d, t, log_prob = self.buffer.all_sample()
        ns = ns[-1] # only last ns is used in ppo.

        # Generalized Advantage Estimation(GAE)
        with torch.no_grad():
            dones = torch.logical_or(d, t).to(torch.float32)
            old_values = self.critic(self.encoder(s.reshape(-1, *self.state_dim)).reshape(self.horizon, self.num_envs, self.encoder.feature_dim))
            previous_value = self.critic(self.encoder(ns))

            advantages = torch.zeros_like(r)
            running_advantage = torch.zeros((self.num_envs, 1))

            for t in reversed(range(self.horizon)):
                running_tderror = r[t] + self.gamma * previous_value * (1 - dones[t]) - old_values[t]
                running_advantage = running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (
                            1 - dones[t])

                previous_value = old_values[t]
                advantages[t] = running_advantage

            returns = advantages + old_values

        s = s.view(-1, *self.state_dim)
        a = a.reshape(-1, 1 if self.discrete_action else self.action_dim)
        log_prob = log_prob.reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)
        returns = returns.reshape(-1, 1)
        old_values = old_values.reshape(-1, 1)

        batch_idxes = np.arange(self.batch_size)
        clip_fracs = []
        for epoch in range(self.num_epochs):
            np.random.shuffle(batch_idxes)

            for start in range(int(self.batch_size // self.minibatch_size)):
                minibatch_idxes = batch_idxes[start * self.minibatch_size: (start + 1) * self.minibatch_size]

                batch_s = s[minibatch_idxes]
                batch_a = a[minibatch_idxes]
                batch_log_prob = log_prob[minibatch_idxes]
                batch_advantages = advantages[minibatch_idxes]
                batch_returns = returns[minibatch_idxes]
                batch_old_values = old_values[minibatch_idxes]

                # advantages normalization
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # for computational efficiency
                batch_feature = self.encoder(batch_s)

                # actor loss
                log_policy = self.actor.log_prob(batch_feature, batch_a)
                log_ratio = log_policy - batch_log_prob
                ratio = log_ratio.exp()
                kl = (ratio - 1) - log_ratio

                surrogate = ratio * batch_advantages

                # different actor loss by ppo mode
                if self.mode == "clip":
                    clipped_surrogate = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                    actor_loss = -torch.minimum(clipped_surrogate, surrogate).mean()
                else:
                    # http://joschu.net/blog/kl-approx.html

                    actor_loss = -(surrogate - self.beta * kl).mean()

                    if self.mode == "adaptive_kl":
                        if kl.mean().item() < self.target_dist / 1.5:
                            self.beta /= 2
                        elif kl.mean().item() > self.target_dist * 1.5:
                            self.beta *= 2

                entropy_loss = self.actor.entropy(batch_feature).mean()
                actor_loss -= self.entropy_coef * entropy_loss

                #critic loss
                new_values = self.critic(batch_feature)

                critic_loss_unclipped = (new_values - batch_returns) ** 2
                clipped_values = batch_old_values + torch.clamp(new_values - batch_old_values, -self.clip_coef,
                                                                self.clip_coef)
                critic_loss_clipped = (clipped_values - batch_returns) ** 2

                critic_loss = self.vf_coef * 0.5 * (torch.max(critic_loss_unclipped, critic_loss_clipped).mean())

                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()

                #calculate debug variables
                with torch.no_grad():
                    old_kl = (-log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

        self.buffer.reset()

        # record last minibatch training result
        result_dict['loss/actor_loss'] = actor_loss.detach().cpu().numpy()
        result_dict['loss/critic_loss'] = critic_loss.detach().cpu().numpy()
        result_dict['loss/entropy_loss'] = entropy_loss.detach().cpu().numpy()
        result_dict['loss/kl'] = kl.mean().detach().cpu().numpy()
        result_dict['old_kl'] = old_kl.detach().cpu().numpy()
        result_dict['clip_frac'] = np.mean(clip_fracs)
        result_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

        return result_dict
