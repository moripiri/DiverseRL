from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer
from diverserl.networks import CategoricalActor, GaussianActor, VNetwork


class PPO(DeepRL):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        network_type: str = "Default",
        network_config: Optional[Dict[str, Any]] = None,
        horizon: int = 128,
        minibatch_size: int = 64,
        num_epochs: int = 4,
        gamma: float = 0.99,
        lambda_gae: float = 0.96,
        mode: str = "clip",
        clip: float = 0.2,
        target_dist: float = 0.01,
        beta: float = 3.0,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
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
        PPO(Proximal Policy Gradients)

        Paper: Proximal Policy Optimization Algorithm, Schulman et al., 2017

        :param observation_space: Observation space of the environment for RL agent to learn from
        :param action_space: Action space of the environment for RL agent to learn from
        :param network_type: Type of the DQN networks to be used.
        :param network_config: Configurations of the DQN networks.
        :param mode: Type of surrogate objectives (clip, adaptive_kl, fixed_kl)
        :param clip: The surrogate clipping value. Only used when the mode is 'clip'
        :param target_dist: Target KL divergence between the old policy and the current policy. Only used when the mode is 'adaptive_kl'.
        :param beta: Hyperparameter for the KL divergence in the surrogate.
        :param gamma: The discount factor
        :param lambda_gae: The lambda for the General Advantage Estimation (High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b))
        :param horizon: The number of steps to gather in each policy rollout
        :param minibatch_size: Minibatch size for optimizer.
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
        assert mode.lower() in ["clip", "adaptive_kl", "fixed_kl"]
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
        self.buffer_size = horizon

        self._build_network()

        self.actor_optimizer = get_optimizer(self.actor.parameters(), actor_lr, actor_optimizer, actor_optimizer_kwargs)
        self.critic_optimizer = get_optimizer(
            self.critic.parameters(), critic_lr, critic_optimizer, critic_optimizer_kwargs
        )

        self.mode = mode.lower()

        self.clip = clip
        self.target_dist = target_dist
        self.beta = beta

        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.minibatch_size = minibatch_size
        self.horizon = horizon
        self.num_epochs = num_epochs

        assert horizon >= minibatch_size

        self.num_epochs = int(horizon // minibatch_size)

    def __repr__(self) -> str:
        return "PPO"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        ppo_network_list = {
            "Default": {"Actor": {"Discrete": CategoricalActor, "Continuous": GaussianActor}, "Critic": VNetwork, "Buffer": ReplayBuffer}
        }
        return ppo_network_list

    def _build_network(self) -> None:
        actor_class = self.network_list()[self.network_type]["Actor"]["Discrete" if self.discrete else "Continuous"]
        actor_config = self.network_config["Actor"]

        if not self.discrete:  # Fix GaussianActor setting for PPO.
            actor_config["squash"] = False
            actor_config["independent_std"] = True

        critic_class = self.network_list()[self.network_type]["Critic"]
        critic_config = self.network_config["Critic"]

        self.actor = actor_class(
            state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, **actor_config
        ).train()
        self.critic = critic_class(state_dim=self.state_dim, device=self.device, **critic_config).train()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]

        self.buffer = buffer_class(
            state_dim=self.state_dim,
            action_dim=1 if self.discrete else self.action_dim,
            save_log_prob=True,
            max_size=self.buffer_size,
            device=self.device,
            **buffer_config
        )
    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._fix_ob_shape(observation)

        self.actor.train()

        with torch.no_grad():
            action, log_prob = self.actor(observation)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._fix_ob_shape(observation)

        self.actor.eval()

        with torch.no_grad():
            action, log_prob = self.actor(observation, deterministic=True)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def train(self) -> Dict[str, Any]:
        total_a_loss, total_c_loss = 0.0, 0.0
        s, a, r, ns, d, t, log_prob = self.buffer.all_sample()

        dones = torch.logical_or(d, t).to(torch.float32)

        old_values = self.critic(s).detach()
        returns = torch.zeros_like(r)
        advantages = torch.zeros_like(r)

        running_return = torch.zeros(1)
        previous_value = self.critic(ns).detach()[-1]
        running_advantage = torch.zeros(1)

        # Generalized Advantage Estimation(GAE)
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - dones[t])
            running_tderror = r[t] + self.gamma * previous_value * (1 - dones[t]) - old_values[t]
            running_advantage = running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - dones[t])

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        horizon_idxes = np.arange(self.horizon)

        for epoch in range(self.num_epochs):
            np.random.shuffle(horizon_idxes)

            for start in range(int(self.horizon // self.minibatch_size)):
                minibatch_idxes = horizon_idxes[start * self.minibatch_size: (start + 1) * self.minibatch_size]

                batch_s = s[minibatch_idxes]
                batch_a = a[minibatch_idxes]
                batch_log_prob = log_prob[minibatch_idxes]
                batch_advantages = advantages[minibatch_idxes]
                batch_returns = returns[minibatch_idxes]

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                log_policy = self.actor.log_prob(batch_s, batch_a)
                log_ratio = log_policy - batch_log_prob
                ratio = log_ratio.exp()

                surrogate = ratio * batch_advantages

                if self.mode == "clip":
                    clipped_surrogate = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                    actor_loss = -torch.minimum(clipped_surrogate, surrogate).mean()
                else:
                    # http://joschu.net/blog/kl-approx.html
                    kl = (ratio - 1) - log_ratio

                    actor_loss = -(surrogate - self.beta * kl).mean()

                    if self.mode == "adaptive_kl":
                        if kl.mean().item() < self.target_dist / 1.5:
                            self.beta /= 2
                        elif kl.mean().item() > self.target_dist * 1.5:
                            self.beta *= 2

                actor_loss -= self.entropy_coef * self.actor.entropy(batch_s).mean()
                critic_loss = self.vf_coef * F.mse_loss(self.critic(batch_s), batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_a_loss += actor_loss.detach().cpu().numpy()
                total_c_loss += critic_loss.detach().cpu().numpy()

        self.buffer.delete()

        return {"loss/actor_loss": total_a_loss, "loss/critic_loss": total_c_loss}
