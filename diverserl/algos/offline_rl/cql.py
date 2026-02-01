from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from diverserl.algos.offline_rl.base import OfflineRL
from diverserl.common.buffer import DatasetBuffer
from diverserl.common.utils import fix_observation, get_optimizer, soft_update
from diverserl.networks import QNetwork
from diverserl.networks.d2rl_networks import D2RLGaussianActor, D2RLQNetwork
from diverserl.networks.gaussian_actor import GaussianActor


class CQL(OfflineRL):
    def __init__(self,
                 buffer: DatasetBuffer,
                 eval_env: gym.vector.VectorEnv,
                 network_type: str = "Default",
                 network_config: Optional[
                     Dict[str, Any]] = None,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 cql_alpha: float = 0.1,
                 train_alpha: bool = True,
                 target_alpha: Optional[float] = None,
                 with_lagrange: bool = True,
                 tau: float = 0.05,
                 target_critic_update: int = 2,
                 num_random: int = 10,
                 temp: float = 1.0,
                 min_q_weight: float = 1.0,
                 target_action_gap: float = 0.0,
                 batch_size: int = 256,
                 actor_lr: float = 3e-5,
                 actor_optimizer: Union[str, Type[
                     torch.optim.Optimizer]] = "Adam",
                 actor_optimizer_kwargs: Optional[
                     Dict[str, Any]] = None,
                 critic_lr: float = 3e-4,
                 critic_optimizer: Union[str, Type[
                     torch.optim.Optimizer]] = "Adam",
                 critic_optimizer_kwargs: Optional[
                     Dict[str, Any]] = None,
                 alpha_lr: float = 0.001,
                 alpha_optimizer: Union[str, Type[
                     torch.optim.Optimizer]] = "Adam",
                 alpha_optimizer_kwargs: Optional[
                     Dict[str, Any]] = None,
                 cql_alpha_lr: float = 0.001,
                 cql_alpha_optimizer: Union[str, Type[
                     torch.optim.Optimizer]] = "Adam",
                 cql_alpha_optimizer_kwargs: Optional[
                     Dict[str, Any]] = None,
                 device: str = "cpu",
                 ) -> None:
        """
        CQL (Conservative Q-Learning for Offline Reinforcement Learning)

        Paper: Conservative Q-Learning for Offline Reinforcement Learning, Kumar et al., 2020

        :param buffer: Dataset buffer containing offline training data.
        :param eval_env: Gymnasium vectorized environment for evaluation.
        :param network_type: Type of the CQL networks to be used ("Default" or "D2RL").
        :param network_config: Configurations of the CQL networks (Actor and Critic configs).
        :param gamma: The discount factor for future rewards.
        :param alpha: The entropy temperature parameter for SAC component.
        :param cql_alpha: The CQL regularization coefficient.
        :param train_alpha: Whether to train the entropy temperature parameter alpha.
        :param target_alpha: Target entropy value (usually set as -|action_dim|). If None, defaults to -action_dim.
        :param with_lagrange: Whether to use Lagrange multiplier for CQL constraint.
        :param tau: Interpolation factor in polyak averaging for target networks.
        :param target_critic_update: Critic will only be updated once for every target_critic_update steps.
        :param num_random: Number of random actions to sample for CQL loss computation.
        :param temp: Temperature parameter for logsumexp in CQL loss.
        :param min_q_weight: Weight for the conservative Q-learning loss term.
        :param target_action_gap: Target gap for CQL constraint when using Lagrange multiplier.
        :param batch_size: Minibatch size for optimizer.
        :param actor_lr: Learning rate for actor network.
        :param actor_optimizer: Optimizer class (or name) for actor network.
        :param actor_optimizer_kwargs: Parameter dict for actor optimizer.
        :param critic_lr: Learning rate for critic networks.
        :param critic_optimizer: Optimizer class (or str) for critic networks.
        :param critic_optimizer_kwargs: Parameter dict for critic optimizer.
        :param alpha_lr: Learning rate for entropy temperature parameter alpha.
        :param alpha_optimizer: Optimizer class (or str) for alpha parameter.
        :param alpha_optimizer_kwargs: Parameter dict for alpha optimizer.
        :param cql_alpha_lr: Learning rate for CQL regularization coefficient.
        :param cql_alpha_optimizer: Optimizer class (or str) for CQL alpha parameter.
        :param cql_alpha_optimizer_kwargs: Parameter dict for CQL alpha optimizer.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        """
        super().__init__(buffer=buffer,
                         env=None,
                         eval_env=eval_env,
                         network_type=network_type,
                         network_list=self.network_list(),
                         network_config=network_config,
                         device=device)


        self._build_network()

        self.log_alpha = torch.tensor(np.log(alpha),
                                      device=device,
                                      requires_grad=train_alpha)
        self.target_alpha = -self.action_dim if target_alpha is None else target_alpha
        self.train_alpha = train_alpha

        self.log_cql_alpha = torch.tensor(np.log(cql_alpha),
                                          device=device,
                                          requires_grad=with_lagrange)
        self.with_lagrange = with_lagrange

        self.target_critic_update = target_critic_update

        self.actor_optimizer = get_optimizer(
            self.actor.parameters(), actor_lr,
            actor_optimizer, actor_optimizer_kwargs)

        self.critic_optimizer = get_optimizer(
            list(chain(*[self.critic.parameters(),
                         self.critic2.parameters()])),
            critic_lr, critic_optimizer,
            critic_optimizer_kwargs
        )

        if self.train_alpha:
            self.alpha_optimizer = get_optimizer(
                [self.log_alpha], alpha_lr, alpha_optimizer,
                alpha_optimizer_kwargs)

        if self.with_lagrange:
            self.cql_alpha_optimizer = get_optimizer(
                [self.log_cql_alpha], cql_alpha_lr, cql_alpha_optimizer,
                cql_alpha_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_random = num_random
        self.temp = temp
        self.min_q_weight = min_q_weight
        self.target_action_gap = target_action_gap

    def __repr__(self) -> str:
        return "CQL"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Actor": GaussianActor,
                            "Critic": QNetwork
                            },
                "D2RL": {"Actor": D2RLGaussianActor,
                         "Critic": D2RLQNetwork,
                         },
                }

    def _build_network(self) -> None:
        actor_class = \
        self.network_list()[self.network_type]["Actor"]
        critic_class = \
        self.network_list()[self.network_type]["Critic"]

        actor_config = self.network_config["Actor"]
        critic_config = self.network_config["Critic"]

        # Fix GaussianActor setting for CQL
        actor_config["squash"] = True
        actor_config["independent_std"] = False

        self.actor = actor_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            device=self.device,
            **actor_config,
        ).train()

        self.critic = critic_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim, device=self.device,
            **critic_config
        ).train()
        self.target_critic = deepcopy(self.critic).eval()

        self.critic2 = critic_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim, device=self.device,
            **critic_config
        ).train()
        self.target_critic2 = deepcopy(self.critic).eval()

    def get_action(self, observation: Union[
        np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the CQL action from an observation (in training mode)

        :param observation: The input observation
        :return: The CQL agent's action
        """
        observation = fix_observation(observation,
                                      self.device)

        self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(observation)

        return action.cpu().numpy()

    def eval_action(self, observation: Union[
        np.ndarray, torch.Tensor]) -> List[float]:
        """
        Get the CQL action from an observation (in evaluation mode)

        :param observation: The input observation
        :return: The CQL agent's action (in evaluation mode)
        """
        observation = fix_observation(observation,
                                      self.device)

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

    def train(self) -> Dict[
        str, Any]:
        """
        Train CQL policy.
        :return: training results
        """
        self.training_count += 1
        result_dict = {}
        self.actor.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        # critic network training
        with torch.no_grad():
            ns_action, ns_logprob = self.actor(ns)

            target_min_aq = torch.minimum(
                self.target_critic((ns, ns_action)),
                self.target_critic2((ns, ns_action)))

            target_q = r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logprob)

        q1, q2 = self.critic((s, a)), self.critic2((s, a))
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # cql
        random_action = torch.FloatTensor(a.shape[0] * self.num_random, a.shape[-1]).uniform_(-1, 1).to(self.device)

        repeated_s = s.unsqueeze(1).repeat(1, self.num_random, 1).view(s.shape[0] * self.num_random, s.shape[1])
        repeated_ns = ns.unsqueeze(1).repeat(1, self.num_random, 1).view(ns.shape[0] * self.num_random, ns.shape[1])

        repeat_s_action, repeat_s_logprob = self.actor(repeated_s)
        repeat_ns_action, repeat_ns_logprob = self.actor(repeated_ns)

        num_repeat = int(random_action.shape[0] / s.shape[0])
        random_q, random_q2 = self.critic((repeated_s, random_action)).reshape(s.shape[0], num_repeat, 1), self.critic2((repeated_s, random_action)).reshape(s.shape[0], num_repeat, 1)
        repeat_s_q, repeat_s_q2 = self.critic((repeated_s, repeat_s_action)).reshape(s.shape[0], num_repeat, 1), self.critic2((repeated_s, repeat_s_action)).reshape(s.shape[0], num_repeat, 1)
        repeat_ns_q, repeat_ns_q2 = self.critic((repeated_ns, repeat_ns_action)).reshape(s.shape[0], num_repeat, 1), self.critic2((repeated_ns, repeat_ns_action)).reshape(s.shape[0], num_repeat, 1)

        cat_q1 = torch.cat([random_q, q1.unsqueeze(1), repeat_ns_q, repeat_s_q], 1)
        cat_q2 = torch.cat([random_q2, q2.unsqueeze(1), repeat_ns_q2, repeat_s_q2], 1)

        cql1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp - q1.mean() * self.min_q_weight
        cql2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp - q2.mean() * self.min_q_weight

        if self.with_lagrange:
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1000000.0)
            cql1_loss = cql_alpha * (cql1_loss - self.target_action_gap)
            cql2_loss = cql_alpha * (cql2_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_loss - cql2_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
            result_dict[
                "loss/cql_alpha_loss"] = cql_alpha_loss.detach().cpu().numpy()

        critic_with_cql_loss = critic_loss + cql1_loss + cql2_loss
        self.critic_optimizer.zero_grad()
        critic_with_cql_loss.backward()
        self.critic_optimizer.step()

        # actor training
        s_action, s_logprob = self.actor(s)
        min_aq_rep = torch.minimum(
            self.critic((s, s_action)),
            self.critic2((s, s_action)))

        actor_loss = (
                self.alpha * s_logprob - min_aq_rep).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha training
        if self.train_alpha:
            alpha_loss = -(self.log_alpha.exp() * (
                    s_logprob + self.target_alpha).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            result_dict[
                "loss/alpha_loss"] = alpha_loss.detach().cpu().numpy()

        # critic update
        if self.training_count % self.target_critic_update == 0:
            soft_update(self.critic, self.target_critic,
                        self.tau)
            soft_update(self.critic2, self.target_critic2,
                        self.tau)

        result_dict[
            "loss/actor_loss"] = actor_loss.detach().cpu().numpy()
        result_dict[
            "loss/critic_loss"] = critic_loss.detach().cpu().numpy()
        result_dict["loss/cql_loss"] = (cql1_loss + cql2_loss).detach().cpu().numpy()
        result_dict[
            "value/alpha"] = self.alpha.cpu().numpy()

        return result_dict
