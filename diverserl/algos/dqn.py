from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from numpy import dtype, ndarray

from diverserl.algos.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, hard_update
from diverserl.networks import DeterministicActor, PixelEncoder
from diverserl.networks.d2rl_networks import D2RLDeterministicActor
from diverserl.networks.noisy_networks import NoisyDeterministicActor


class DQN(DeepRL):
    def __init__(
        self,
            env: gym.vector.SyncVectorEnv,
            eval_env: gym.Env,
            network_type: str = "Default",
            network_config: Optional[Dict[str, Any]] = None,
            eps_initial: float = 1.0,
            eps_final: float = 0.05,
            decay_fraction: float = 0.5,
            gamma: float = 0.9,
            batch_size: int = 256,
            buffer_size: int = 10 ** 6,
            learning_rate: float = 0.001,
            optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            anneal_lr: bool = False,
            target_copy_freq: int = 10,
            training_start: int = 1000,
            max_step: int = 1000000,
            device: str = "cpu",
    ) -> None:
        """
        DQN(Deep-Q Network).

        Paper: Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013.

        :param env: Gymnasium environment to train the DQN algorithm
        :param network_type: Type of the DQN networks to be used.
        :param network_config: Configurations of the DQN networks.
        :param eps_initial: Initial probability to conduct random action during training
        :param eps_final: Final probability to conduct random action during training
        :param decay_fraction: Fraction of max_step to perform epsilon linear decay during training.
        :param gamma: The discount factor
        :param batch_size: Minibatch size for optimizer.
        :param buffer_size: Maximum length of replay buffer.
        :param learning_rate: Learning rate of the Q-network
        :param optimizer: Optimizer class (or str) for the Q-network
        :param optimizer_kwargs: Parameter dict for the optimizer
        :param anneal_lr: Whether to linearly decrease the learning rate during training.
        :param target_copy_freq: How many training step to pass to copy Q-network to target Q-network
        :param training_start: In which total_step to start the training of the Deep RL algorithm
        :param max_step: Maximum step to run the training
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(
            env=env, eval_env=eval_env, network_type=network_type, network_list=self.network_list(), network_config=network_config, device=device
        )

        assert isinstance(self.observation_space, spaces.Box) and isinstance(
            self.action_space, spaces.Discrete
        ), f"{self} supports only Box type state observation space and Discrete type action space."

        self.buffer_size = buffer_size

        self._build_network()

        self.optimizer = get_optimizer(self.q_network.parameters(), learning_rate, optimizer, optimizer_kwargs)
        self.anneal_lr = anneal_lr

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_copy_freq = target_copy_freq

        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.decay_fraction = decay_fraction
        self.training_start = training_start

        self.max_step = max_step
        self.action_count = 0

    def __repr__(self) -> str:
        return "DQN"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Q_network": DeterministicActor, "Encoder": PixelEncoder, "Buffer": ReplayBuffer},
                "Noisy": {"Q_network": NoisyDeterministicActor, "Encoder": PixelEncoder, "Buffer": ReplayBuffer},
                "D2RL": {"Q_network": D2RLDeterministicActor, "Encoder": PixelEncoder, "Buffer": ReplayBuffer},
                }

    def _build_network(self) -> None:
        if not isinstance(self.state_dim, int):
            encoder_class = self.network_list()[self.network_type]["Encoder"]
            encoder_config = self.network_config["Encoder"]
            self.encoder = encoder_class(state_dim=self.state_dim, **encoder_config)

            feature_dim = self.encoder.feature_dim

        else:
            self.encoder = None
            feature_dim = self.state_dim

        q_network_class = self.network_list()[self.network_type]["Q_network"]
        q_network_config = self.network_config["Q_network"]

        self.q_network = q_network_class(
            state_dim=feature_dim,
            action_dim=self.action_dim,
            device=self.device,
            feature_encoder=self.encoder,
            **q_network_config,
        ).train()

        self.target_q_network = deepcopy(self.q_network).eval()

        buffer_class = self.network_list()[self.network_type]["Buffer"]
        buffer_config = self.network_config["Buffer"]

        self.buffer = buffer_class(state_dim=self.state_dim, action_dim=1, device=self.device, max_size=self.buffer_size, **buffer_config)

    def update_eps(self):
        """
        linearly update the dqn exploration parameter.
        :return:
        """
        slope = (self.eps_final - self.eps_initial) / int(self.decay_fraction * self.max_step)
        self.eps = max(slope * (self.training_start + self.action_count) + self.eps_initial, self.eps_final)

        self.action_count += 1

    def get_action(self, observation: Union[np.ndarray, torch.Tensor]) -> ndarray[Any, dtype[Any]] | Any:
        """
        Get the DQN action from an observation (in training mode).

        :param observation: The input observation
        :return: The DQN agent's action
        """
        self.update_eps()

        if np.random.rand() < self.eps:
            return np.array([np.random.randint(self.action_dim)])
        else:
            observation = self._fix_observation(observation)
            with torch.no_grad():
                action = self.q_network(observation).argmax(1).cpu().numpy()

            return action

    def eval_action(self, observation: Union[np.ndarray, torch.Tensor]) -> Union[int, List[int]]:
        """
        Get the DQN action from an observation (in evaluation mode).

        :param observation: The input observation
        :return: The DQN agent's action (in evaluation mode)
        """

        observation = self._fix_observation(observation)
        observation = torch.unsqueeze(observation, dim=0)

        self.q_network.eval()
        with torch.no_grad():
            action = self.q_network(observation).argmax(1).cpu().numpy()

        return action

    def train(self, total_step: int, max_step: int) -> Dict[str, Any]:
        """
        Train the DQN policy.

        :return: Training result (loss)
        """
        if self.anneal_lr:
            self.optimizer.param_groups[0]['lr'] = (1 - total_step / max_step) * self.learning_rate

        self.training_count += 1
        self.q_network.train()

        s, a, r, ns, d, t = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_value = r + self.gamma * (1 - d) * (self.target_q_network(ns).max(1, keepdims=True)[0])

        selected_value = self.q_network(s).gather(1, a.to(torch.int64))
        loss = F.smooth_l1_loss(selected_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_count % self.target_copy_freq == 0:
            hard_update(self.q_network, self.target_q_network)

        return {"loss/loss": loss.detach().cpu().numpy(), "eps": self.eps, "learning_rate": self.optimizer.param_groups[0]['lr']}
