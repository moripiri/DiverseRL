from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import torch

from diverserl.algos.dqns import DDQN
from diverserl.common.buffer import ReplayBuffer
from diverserl.networks.dueling_network import DuelingNetwork
from diverserl.networks.noisy_networks import NoisyDuelingNetwork


class DuelingDQN(DDQN):
    def __init__(
            self,
            env: gym.vector.VectorEnv,
            eval_env: gym.vector.VectorEnv,
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
        Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015.

        :param env: Gymnasium environment to train the Dueling DQN algorithm
        :param network_type: Type of the Dueling DQN networks to be used.
        :param network_config: Configurations of the Dueling DQN networks.
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
            env=env,
            eval_env=eval_env,
            network_type=network_type,
            network_config=network_config,
            eps_initial=eps_initial,
            eps_final=eps_final,
            decay_fraction=decay_fraction,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            anneal_lr=anneal_lr,
            target_copy_freq=target_copy_freq,
            training_start=training_start,
            max_step=max_step,
            device=device,
            )

    def __repr__(self) -> str:
        return "Dueling_DQN"

    @staticmethod
    def network_list() -> Dict[str, Any]:
        return {"Default": {"Q_network": DuelingNetwork, "Buffer": ReplayBuffer},
                "Noisy": {"Q_network": NoisyDuelingNetwork, "Buffer": ReplayBuffer}}
