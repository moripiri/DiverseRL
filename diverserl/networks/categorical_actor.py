from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks.basic_networks import MLP


class CategoricalActor(MLP):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Tuple[int, ...] = (64, 64),
        mid_activation: Optional[_activation] = nn.ReLU,
        mid_activation_kwargs: Optional[_kwargs] = None,
        kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[_kwargs] = None,
        bias_initializer: Optional[_initializer] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[_kwargs] = None,
        use_bias: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Actor that generates stochastic actions from categorical distributions.

        :param state_dim: Dimension of the state
        :param action_dim: Dimension of the action
        :param hidden_units: Size of the hidden layers in CategoricalActor
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: Whether to use bias in linear layer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        MLP.__init__(
            self,
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            last_activation=nn.Softmax,
            last_activation_kwargs={"dim": -1},
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return output of the Categorical Actor for the given state.

        :param state: state (1 torch tensor)
        :param deterministic: whether to sample action from the computed distribution.
        :return: output
        """

        dist = self.compute_dist(state)

        if deterministic:
            action = dist.logits.argmax(axis=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def compute_dist(self, state: torch.Tensor) -> Categorical:
        """
        Return Categorical distribution of the Categorical actor for the given state.

        :param state: state(a torch tensor)
        :return: Categorical distribution
        """
        state = state.to(self.device)

        probs = MLP.forward(self, state)
        dist = Categorical(probs=probs)

        return dist

    def prob(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return probability of the Categorical actor for the given state.

        :param state: state(a torch tensor)
        :return: prob
        """
        dist = self.compute_dist(state)

        return dist.logits

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Return log_probability of the Categorical actor for the given state.

        :param state: state(a torch tensor)
        :param action: wanted action to calculate its log_probability

        :return: log_prob
        """

        dist = self.compute_dist(state)

        return dist.log_prob(torch.squeeze(action)).reshape(len(state), -1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return entropy of the Categorical actor for the given state.

        :param state: state(a torch tensor)

        :return: entropy
        """
        dist = self.compute_dist(state)
        return dist.entropy().sum(dim=-1)


if __name__ == "__main__":
    a = CategoricalActor(5, 3)
    print(a)
    print(a(torch.ones((1, 5)), deterministic=True))
