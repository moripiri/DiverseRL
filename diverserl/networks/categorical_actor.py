from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from diverserl.networks.basic_networks import MLP


class CategoricalActor(MLP):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Tuple[int, ...] = (64, 64),
        mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
        mid_activation_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
        bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
        use_bias: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
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

    def forward(self, state: torch.Tensor, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor]:
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

        probs = self.layers(state)
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
