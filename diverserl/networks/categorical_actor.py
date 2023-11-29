from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from diverserl.networks.base import Network


class CategoricalActor(Network):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
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

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        """
        Make Categorical Actor layers from layer dimensions and activations and initialize its weights and biases.
        """
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        layers.append(nn.Linear(self.hidden_units[-1], self.output_dim, bias=self.use_bias, device=self.device))
        layers.append(self.last_activation(**self.last_activation_kwargs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)

    def forward(self, input: torch.Tensor, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return output of the Categorical Actor for the given input.

        :param input: input(1 torch tensor)
        :return: output
        """
        input = input.to(self.device)

        probs = self.layers(input)
        self.dist = Categorical(probs)

        if deterministic:
            action = self.dist.logits.argmax(axis=-1)
        else:
            action = self.dist.sample()

        log_prob = self.dist.log_prob(action)

        return action, log_prob


if __name__ == "__main__":
    a = CategoricalActor(5, 3, device='cuda')
    print(a)
    print(a(torch.ones((1, 5)), deterministic=True))
