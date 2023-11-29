from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributions.normal import Normal

from diverserl.networks.base import Network

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(Network):
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
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )

        self.action_scale = action_scale
        self.action_bias = action_bias

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        """
        Make Gaussian Actor layers from layer dimensions and activations and initialize its weights and biases.
        """
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        self.trunks = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(self.hidden_units[-1], self.output_dim, bias=False, device=self.device)
        self.logstd_layer = nn.Linear(self.hidden_units[-1], self.output_dim, bias=False, device=self.device)

        self.trunks.apply(self._init_weights)
        self.mean_layer.apply(self._init_weights)
        torch.nn.init.constant_(self.logstd_layer.weight, val=0.0)

    def forward(self, input: Union[torch.Tensor], deterministic=False) -> torch.tensor:
        """
        Return output of the Gaussian actor for the given input.

        :param input: input(1~2 torch tensor)
        :return: output (scaled and biased)
        """
        dist = self.compute_dist(input)

        if deterministic:
            sample = dist.mean
        else:
            sample = dist.rsample()

        tanh_sample = torch.tanh(sample)
        log_prob = dist.log_prob(sample)
        log_prob -= torch.log(self.action_scale * (1 - tanh_sample.pow(2)) + 1e-10)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = self.action_scale * tanh_sample + self.action_bias

        return action, log_prob

    def compute_dist(self, input: Union[torch.Tensor]) -> torch.tensor:
        """
        Return Normal distribution of the Gaussian actor for the given input.

        :param input: input(a torch tensor)
        :return: output (scaled and biased)
        """
        input = input.to(self.device)
        trunk_output = self.trunks(input)
        output_mean = self.mean_layer(trunk_output)

        output_std = torch.clamp(self.logstd_layer(trunk_output), LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(loc=output_mean, scale=output_std)

        return dist


if __name__ == "__main__":
    a = GaussianActor(5, 2)
    print(a)
    print(a(torch.ones((1, 5))))
