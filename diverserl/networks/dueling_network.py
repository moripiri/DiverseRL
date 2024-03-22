from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.networks.basic_networks import MLP


class DuelingNetwork(MLP):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            estimator: str = 'mean',
            mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
            mid_activation_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
            bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
            use_bias: bool = True,
            feature_encoder: Optional[nn.Module] = None,
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
        assert estimator in ['mean', 'max']
        self.estimator = estimator
        self.feature_encoder = feature_encoder

    def _make_layers(self) -> None:
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        self.layers = nn.Sequential(*layers)

        self.value = nn.Linear(layer_units[-1], 1, bias=self.use_bias, device=self.device)
        self.advantage = nn.Linear(layer_units[-1], self.output_dim, bias=self.use_bias, device=self.device)

        self.layers.apply(self._init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))

        output = self.layers(input)

        value = self.value(output)
        advantage = self.advantage(output)

        if self.estimator == 'mean':
            return value + (advantage - advantage.mean(axis=1, keepdims=True))
        else:
            return value + (advantage - advantage.max(axis=1, keepdims=True))
