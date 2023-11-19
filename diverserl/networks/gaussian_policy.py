from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.networks import MLP


class GaussianPolicy(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
        independent_std: bool = False,
        mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
        mid_activation_kwargs: Optional[Dict[str, Any]] = None,
        last_activation: Optional[Union[str, Type[nn.Module]]] = None,
        last_activation_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
        bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
        output_scale: float = 1.0,
        output_bias: float = 0.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            output_scale=output_scale,
            output_bias=output_bias,
            use_bias=use_bias,
            device=device,
        )

        self.independent_std = independent_std

    def _make_layers(self) -> None:
        """
        Make MLP layers from layer dimensions and activations and initialize its weights and biases.
        """
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        if self.independent_std:
            pass
        else:
            layers.append(nn.Linear(self.hidden_units[-1], 2 * self.output_dim, bias=self.use_bias, device=self.device))

        if self.last_activation is not None:
            layers.append(self.last_activation(**self.last_activation_kwargs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        :return:
        """
        if isinstance(m, nn.Linear):
            self.kernel_initializer(m.weight, **self.kernel_initializer_kwargs)
            if m.bias is not None:
                self.bias_initializer(m.bias, **self.bias_initializer_kwargs)
