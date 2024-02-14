from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.common.type_aliases import (_int_or_tuple_any_int,
                                           _layers_size_any_int)
from diverserl.networks.base import Network


class PixelEncoder(Network):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        feature_dim: int = 50,
        layer_num: int = 3,
        channel_num: _int_or_tuple_any_int = (32, 64, 64),
        kernel_size: _int_or_tuple_any_int = (8, 4, 3),
        strides: Optional[_layers_size_any_int] = (4, 2, 1),
        mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
        mid_activation_kwargs: Optional[Dict[str, Any]] = None,
        last_activation: Optional[Union[str, Type[nn.Module]]] = None,
        last_activation_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
        bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=state_dim,
            output_dim=feature_dim,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.correct = (
            lambda x: [x for _ in range(self.layer_num)] if (isinstance(x, int) or len(x) != self.layer_num) else x
        )

        self.channel_num = self.correct(channel_num)
        self.kernel_size = self.correct(kernel_size)
        self.strides = self.correct(strides)

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        layers = []
        layer_units = [self.input_dim[0], *self.channel_num]

        for i in range(self.layer_num):
            layers.append(
                nn.Conv2d(
                    in_channels=layer_units[i],
                    out_channels=layer_units[i + 1],
                    kernel_size=self.kernel_size[i],
                    stride=self.strides[i],
                )
            )
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        with torch.no_grad():
            flatten_dim = self.layers(torch.randn(1, *self.input_dim)).shape[1]

        self.layers.append(nn.Linear(flatten_dim, self.output_dim, bias=self.use_bias, device=self.device))
        if self.last_activation is not None:
            self.layers.append(self.last_activation(**self.last_activation_kwargs))

        self.layers.apply(self._init_weights)

    def forward(self, input):
        input = input.to(self.device)
        output = self.layers(input)

        return output
