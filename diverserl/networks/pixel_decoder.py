from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import nn

from diverserl.common.type_aliases import (_activation, _initializer,
                                           _int_or_tuple_any_int, _kwargs,
                                           _layers_size_any_int)
from diverserl.networks.utils import get_activation, get_initializer

OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelDecoder(nn.Module):
    def __init__(
            self,
            state_dim: Tuple[int, int, int],
            feature_dim: int = 50,
            layer_num: int = 4,
            channel_num: _int_or_tuple_any_int = (32, 32, 32, 32),
            kernel_size: _int_or_tuple_any_int = (3, 3, 3, 3),
            strides: Optional[_layers_size_any_int] = (1, 1, 1, 2),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            last_activation: Optional[_activation] = None,
            last_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ):
        nn.Module.__init__(self)
        assert state_dim[1:] == (84, 84), "Currently Only 84 * 84 image is supported."

        self.input_dim = state_dim
        self.output_dim = feature_dim
        self.feature_dim = feature_dim

        self.mid_activation, self.mid_activation_kwargs = get_activation(mid_activation, mid_activation_kwargs)
        self.last_activation, self.last_activation_kwargs = get_activation(last_activation, last_activation_kwargs)

        self.kernel_initializer, self.kernel_initializer_kwargs = get_initializer(kernel_initializer,
                                                                                  kernel_initializer_kwargs)
        self.bias_initializer, self.bias_initializer_kwargs = get_initializer(bias_initializer,
                                                                              bias_initializer_kwargs)
        self.use_bias = use_bias
        self.device = device

        self.layer_num = layer_num
        self.out_dim = OUT_DIM[layer_num]

        correct = (
            lambda x: [x for _ in range(self.layer_num)] if (isinstance(x, int) or len(x) != self.layer_num) else x
        )
        self.channel_num = correct(channel_num)
        self.kernel_size = correct(kernel_size)
        self.strides = correct(strides)

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        layers = OrderedDict()
        layer_units = [*self.channel_num]
        layers['linear'] = nn.Linear(self.feature_dim, layer_units[0] * self.out_dim * self.out_dim, bias=self.use_bias,
                                     device=self.device)

        if self.mid_activation is not None:
            layers[f'activation0'] = self.mid_activation(**self.mid_activation_kwargs)

        layers['unflatten'] = nn.Unflatten(1, (layer_units[0], self.out_dim, self.out_dim))

        for i in range(len(layer_units) - 1):
            layers[f'convtranspose2d{i}'] = nn.ConvTranspose2d(
                in_channels=layer_units[i],
                out_channels=layer_units[i + 1],
                kernel_size=self.kernel_size[i],
                stride=self.strides[i],
            )

            if self.mid_activation is not None:
                layers[f'activation{i + 1}'] = self.mid_activation(**self.mid_activation_kwargs)

        layers[f'convtranspose2d{len(layer_units)}'] = nn.ConvTranspose2d(
            in_channels=layer_units[-1],
            out_channels=self.input_dim[0],
            kernel_size=self.kernel_size[-1],
            stride=self.strides[-1],
            output_padding=1,
        )

        if self.last_activation is not None:
            layers[f'activation{len(layer_units)}'] = self.last_activation(**self.last_activation_kwargs)

        self.layers = nn.Sequential(layers)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.kernel_initializer is not None:
                self.kernel_initializer(m.weight, **self.kernel_initializer_kwargs)
            if m.bias is not None and self.bias_initializer is not None:
                self.bias_initializer(m.bias, **self.bias_initializer_kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return feature of the given input.

        :param input: input image tensor (B, C, H, W)
        :return: feature tensor
        """
        input = input.to(self.device)
        output = self.layers(input)

        return output


if __name__ == '__main__':
    a = PixelDecoder((3, 84, 84))
    print(a)
    print(a(torch.zeros(1, 50)).shape)
