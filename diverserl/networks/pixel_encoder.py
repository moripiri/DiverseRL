from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import nn

from diverserl.common.type_aliases import (_activation, _initializer,
                                           _int_or_tuple_any_int, _kwargs,
                                           _layers_size_any_int)
from diverserl.networks.utils import get_activation, get_initializer


class PixelEncoder(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        feature_dim: int = 512,
        layer_num: int = 3,
        channel_num: _int_or_tuple_any_int = (32, 64, 64),
        kernel_size: _int_or_tuple_any_int = (8, 4, 3),
        strides: Optional[_layers_size_any_int] = (4, 2, 1),
        mid_activation: Optional[_activation] = nn.ReLU,
        mid_activation_kwargs: Optional[_kwargs] = None,
        last_activation: Optional[_activation] = nn.ReLU,
        last_activation_kwargs: Optional[_kwargs] = None,
        kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[_kwargs] = None,
        bias_initializer: Optional[_initializer] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[_kwargs] = None,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        """
        Encoder to compress the input image into a latent space representation.

        :param state_dim: state(image) dimension
        :param feature_dim: feature dimension
        :param layer_num: number of layers
        :param channel_num: number of channels of the convolutional layers
        :param kernel_size: kernel sizes of the convolutional layers
        :param strides: strides of the convolutional layers
        :param mid_activation: activation function of the hidden convolutional layers
        :param mid_activation_kwargs: keyword arguments of the middle activation function
        :param last_activation: activation function of the last layer
        :param last_activation_kwargs: keyword arguments of the last activation function
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: whether to use bias in the network layers
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        # Todo: add padding and other conv2d settings
        assert last_activation is not None

        nn.Module.__init__(self)

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

        correct = (
            lambda x: [x for _ in range(self.layer_num)] if (isinstance(x, int) or len(x) != self.layer_num) else x
        )
        self.channel_num = correct(channel_num)
        self.kernel_size = correct(kernel_size)
        self.strides = correct(strides)

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        layers = []
        layers = OrderedDict()
        layer_units = [self.input_dim[0], *self.channel_num]

        for i in range(len(layer_units) - 1):
            layers[f'conv{i}'] = nn.Conv2d(
                    in_channels=layer_units[i],
                    out_channels=layer_units[i + 1],
                    kernel_size=self.kernel_size[i],
                    stride=self.strides[i],
                )

            if self.mid_activation is not None:
                layers[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)

        layers['flatten'] = nn.Flatten()

        with torch.no_grad():
            temp = nn.Sequential(layers)
            flatten_dim = temp(torch.zeros(1, *self.input_dim)).shape[1]

        layers['linear'] = nn.Linear(flatten_dim, self.output_dim, bias=self.use_bias, device=self.device)

        if self.last_activation is not None:
            layers[f'activation{len(layer_units) - 1}'] = self.last_activation(**self.last_activation_kwargs)

        self.layers = nn.Sequential(layers)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            self.kernel_initializer(m.weight, **self.kernel_initializer_kwargs)
            if m.bias is not None:
                self.bias_initializer(m.bias, **self.bias_initializer_kwargs)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Return feature of the given input.

        :param input: input image tensor (B, C, H, W)
        :return: feature tensor
        """
        input = input.to(self.device)
        output = self.layers(input)

        return output
