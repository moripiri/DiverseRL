from typing import Any, Optional, Tuple, Type, Union

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
        mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
        last_activation: Optional[Union[str, Type[nn.Module]]] = None,
        output_scale: float = 1.0,
        output_bias: float = 0.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        """
        Multi layered perceptron (MLP), a collection of fully-connected layers each followed by an activation function.

        :param input_dim: Dimension of the input
        :param output_dim: Dimension of the output
        :param hidden_units: Size of the hidden layers in MLP
        :param mid_activation: Activation function of hidden layers
        :param last_activation: Activation function of the last MLP layer
        :param output_scale: How much to scale the output of the MLP
        :param output_bias: How much to bias the output of the MLP
        :param use_bias: Whether to use bias in linear layer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units

        self.mid_activation = getattr(nn, mid_activation) if isinstance(mid_activation, str) else mid_activation
        self.last_activation = getattr(nn, last_activation) if isinstance(last_activation, str) else last_activation

        self.output_scale = output_scale
        self.output_bias = output_bias

        self.use_bias = use_bias
        self.device = device

        layers = []
        layer_units = [input_dim, *hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=use_bias, device=device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation())

        layers.append(nn.Linear(hidden_units[-1], self.output_dim, bias=use_bias, device=device))
        if self.last_activation is not None:
            layers.append(self.last_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Return output of the MLP for the given input.

        :param input: input(1~2 torch tensor)
        :return: output (scaled and biased)
        """
        if isinstance(input, tuple):
            input = torch.cat(input, dim=1)

        return self.output_scale * self.layers(input) + self.output_bias


if __name__ == "__main__":
    print(getattr(nn, "ReLU") == nn.ReLU)
    a = MLP(5, 2, mid_activation="ReLU6", last_activation=nn.Softmax)
    print(a)
    # print(a(torch.ones((1, 5))).argmax(1).item())
