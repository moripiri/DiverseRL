from typing import Optional, Type

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: tuple[int] = (256, 256),
        mid_activation: Optional[str | Type[nn.Module]] = nn.ReLU,
        last_activation: Optional[str | Type[nn.Module]] = None,
        output_scale=1,
        output_bias=0,
        use_bias=True,
        device=None,
    ):
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

    def forward(self, input: torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(input, tuple):
            input = torch.cat(input, dim=1)

        return self.output_scale * self.layers(input) + self.output_bias


if __name__ == "__main__":
    print(getattr(nn, "ReLU") == nn.ReLU)
    a = MLP(5, 2, mid_activation="ReLU6", last_activation=nn.Softmax)
    print(a)
    print(a(torch.ones((1, 5))).argmax(1).item())
