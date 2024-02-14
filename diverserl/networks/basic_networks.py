from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.networks.base import Network


class MLP(Network):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
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
        """
        Multi layered perceptron (MLP), a collection of fully-connected layers each followed by an activation function.

        :param input_dim: Dimension of the input
        :param output_dim: Dimension of the output
        :param hidden_units: Size of the hidden layers in MLP
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for middle activation
        :param last_activation: Activation function of the last MLP layer
        :param last_activation_kwargs: Parameters for last activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param output_scale: How much to scale the output of the MLP
        :param output_bias: How much to bias the output of the MLP
        :param use_bias: Whether to use bias in linear layer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
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
        self.hidden_units = hidden_units

        self._make_layers()
        self.to(torch.device(device))

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

        layers.append(nn.Linear(self.hidden_units[-1], self.output_dim, bias=self.use_bias, device=self.device))
        if self.last_activation is not None:
            layers.append(self.last_activation(**self.last_activation_kwargs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Return output of the MLP for the given input.

        :param input: input(1~2 torch tensor)
        :return: output (scaled and biased)
        """
        if isinstance(input, tuple):
            input = torch.cat(input, dim=1)

        input = input.to(self.device)
        output = self.layers(input)

        return output


class DeterministicActor(MLP):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
        mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
        mid_activation_kwargs: Optional[Dict[str, Any]] = None,
        last_activation: Optional[Union[str, Type[nn.Module]]] = None,
        last_activation_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
        kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
        bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
        bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
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
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )
        self.feature_encoder = feature_encoder

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))

        output = super().forward(input)

        return self.action_scale * output + self.action_bias


class QNetwork(MLP):
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
        feature_encoder: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=state_dim + action_dim,
            output_dim=1,
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
        self.feature_encoder = feature_encoder

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.feature_encoder is not None:
            input = (self.feature_encoder(input[0].to(self.device)), input[1])

        return super().forward(input)


class VNetwork(MLP):
    def __init__(
        self,
        state_dim: int,
        hidden_units: Tuple[int, ...] = (256, 256),
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
            output_dim=1,
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
        self.feature_encoder = feature_encoder

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))

        output = super().forward(input)

        return output


if __name__ == "__main__":
    print(getattr(nn, "ReLU") == nn.ReLU)
    a = QNetwork(5, 2, kernel_initializer="orthogonal_", bias_initializer="zeros_", bias_initializer_kwargs={})
    print(a)
