from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks.utils import get_activation, get_initializer


class MLP(ABC, nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[Union[_kwargs]] = None,
            last_activation: Optional[_activation] = None,
            last_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
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
        :param use_bias: Whether to use bias in linear layer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_units = hidden_units

        self.mid_activation, self.mid_activation_kwargs = get_activation(mid_activation, mid_activation_kwargs)
        self.last_activation, self.last_activation_kwargs = get_activation(last_activation, last_activation_kwargs)

        self.kernel_initializer, self.kernel_initializer_kwargs = get_initializer(kernel_initializer,
                                                                                  kernel_initializer_kwargs)
        self.bias_initializer, self.bias_initializer_kwargs = get_initializer(bias_initializer,
                                                                              bias_initializer_kwargs)

        self.use_bias = use_bias
        self.device = device

        self._make_layers()
        self.to(torch.device(device))

    def _make_layers(self) -> None:
        """
        Make MLP layers from layer dimensions and activations and initialize its weights and biases.
        """

        layers = OrderedDict()
        layer_units = [self.input_dim, *self.hidden_units, self.output_dim]

        for i in range(len(layer_units) - 1):
            layers[f'linear{i}'] = nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device)
            if self.mid_activation is not None and i < len(layer_units) - 2:
                layers[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)
            if self.last_activation is not None and i == len(layer_units) - 2:
                layers[f'activation{i}'] = self.last_activation(**self.last_activation_kwargs)

        self.layers = nn.Sequential(layers)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        """
        if isinstance(m, nn.Linear):
            self.kernel_initializer(m.weight, **self.kernel_initializer_kwargs)
            if m.bias is not None:
                self.bias_initializer(m.bias, **self.bias_initializer_kwargs)

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
            hidden_units: Tuple[int, ...] = (64, 64),
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
        """
        Deterministic Actor class for Deep Reinforcement Learning.

        :param state_dim: Dimension of the state
        :param action_dim: Dimension of the action
        :param hidden_units: Size of the hidden layers in Deterministic Actor
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
        :param last_activation: Activation function of the last layer
        :param last_activation_kwargs: Parameters for the last activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param action_scale: How much to scale the output action
        :param action_bias: How much to bias the output action
        :param use_bias: Whether to use bias in linear layer
        :param feature_encoder: Optional feature encoder to attach to the MLP layers.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
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

    def forward(self, input: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        """
        Return output of the Deterministic Actor for the given input.

        :param input: state tensor
        :param detach_encoder: whether to detach the encoder from training
        :return: action tensor
        """
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))
            if detach_encoder:
                input = input.detach()

        output = super().forward(input)

        return self.action_scale * output + self.action_bias


class QNetwork(MLP):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
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
        """
        Q-Network class for Deep Reinforcement Learning.

        :param state_dim: Dimension of the state
        :param action_dim: Dimension of the action
        :param hidden_units: Size of the hidden layers in Q-network
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: Whether to use bias in linear layer
        :param feature_encoder: Optional feature encoder to attach to the MLP layers.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
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

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor], detach_encoder: bool = False) -> torch.Tensor:
        """
        Return Q-value for the given input.

        :param input: state and action tensors
        :param detach_encoder: whether to detach the encoder from training
        :return: Q-value
        """
        if self.feature_encoder is not None:
            feature = self.feature_encoder(input[0].to(self.device))

            if detach_encoder:
                feature = feature.detach()

            input = (feature, input[1])

        return super().forward(input)


class VNetwork(MLP):
    def __init__(
            self,
            state_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
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
        """
        V-Network class for Deep Reinforcement Learning.

        :param state_dim: Dimension of the state
        :param hidden_units: Size of the hidden layers in Value network
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: Whether to use bias in linear layer
        :param feature_encoder: Optional feature encoder to attach to the MLP layers.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
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

    def forward(self, input: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        """
        Return value for the given input.

        :param input: state tensor
        :param detach_encoder: whether to detach the encoder from training
        :return: Value for the given input
        """
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))
            if detach_encoder:
                input = input.detach()

        output = super().forward(input)

        return output


if __name__ == "__main__":
    print(getattr(nn, "ReLU") == nn.ReLU)
    a = MLP(5, 2, hidden_units=(64, 128, 64), last_activation='Softmax', kernel_initializer="orthogonal_",
            bias_initializer="zeros_", bias_initializer_kwargs={})
    print(a)
    print(a.kernel_initializer, a.kernel_initializer_kwargs)
