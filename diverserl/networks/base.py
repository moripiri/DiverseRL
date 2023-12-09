from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn


class Network(ABC, nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Tuple[int, ...],
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
        Base Network

        :param input_dim: Dimension of the input
        :param output_dim: Dimension of the output
        :param hidden_units: Size of the hidden units in network
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for middle activation
        :param last_activation: Activation function of the last layer
        :param last_activation_kwargs: Parameters for last activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: Whether to use bias in the layer
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units

        self.mid_activation = getattr(nn, mid_activation) if isinstance(mid_activation, str) else mid_activation
        self.last_activation = getattr(nn, last_activation) if isinstance(last_activation, str) else last_activation

        self.mid_activation_kwargs = {} if mid_activation_kwargs is None else mid_activation_kwargs
        self.last_activation_kwargs = {} if last_activation_kwargs is None else last_activation_kwargs

        self.kernel_initializer = (
            getattr(nn.init, kernel_initializer) if isinstance(kernel_initializer, str) else kernel_initializer
        )
        self.bias_initializer = (
            getattr(nn.init, bias_initializer) if isinstance(bias_initializer, str) else bias_initializer
        )

        self.kernel_initializer_kwargs = {} if kernel_initializer_kwargs is None else kernel_initializer_kwargs
        self.bias_initializer_kwargs = {} if bias_initializer_kwargs is None else bias_initializer_kwargs

        self.use_bias = use_bias
        self.device = device

    @abstractmethod
    def _make_layers(self) -> None:
        pass

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
