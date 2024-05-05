from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from torch import nn


class Network(ABC, nn.Module):
    def __init__(
            self,
            input_dim: Union[int, Tuple[int, int, int]],
            output_dim: int,
            mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
            mid_activation_kwargs: Optional[Dict[str, Any]] = None,
            last_activation: Optional[Union[str, Type[nn.Module]]] = None,
            last_activation_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Optional[Union[str, Callable]] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
            bias_initializer: Optional[Union[str, Callable]] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ):
        """
        Base Network

        :param input_dim: Dimension of the input
        :param output_dim: Dimension of the output
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

        self.mid_activation, self.mid_activation_kwargs = self.get_activation(mid_activation, mid_activation_kwargs)
        self.last_activation, self.last_activation_kwargs = self.get_activation(last_activation, last_activation_kwargs)

        self.kernel_initializer, self.kernel_initializer_kwargs = self.get_initializer(kernel_initializer,
                                                                                       kernel_initializer_kwargs)
        self.bias_initializer, self.bias_initializer_kwargs = self.get_initializer(bias_initializer,
                                                                                   bias_initializer_kwargs)

        self.use_bias = use_bias
        self.device = device

    @abstractmethod
    def _make_layers(self) -> None:
        pass

    @staticmethod
    def get_activation(activation: Union[str, Type[nn.Module]], activation_kwargs: Dict[str, Any]) -> Tuple[
        nn.Module, Dict[str, Any]]:
        """
        Returns activation function and activation kwargs from given arguments.

        :param activation: Name, or class of the activation function.
        :param activation_kwargs: Arguments for the activation function.
        :return:
        """
        activation = getattr(nn, activation) if isinstance(activation, str) else activation

        if activation_kwargs is None:
            activation_kwargs = {}
        else:
            for key, value in activation_kwargs.items():
                assert isinstance(value, Union[
                    int, float, bool, str]), "Value of activation_kwargs must be set as int, float, boolean or string"
                if isinstance(value, str):
                    activation_kwargs[key] = eval(value)
                else:
                    activation_kwargs[key] = value

        return activation, activation_kwargs

    @staticmethod
    def get_initializer(initializer: Union[str, Callable], initializer_kwargs: Optional[Dict[str, Any]]) -> Tuple[
        Callable, Dict[str, Any]]:
        """
        Returns initializer function and initializer kwargs from given arguments.

        :param initializer: Name or class of the initializer function.
        :param initializer_kwargs: Arguments for the initializer function.
        :return: Initializer function and initializer kwargs.
        """
        initializer = (
            getattr(nn.init, initializer) if isinstance(initializer, str) else initializer
        )
        if initializer_kwargs is None:
            initializer_kwargs = {}
        else:
            for key, value in initializer_kwargs.items():
                assert isinstance(value, Union[
                    int, float, bool, str]), "Value of initializer_kwargs must be set as int, float, boolean or string"
                if isinstance(value, str):
                    initializer_kwargs[key] = eval(value)
                else:
                    initializer_kwargs[key] = value

        return initializer, initializer_kwargs

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            self.kernel_initializer(m.weight, **self.kernel_initializer_kwargs)
            if m.bias is not None:
                self.bias_initializer(m.bias, **self.bias_initializer_kwargs)
