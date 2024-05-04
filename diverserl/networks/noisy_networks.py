import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn

from diverserl.networks.basic_networks import MLP, DeterministicActor
from diverserl.networks.dueling_network import DuelingNetwork


class NoisyMLP(MLP):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
            mid_activation_kwargs: Optional[Dict[str, Any]] = None,
            last_activation: Optional[Union[str, Type[nn.Module]]] = None,
            last_activation_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
            bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
            use_bias: bool = True,
            std_init: float = 0.5,
            noise_type: str = "factorized",
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
        self.std_init = std_init
        self.noise_type = noise_type

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
            use_bias=use_bias,
            device=device,
        )

    def _make_layers(self) -> None:
        """
        Make MLP Noisy layers from layer dimensions and activations and initialize its weights and biases.
        """
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(NoisyLinear(layer_units[i], layer_units[i + 1], bias=self.use_bias, std_init=self.std_init,
                                      noise_type=self.noise_type, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        layers.append(NoisyLinear(layer_units[-1], self.output_dim, bias=self.use_bias, std_init=self.std_init,
                                  noise_type=self.noise_type, device=self.device))
        if self.last_activation is not None:
            layers.append(self.last_activation(**self.last_activation_kwargs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)


class NoisyDeterministicActor(NoisyMLP):
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
            std_init: float = 0.5,
            noise_type: str = 'factorized',
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
                          std_init=std_init,
                          noise_type=noise_type,
                          device=device,
                          )
        self.feature_encoder = feature_encoder

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, input: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))
            if detach_encoder:
                input = input.detach()

        output = super().forward(input)

        return self.action_scale * output + self.action_bias

class NoisyDuelingNetwork(DuelingNetwork):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            estimator: str = 'mean',
            mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
            mid_activation_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[Dict[str, Any]] = None,
            bias_initializer: Optional[Union[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[Dict[str, Any]] = None,
            use_bias: bool = True,
            std_init: float = 0.5,
            noise_type: str = 'factorized',
            feature_encoder: Optional[nn.Module] = None,
            device: str = "cpu",
    ):
        self.std_init = std_init
        self.noise_type = noise_type

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            estimator=estimator,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            feature_encoder=feature_encoder,
            device=device,
        )

    def _make_layers(self) -> None:
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(NoisyLinear(layer_units[i], layer_units[i + 1], std_init=self.std_init,
                                      noise_type=self.noise_type, bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        self.trunk = nn.Sequential(*layers)

        self.value = NoisyLinear(layer_units[-1], 1, bias=self.use_bias, std_init=self.std_init,
                                 noise_type=self.noise_type, device=self.device)
        self.advantage = NoisyLinear(layer_units[-1], self.output_dim, std_init=self.std_init,
                                     noise_type=self.noise_type, bias=self.use_bias, device=self.device)

        self.layers = nn.ModuleDict({"trunk": self.trunk, "value": self.value, "advantage": self.advantage})
        self.layers.apply(self._init_weights)


class NoisyLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device: str = 'cpu',
            dtype: Optional[torch.dtype] = None,
            std_init: float = 0.05,
            noise_type: str = 'factorized'
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init
        self.noise_type = noise_type

        assert noise_type in ['factorized', 'independent'], "Noise type must be either 'factorized' or 'independent'."

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()

        self.reset_noise()

    def reset_parameters(self) -> None:
        if self.noise_type == 'factorized':
            mu_range = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))  # in paper, std_init is 0.5
            if self.bias_mu is not None:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        else:
            mu_range = math.sqrt(3 / self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)  # in paper, std_init was 0.017
            if self.bias_mu is not None:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def reset_noise(self) -> None:
        if self.noise_type == 'factorized':
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            if self.bias_mu is not None:
                self.bias_epsilon.copy_(epsilon_out)
        else:
            self.weight_epsilon.copy_(self._scale_noise((self.out_features, self.in_features)))

            if self.bias_mu is not None:
                self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)

        if self.noise_type == 'factorized':
            return x.sign().mul_(x.abs().sqrt_())
        else:
            return x

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, noise_type={self.noise_type}'


if __name__ == '__main__':
    # print(NoisyDeterministicActor.mro())
    a = NoisyDeterministicActor(5, 2)
    # print(a.mro())
    print(a(torch.randn(1, 5)))
