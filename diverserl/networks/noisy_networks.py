import math
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks.basic_networks import MLP
from diverserl.networks.dueling_network import DuelingNetwork


class NoisyMLP(MLP):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            last_activation: Optional[_activation] = None,
            last_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            std_init: float = 0.5,
            noise_type: str = "factorized",
            device: str = "cpu",
    ):
        """
        Noisy Networks for Exploration, Fortunato et al, 2019.

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
        :param std_init: Initial standard deviation of the NoisyLinear layer.
        :param noise_type: Type of NoisyLinear noise proposed in NoisyNet
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        self.std_init = std_init
        self.noise_type = noise_type

        MLP.__init__(
            self,
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
        layer_units = [self.input_dim, *self.hidden_units, self.output_dim]
        layers = OrderedDict()

        for i in range(len(layer_units) - 1):
            layers[f'noisylinear{i}'] = NoisyLinear(layer_units[i], layer_units[i + 1], bias=self.use_bias, std_init=self.std_init,
                                      noise_type=self.noise_type, device=self.device)

            if self.mid_activation is not None and i < len(layer_units) - 2:
                layers[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)
            if self.last_activation is not None and i == len(layer_units) - 2:
                layers[f'activation{i}'] = self.last_activation(**self.last_activation_kwargs)

        self.layers = nn.ModuleDict(layers)
        self.layers.apply(self._init_weights)


class NoisyDeterministicActor(NoisyMLP):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            last_activation: Optional[_activation] = None,
            last_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            action_scale: float = 1.0,
            action_bias: float = 0.0,
            use_bias: bool = True,
            std_init: float = 0.5,
            noise_type: str = 'factorized',
            device: str = "cpu",
    ):
        """
        Deterministic Actor with NoisyNet layers.

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
        :param std_init: Initial standard deviation of the NoisyLinear layer.
        :param noise_type: Type of NoisyLinear noise proposed in NoisyNet
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        NoisyMLP.__init__(
            self,
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

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return output of the NoisyNet Deterministic Actor for the given input.

        :param input: state tensor
        :return: action tensor
        """

        output = NoisyMLP.forward(self, input)

        return self.action_scale * output + self.action_bias


class NoisyDuelingNetwork(DuelingNetwork):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            estimator: str = 'mean',
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            std_init: float = 0.5,
            noise_type: str = 'factorized',
            device: str = "cpu",
    ):
        """
        Dueling Network with NoisyNet layers.

        :param state_dim: Dimension of the state
        :param action_dim: Dimension of the action
        :param hidden_units: Size of the hidden layers in CategoricalActor
        :param estimator: Type of the advantage estimator.
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
        :param kernel_initializer: Kernel initializer function for the network layers
        :param kernel_initializer_kwargs: Parameters for the kernel initializer
        :param bias_initializer: Bias initializer function for the network bias
        :param bias_initializer_kwargs: Parameters for the bias initializer
        :param use_bias: Whether to use bias in linear layer
        :param std_init: Initial standard deviation of the NoisyLinear layer.
        :param noise_type: Type of NoisyLinear noise proposed in NoisyNet
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        self.std_init = std_init
        self.noise_type = noise_type

        DuelingNetwork.__init__(
            self,
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
            device=device,
        )

    def _make_layers(self) -> None:

        trunks = OrderedDict()
        trunk_units = [self.input_dim, *self.hidden_units]

        for i in range(len(trunk_units) - 1):
            trunks[f'noisylinear{i}'] = NoisyLinear(trunk_units[i], trunk_units[i + 1], std_init=self.std_init,
                                      noise_type=self.noise_type, bias=self.use_bias, device=self.device)
            if self.mid_activation is not None and i < len(trunk_units) - 2:
                trunks[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)

        value = NoisyLinear(trunk_units[-1], 1, bias=self.use_bias, std_init=self.std_init,
                                 noise_type=self.noise_type, device=self.device)
        advantage = NoisyLinear(trunk_units[-1], self.output_dim, std_init=self.std_init,
                                     noise_type=self.noise_type, bias=self.use_bias, device=self.device)

        self.layers = nn.ModuleDict({'trunk': nn.Sequential(trunks), 'value': value, 'advantage': advantage})
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
        """
        Linear layer for NoisyNet.

        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param bias: If set to False, the layer will not learn an additive bias. Default: True
        :param device: Device (cpu, cuda, ...) on which the code should be run
        :param dtype: Data type (float16, float32, int8, int16, int32, int64, uint8, uint16) of the layer
        :param std_init: Initial standard deviation value for the weight initialization.
        :param noise_type: Type of noise proposed in NoisyNet.
        """
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
        """
        Reset NoisyLinear weight parameters.
        """
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
        """
        Reset NoisyLinear noise parameters.
        """
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
        """
        Returns the weight of the noisy layer.
        :return: weight
        """
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """
        Returns the bias of the noisy layer.
        :return: bias
        """
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
    a = NoisyDeterministicActor(5, 2, last_activation='Softmax', last_activation_kwargs={'dim': -1})
    print(a)
    print(a(torch.randn(1, 5)))
