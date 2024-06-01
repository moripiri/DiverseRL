from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks.basic_networks import MLP


class DuelingNetwork(MLP):
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
            feature_encoder: Optional[nn.Module] = None,
            device: str = "cpu",
    ):
        """
        Dueling Network for Dueling Deep Q-Network(Dueling DQN) Algorithm.

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
        :param feature_encoder: Optional feature encoder to attach to the MLP layers.
        :param device: Device (cpu, cuda, ...) on which the code should be run
        """
        MLP.__init__(
            self,
            input_dim=state_dim,
            output_dim=action_dim,
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
        assert estimator in ['mean', 'max']
        self.estimator = estimator
        self.feature_encoder = feature_encoder

    def _make_layers(self) -> None:

        trunks = OrderedDict()
        trunk_units = [self.input_dim, *self.hidden_units]

        for i in range(len(trunk_units) - 1):
            trunks[f'linear{i}'] = nn.Linear(trunk_units[i], trunk_units[i + 1], bias=self.use_bias, device=self.device)
            if self.mid_activation is not None and i < len(trunk_units) - 2:
                trunks[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)

        value = nn.Linear(trunk_units[-1], 1, bias=self.use_bias, device=self.device)
        advantage = nn.Linear(trunk_units[-1], self.output_dim, bias=self.use_bias, device=self.device)

        self.layers = nn.ModuleDict({'trunk': nn.Sequential(trunks), 'value': value, 'advantage': advantage})
        self.layers.apply(self._init_weights)

    def forward(self, input: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        """
        Return modified value from the Dueling Network

        :param input: input tensor
        :param detach_encoder: whether to detach feature encoder from training
        :return: modified value from the Dueling Network
        """
        if self.feature_encoder is not None:
            input = self.feature_encoder(input.to(self.device))
            if detach_encoder:
                input = input.detach()

        output = self.layers['trunk'](input)

        value = self.layers['value'](output)
        advantage = self.layers['advantage'](output)

        if self.estimator == 'mean':
            return value + (advantage - advantage.mean(axis=1, keepdims=True))
        else:
            return value + (advantage - advantage.max(axis=1, keepdims=True))

if __name__ == '__main__':
    a = DuelingNetwork(5, 3)
    print(a)
    print(a(torch.randn(1, 5)))
