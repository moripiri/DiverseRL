from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks import (MLP, CategoricalActor, DeterministicActor,
                                GaussianActor, QNetwork, VNetwork)
from diverserl.networks.gaussian_actor import LOG_STD_MAX, LOG_STD_MIN


class D2RLMLP(MLP):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64, 64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            last_activation: Optional[_activation] = None,
            last_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ):
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
        Make D2RL layers from layer dimensions and activations and initialize its weights and biases.
        """

        layers = OrderedDict()
        layer_units = [self.input_dim, *self.hidden_units, self.output_dim]

        for i in range(len(layer_units) - 1):
            layers[f'linear{i}'] = nn.Linear(
                layer_units[i] + int(np.where(i > 0 and i < len(layer_units) - 2, self.input_dim, 0)), layer_units[i + 1],
                bias=self.use_bias, device=self.device)

            if self.mid_activation is not None and i < len(layer_units) - 2:
                layers[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)
            if self.last_activation is not None and i == len(layer_units) - 2:
                layers[f'activation{i}'] = self.last_activation(**self.last_activation_kwargs)

        self.layers = nn.ModuleDict(layers)
        self.layers.apply(self._init_weights)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Return output of the MLP for the given input.

        :param input: input(1~2 torch tensor)
        :return: output (scaled and biased)
        """
        if isinstance(input, tuple):
            input = torch.cat(input, dim=1)

        output = input.to(self.device)
        concat_names = [f'linear{i}' for i in range(1, len(self.hidden_units))]

        for name, layer in self.layers.items():
            if name in concat_names:
                output = torch.cat([output, input.to(self.device)], dim=-1)

            output = layer(output)

        return output


class D2RLDeterministicActor(D2RLMLP, DeterministicActor):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64, 64, 64),
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
            device: str = "cpu",
    ):
        D2RLMLP.__init__(
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
            device=device,
        )

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return output of the Deterministic Actor for the given input.

        :param input: state tensor
        :return: action tensor
        """

        output = D2RLMLP.forward(self, input)

        return self.action_scale * output + self.action_bias


class D2RLQNetwork(D2RLMLP, QNetwork):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ):
        D2RLMLP.__init__(
            self,
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

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Return Q-value for the given input.

        :param input: state and action tensors
        :return: Q-value
        """
        output = D2RLMLP.forward(self, input)

        return output


class D2RLVNetwork(D2RLMLP, VNetwork):
    def __init__(
            self,
            state_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ):
        D2RLMLP.__init__(
            self,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return value for the given input.

        :param input: state tensor
        :return: Value for the given input
        """
        output = D2RLMLP.forward(self, input)

        return output


class D2RLCategoricalActor(D2RLMLP, CategoricalActor):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: Tuple[int, ...] = (64, 64, 64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            use_bias: bool = True,
            device: str = "cpu",
    ) -> None:
        D2RLMLP.__init__(
            self,
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            last_activation=nn.Softmax,
            last_activation_kwargs={"dim": -1},
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        return CategoricalActor.forward(self, state, deterministic)

    def compute_dist(self, state: torch.Tensor) -> Categorical:
        """
        Return Categorical distribution of the Categorical actor for the given state.

        :param state: state(a torch tensor)
        :return: Categorical distribution
        """
        state = state.to(self.device)

        probs = D2RLMLP.forward(self, state)
        dist = Categorical(probs=probs)

        return dist


class D2RLGaussianActor(D2RLMLP, GaussianActor):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            squash: bool = True,
            independent_std: bool = False,
            logstd_init: float = 0.0,
            hidden_units: Tuple[int, ...] = (64, 64, 64, 64),
            mid_activation: Optional[_activation] = nn.ReLU,
            mid_activation_kwargs: Optional[_kwargs] = None,
            kernel_initializer: Optional[_initializer] = nn.init.orthogonal_,
            kernel_initializer_kwargs: Optional[_kwargs] = None,
            bias_initializer: Optional[_initializer] = nn.init.zeros_,
            bias_initializer_kwargs: Optional[_kwargs] = None,
            action_scale: float = 1.0,
            action_bias: float = 0.0,
            use_bias: bool = True,
            device: str = "cpu",
            ):

        self.squash = squash
        self.independent_std = independent_std
        assert not (self.independent_std and self.squash)

        self.logstd_init = logstd_init
        self.action_scale = action_scale
        self.action_bias = action_bias

        D2RLMLP.__init__(self,
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

        if self.independent_std:
            self.std = torch.tensor(self.logstd_init, device=self.device).exp()


    def _make_layers(self) -> None:
        """
        Make D2RL Gaussian Actor layers from layer dimensions and activations and initialize its weights and biases.
        """
        trunks = OrderedDict()
        trunk_units = [self.input_dim, *self.hidden_units]

        for i in range(len(trunk_units) - 1):
            trunks[f'linear{i}'] = nn.Linear(trunk_units[i] + int(np.where(i > 0 and i < len(trunk_units) - 1, self.input_dim, 0)),
                                             trunk_units[i + 1], bias=self.use_bias, device=self.device)

            if self.mid_activation is not None and i < len(trunk_units) - 2:
                trunks[f'activation{i}'] = self.mid_activation(**self.mid_activation_kwargs)

        mean_layer = nn.Linear(trunk_units[-1], self.output_dim, bias=False, device=self.device)

        self.layers = nn.ModuleDict({"trunk": nn.ModuleDict(trunks), "mean": mean_layer})
        self.layers.apply(self._init_weights)

        if not self.independent_std:
            logstd_layer = nn.Linear(trunk_units[-1], self.output_dim, bias=False, device=self.device)
            torch.nn.init.constant_(logstd_layer.weight, val=self.logstd_init)
            self.layers['logstd'] = logstd_layer


    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:

        return GaussianActor.forward(self, state, deterministic)


    def compute_dist(self, state: torch.Tensor) -> Normal:
        """
        Return Normal distribution of the Gaussian actor for the given state.

        :param state: state(a torch tensor)
        :return: Normal distribution
        """
        state = state.to(self.device)


        trunk_output = state.to(self.device)
        concat_names = [f'linear{i}' for i in range(1, len(self.hidden_units))]

        trunk = self.layers['trunk']
        assert isinstance(trunk, nn.ModuleDict)
        for name, layer in trunk.items():
            if name in concat_names:
                trunk_output = torch.cat([trunk_output, state.to(self.device)], dim=-1)
            trunk_output = layer(trunk_output)

        output_mean = self.layers['mean'](trunk_output)

        if self.independent_std:
            output_std = self.std.expand_as(output_mean)
        else:
            output_std = torch.clamp(self.layers['logstd'](trunk_output), LOG_STD_MIN, LOG_STD_MAX).exp()

        dist = Normal(loc=output_mean, scale=output_std)

        return dist


if __name__ == '__main__':
    # a = D2RLVNetwork(5, hidden_units=(64, 64, 64))
    # print(isinstance(a, VNetwork))
    # print(a)
    # print(a(torch.ones(2, 5)))
    # for name, layer in a.layers.items():
    #     print(name, layer)
    a = D2RLGaussianActor(5, 3)
    print(a)
    print(a(torch.ones(2, 5)))
    print(a.entropy(torch.ones(2, 5)))
