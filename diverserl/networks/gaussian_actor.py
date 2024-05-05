from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributions.normal import Normal

from diverserl.networks.basic_networks import MLP

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(MLP):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            squash: bool = True,
            independent_std: bool = False,
            logstd_init: float = 0.0,
            hidden_units: Tuple[int, ...] = (64, 64),
            mid_activation: Optional[Union[str, Type[nn.Module]]] = nn.ReLU,
            mid_activation_kwargs: Optional[Dict[str, Any]] = None,
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
        Actor that generates stochastic actions from Gaussian distributions.

        :param state_dim: Dimension of the state
        :param action_dim: Dimension of the action
        :param squash: Whether to apply invertible squashing function (tanh) to the Gaussian samples,
        :param independent_std: Whether to use fixed independent standard deviation
        :param logstd_init: Initial log standard deviation(if not using fixed independent standard deviation)
        :param hidden_units: Size of the hidden layers in Q-network
        :param mid_activation: Activation function of hidden layers
        :param mid_activation_kwargs: Parameters for the middle activation
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
        self.squash = squash
        self.independent_std = independent_std
        assert not (self.independent_std and self.squash)

        self.logstd_init = logstd_init
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.feature_encoder = feature_encoder

        super().__init__(
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
        Make Gaussian Actor layers from layer dimensions and activations and initialize its weights and biases.
        """
        layers = []
        layer_units = [self.input_dim, *self.hidden_units]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1], bias=self.use_bias, device=self.device))
            if self.mid_activation is not None:
                layers.append(self.mid_activation(**self.mid_activation_kwargs))

        self.trunks = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(layer_units[-1], self.output_dim, bias=False, device=self.device)

        self.layers = nn.ModuleDict({"trunk": self.trunks, "mean": self.mean_layer})
        self.layers.apply(self._init_weights)

        if not self.independent_std:
            self.logstd_layer = nn.Linear(self.hidden_units[-1], self.output_dim, bias=False, device=self.device)
            torch.nn.init.constant_(self.logstd_layer.weight, val=self.logstd_init)
            self.layers['logstd'] = self.logstd_layer

    def forward(self, state: Union[torch.Tensor], deterministic: bool = False, detach_encoder: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Return action and its log_probability of the Gaussian actor for the given state.

        :param state: state(1 torch tensor)
        :param deterministic: whether to sample action from the computed distribution.
        :param detach_encoder: whether to detach encoder weights while training.
        :return: action (scaled and biased), log_prob
        """
        dist = self.compute_dist(state, detach_encoder)

        if deterministic:
            sample = dist.mean
        else:
            if self.squash:
                sample = dist.rsample()
            else:
                sample = dist.sample()

        log_prob = dist.log_prob(sample)

        if self.squash:
            tanh_sample = torch.tanh(sample)
            log_prob -= torch.log(self.action_scale * (1 - tanh_sample.pow(2)) + 1e-10)

            action = self.action_scale * tanh_sample + self.action_bias
        else:
            action = self.action_scale * sample + self.action_bias

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def compute_dist(self, state: torch.Tensor, detach_encoder: bool = False) -> Normal:
        """
        Return Normal distribution of the Gaussian actor for the given state.

        :param state: state(a torch tensor)
        :return: Normal distribution
        """
        state = state.to(self.device)

        if self.feature_encoder is not None:
            state = self.feature_encoder(state)

            if detach_encoder:
                state = state.detach()

        trunk_output = self.trunks(state)
        output_mean = self.mean_layer(trunk_output)

        if self.independent_std:
            output_std = self.std.expand_as(output_mean)
        else:
            output_std = torch.clamp(self.logstd_layer(trunk_output), LOG_STD_MIN, LOG_STD_MAX).exp()

        dist = Normal(loc=output_mean, scale=output_std)

        return dist

    def log_prob(self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        """
        Return log_probability of the Gaussian actor for the given state.

        :param state: state(a torch tensor)
        :param action: wanted action to calculate its log_probability
        :return: log_prob
        """
        dist = self.compute_dist(state, detach_encoder)
        action = (action - self.action_bias) / self.action_scale

        if self.squash:
            eps = torch.finfo(action.dtype).eps
            action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)

            action = 0.5 * (action.log1p() - (-action).log1p())

        return dist.log_prob(action).sum(dim=-1, keepdim=True)

    def entropy(self, state: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        """
        Return entropy of the Gaussian actor for the given state.

        :param state: state(a torch tensor)
        :return: entropy
        """
        dist = self.compute_dist(state, detach_encoder)
        return dist.entropy().sum(dim=-1)


if __name__ == "__main__":
    a = GaussianActor(5, 2)
    print(a)
    print(a(torch.ones((1, 5))))
    print(a.layers.apply(a._init_weights))
    print(a.layers['mean'].weight)
    # for layer, param in zip(a.layers.parameters(), a.parameters()):
    #     print(layer.shape, param.shape)
    #
