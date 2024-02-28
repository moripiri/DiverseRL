from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from diverserl.networks.base import Network


class DuelingNetwork(Network):
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
        super().__init__(
            input_dim=state_dim,
            output_dim=action_dim,
            mid_activation=mid_activation,
            mid_activation_kwargs=mid_activation_kwargs,
            kernel_initializer=kernel_initializer,
            kernel_initializer_kwargs=kernel_initializer_kwargs,
            bias_initializer=bias_initializer,
            bias_initializer_kwargs=bias_initializer_kwargs,
            use_bias=use_bias,
            device=device,
        )
        self.hidden_units = hidden_units
        self.feature_encoder = feature_encoder
