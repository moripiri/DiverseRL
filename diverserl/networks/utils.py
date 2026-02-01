from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from torch import nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs


def get_activation(
    activation: Optional[_activation], activation_kwarg: Optional[_kwargs]
) -> Tuple[Optional[Type[nn.Module]], _kwargs]:
    resolved: Optional[Type[nn.Module]] = getattr(nn, activation) if isinstance(activation, str) else activation  # type: ignore[assignment]

    if activation_kwarg is None:
        activation_kwarg = {}
    else:
        for key, value in activation_kwarg.items():
            assert isinstance(
                value, (int, float, bool, str)
            ), "Value of activation_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                activation_kwarg[key] = eval(value)
            else:
                activation_kwarg[key] = value

    return resolved, activation_kwarg


def get_initializer(
    initializer: Optional[_initializer], initializer_kwarg: Optional[_kwargs]
) -> Tuple[Optional[Callable[..., Any]], _kwargs]:
    resolved: Optional[Callable[..., Any]] = (
        getattr(nn.init, initializer) if isinstance(initializer, str) else initializer  # type: ignore[assignment]
    )
    if initializer_kwarg is None:
        initializer_kwarg = {}
    else:
        for key, value in initializer_kwarg.items():
            assert isinstance(
                value, (int, float, bool, str)
            ), "Value of initializer_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                initializer_kwarg[key] = eval(value)
            else:
                initializer_kwarg[key] = value

    return resolved, initializer_kwarg
