from typing import Tuple, Union

import numpy as np
from torch import nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs


def get_activation(activation: _activation, activation_kwarg: _kwargs) -> Tuple[nn.Module, _kwargs]:
    activation = getattr(nn, activation) if isinstance(activation, str) else activation

    if activation_kwarg is None:
        activation_kwarg = {}
    else:
        for key, value in activation_kwarg.items():
            assert isinstance(value, Union[
                int, float, bool, str]), "Value of activation_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                activation_kwarg[key] = eval(value)
            else:
                activation_kwarg[key] = value

    return activation, activation_kwarg


def get_initializer(initializer: _initializer, initializer_kwarg: _kwargs):
    initializer = (
        getattr(nn.init, initializer) if isinstance(initializer, str) else initializer
    )
    if initializer_kwarg is None:
        initializer_kwarg = {}
    else:
        for key, value in initializer_kwarg.items():
            assert isinstance(value, Union[
                int, float, bool, str]), "Value of initializer_kwargs must be set as int, float, boolean or string"
            if isinstance(value, str):
                initializer_kwarg[key] = eval(value)
            else:
                initializer_kwarg[key] = value

    return initializer, initializer_kwarg
