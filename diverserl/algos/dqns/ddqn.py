from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from diverserl.algos.deep_rl.base import DeepRL
from diverserl.common.buffer import ReplayBuffer
from diverserl.common.utils import get_optimizer, hard_update
from diverserl.networks import DeterministicActor, PixelEncoder


class DDQN(DeepRL):
    pass
