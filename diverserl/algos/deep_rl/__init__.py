"""
Deep_rl contains model-free deep rl algorithms such as DQN, DDPG, etc.
"""

from diverserl.algos.deep_rl.ddpg import DDPG
from diverserl.algos.deep_rl.dqn import DQN
from diverserl.algos.deep_rl.sac import SACv2
from diverserl.algos.deep_rl.td3 import TD3

__all__ = ["DQN", "DDPG", "TD3", "SACv2"]
