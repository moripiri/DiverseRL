"""
deep_rl contains model-free deep rl algorithms such as DQN, DDPG, etc.
"""

from diverserl.algos.deep_rl.ddpg import DDPG
from diverserl.algos.deep_rl.dqn import DQN

__all__ = ["DQN", "DDPG"]
