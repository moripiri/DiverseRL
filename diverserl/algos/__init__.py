"""
Deep_rl contains model-free deep rl algorithms such as DQN, DDPG, etc.
"""
from diverserl.algos.ddpg import DDPG
from diverserl.algos.dqn import DQN
from diverserl.algos.ppo import PPO
from diverserl.algos.sac import SACv1, SACv2
from diverserl.algos.td3 import TD3

__all__ = ["DQN", "DDPG", "TD3", "SACv1", "SACv2", "PPO"]
