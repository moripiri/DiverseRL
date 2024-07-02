"""
Pixel-RL contains algorithms that train on pixel-level observations.
"""

from diverserl.algos.pixel_rl.dqn import DQN
from diverserl.algos.pixel_rl.ppo import PPO
from diverserl.algos.pixel_rl.sac_ae import SAC_AE

__all__ = ["SAC_AE", "DQN", "PPO"]
