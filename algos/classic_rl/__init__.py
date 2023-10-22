"""
classic_rl contains classic rl algorithms such as q-learning, sarsa.
"""

from algos.classic_rl.sarsa import SARSA
from algos.classic_rl.q_learning import QLearning
from algos.classic_rl.dyna_q import DynaQ
from algos.classic_rl.monte_carlo import MonteCarlo

__all__ = ['SARSA', 'QLearning', 'DynaQ', 'MonteCarlo']