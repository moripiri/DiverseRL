"""
classic_rl contains classic rl algorithms such as q-learning, sarsa.
"""
from diverserl.algos.classic_rl.dyna_q import DynaQ
from diverserl.algos.classic_rl.monte_carlo import MonteCarlo
from diverserl.algos.classic_rl.q_learning import QLearning
from diverserl.algos.classic_rl.sarsa import SARSA

__all__ = ['SARSA', 'QLearning', 'DynaQ', 'MonteCarlo']