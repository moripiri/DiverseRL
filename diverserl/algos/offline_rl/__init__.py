"""
Offline RL contains RL algorithms learns only from data trajectories.

ref: https://corl-team.github.io/CORL/
"""
from diverserl.algos.offline_rl.bc import BC
from diverserl.algos.offline_rl.cql import CQL
from diverserl.algos.offline_rl.dt import DT

__all__ = ["BC", "CQL", "DT"]
