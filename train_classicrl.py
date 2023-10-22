from algos.classic_rl import SARSA, QLearning, DynaQ, MonteCarlo
from trainers import ClassicTrainer

import gymnasium as gym

# toytext = ['Blackjack-v1', 'Taxi-v3', "CliffWalking-v0", "FrozenLake-v1"]

#env = gym.make("FrozenLake-v1", map_name='8x8', is_slippery=False)
#env = gym.make("Blackjack-v1")
#env = gym.make("Taxi-v3")
env = gym.make("CliffWalking-v0", max_episode_steps=50)

algo = MonteCarlo(env)
#algo = QLearning(env)
trainer = ClassicTrainer(algo, env, max_episode=10000)
trainer.run()
