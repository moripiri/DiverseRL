import gymnasium as gym

from diverserl.algos.classic_rl import QLearning
from diverserl.trainers import ClassicTrainer

# toytext = ['Blackjack-v1', 'Taxi-v3', "CliffWalking-v0", "FrozenLake-v1"]

# env = gym.make("FrozenLake-v1", map_name='8x8', is_slippery=False)
# env = gym.make("Blackjack-v1")
env = gym.make("Taxi-v3")
eval_env = gym.make("Taxi-v3")

# env = gym.make("CliffWalking-v0", max_episode_steps=50)

# algo = MonteCarlo(env)
algo = QLearning(env)
trainer = ClassicTrainer(algo, env, eval_env, max_episode=1000, eval_every=100)
trainer.run()
