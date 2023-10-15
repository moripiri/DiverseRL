import gymnasium as gym
print(gym.__version__)
toytext = ['Blackjack-v1', 'Taxi-v3', "CliffWalking-v0", "FrozenLake-v1"]
for toy in toytext:
    env = gym.make(toy)
    print(env.reset())
print(*gym.envs.registry.keys(), sep='\n')