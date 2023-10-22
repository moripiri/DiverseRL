import gymnasium as gym

toytext = ['Blackjack-v1', 'Taxi-v3', "CliffWalking-v0", "FrozenLake-v1"]

env = gym.make(toytext[0])

print(env.observation_space)
