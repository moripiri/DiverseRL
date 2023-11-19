import gymnasium as gym

from diverserl.algos.deep_rl import DDPG, DQN, TD3
from diverserl.trainers import DeepRLTrainer

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
# env = gym.make("InvertedDoublePendulum-v4")

algo = DQN(env)
# algo = DDPG(env)
# algo = TD3(env)


trainer = DeepRLTrainer(algo, env, eval_env, training_start=1000, max_step=100000, eval_every=2000)
trainer.run()

print(env.spec.id)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(type(env.reset()[0]))
