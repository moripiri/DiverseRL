from copy import deepcopy

import gymnasium as gym

from diverserl.algos.deep_rl import DDPG, DQN, REINFORCE, TD3, SACv1, SACv2, PPO
from diverserl.trainers import DeepRLTrainer, OnPolicyTrainer

env = gym.make("CartPole-v1")

# eval_env = gym.make("CartPole-v1")

#env = gym.make("InvertedPendulum-v4")
eval_env = deepcopy(env)

# algo = DQN(env.observation_space, env.action_space)
# algo = DDPG(env)
# algo = TD3(env)
# algo = REINFORCE(
#     env.observation_space,
#     env.action_space,
#     buffer_size=env.spec.max_episode_steps,
#     network_config={"Continuous": {"squash": True}},
#     device="cuda",
# )
algo = PPO(env.observation_space, env.action_space, horizon=1024, batch_size=256, device='cpu')
trainer = OnPolicyTrainer(algo, env, eval_env, eval_ep=1, max_step=100000)
#trainer = DeepRLTrainer(algo, env, eval_env, train_type="offline", do_eval=False, training_start=0, max_step=1000000)
trainer.run()

print(env.spec.id)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(type(env.reset()[0]))
