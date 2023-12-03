from copy import deepcopy

import gymnasium as gym

from diverserl.algos.deep_rl import DDPG, DQN, REINFORCE, TD3, SACv1, SACv2
from diverserl.trainers import DeepRLTrainer

env = gym.make("CartPole-v1")

# eval_env = gym.make("CartPole-v1")

# env = gym.make("InvertedPendulum-v4", render_mode='human')
eval_env = deepcopy(env)

# algo = DQN(env.observation_space, env.action_space)
# algo = DDPG(env)
# algo = TD3(env)
algo = REINFORCE(
    env.observation_space,
    env.action_space,
    buffer_size=env.spec.max_episode_steps,
    network_config={"Continuous": {"squash": True}},
    device="cuda",
)

trainer = DeepRLTrainer(algo, env, eval_env, train_type="offline", do_eval=False, training_start=0, max_step=1000000)
trainer.run()

print(env.spec.id)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(type(env.reset()[0]))
