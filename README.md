# DiverseRL

---
**DiverseRL** is a repository that aims to implement and benchmark reinforcement learning algorithms.

This repo aims to implement algorithms of various sub-topics in RL (e.g. model-based RL, offline RL), in wide variety of environments.

## Features


- Wandb logging
- Tensorboard

## Installation

---
You can install the requirements by using **Poetry**.
```bash
git clone https://github.com/moripiri/DiverseRL.git
cd DiverseRL

poetry install
```


## Algorithms

---
Currently, the following algorithms are available.


#### Model-free Deep RL
- [DQN (Deep Q-Network)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - [Double DQN](https://arxiv.org/abs/1509.06461)
  - [Dueling DQN](https://arxiv.org/abs/1511.06581)
  - [NoisyNet](https://arxiv.org/abs/1706.10295)
- [DDPG (Deep Deterministic Policy Gradients)](https://arxiv.org/abs/1509.02971)
- [TD3 (Twin Deep Delayed Deterministic Policy Gradients)](https://arxiv.org/abs/1802.09477)
- [SAC (Soft Actor Critic)](https://arxiv.org/abs/1812.05905)
- [PPO (Proximal Policy Gradients)](https://arxiv.org/abs/1707.06347)

#### Classic RL
Classic RL algorithms that are mostly known by Sutton's [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf).
Can be trained in Gymnasium's toy text environments.
- SARSA
- Q-learning
- Model-free Monte-Carlo Control
- Dyna-Q


Getting Started
---

Training requires two gymnasium environments(for training and evaluation), algorithm, and trainer.

`examples/` folder provides python files for training of each implemented RL algorithms.

```python
# extracted from examples/run_dqn.py
import gymnasium as gym
from diverserl.algos import DQN
from diverserl.trainers import DeepRLTrainer
from diverserl.common.utils import make_envs

env, eval_env = make_envs(env_id='CartPole-v1')

algo = DQN(env=env, **config)

trainer = DeepRLTrainer(
    algo=algo,
    env=env,
    eval_env=eval_env,
)

trainer.run()



```
Or in `examples` folder, you can pass configuration parameters from command line arguments.
```bash
python examples/run_dqn.py --env-id CartPole-v1
```
Or yaml files for configurations.

```bash
python examples/run_dqn.py --configs-path configurations/dqn_classic_control.yaml
```
