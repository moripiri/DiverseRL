# DiverseRL

---
**DiverseRL** is a repository that aims to implement and benchmark reinforcement learning algorithms.

This repo aims to implement algorithms of various sub-topics in RL (e.g. model-based RL, offline RL), in wide variety of environments.

## Features
- **Training loop in a single function**
  - Each algorithm's training procedure is written in a single function for readers' convenience.
- **Single-file configuration file**
  - With the use of [Hydra](https://hydra.cc/), you can manage settings of your experiment in a single yaml file.
- **Logging**
  - WandB
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
Model-free Deep RL algorithms are set of algorithms that can train environments with state-based observations without model.

- [DQN (Deep Q-Network)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - [Double DQN](https://arxiv.org/abs/1509.06461)
  - [Dueling DQN](https://arxiv.org/abs/1511.06581)
  - [NoisyNet](https://arxiv.org/abs/1706.10295)
- [DDPG (Deep Deterministic Policy Gradients)](https://arxiv.org/abs/1509.02971)
- [TD3 (Twin Deep Delayed Deterministic Policy Gradients)](https://arxiv.org/abs/1802.09477)
- [SAC (Soft Actor Critic)](https://arxiv.org/abs/1812.05905)
- [PPO (Proximal Policy Gradients)](https://arxiv.org/abs/1707.06347)

#### Pixel RL
Pixel RL contains set of algorithms that are set to train environments with high-dimensional images as observations, Such as Atari 2600 and dm-control.

- [DQN (Deep Q-Network)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (for Atari 2600)
- [SAC (Soft Actor Critic)](https://arxiv.org/abs/1812.05905) (for dm-control)
- [PPO (Proximal Policy Gradients)](https://arxiv.org/abs/1707.06347) (for Atari 2600)
- [SAC-AE (Soft Actor Critic with Autoencoders)](https://arxiv.org/abs/1910.01741) (for dm-control)
- [CURL (Contrastive Unsupervised Representations for Reinforcement Learning)](https://arxiv.org/abs/2004.04136) (for dm-control)
- [RAD (Reinforcement Learning with Augmented Data)](https://arxiv.org/abs/2004.14990) (for dm-control)
- [DrQ (Data-Regularized Q)](https://arxiv.org/abs/2004.13649) (for dm-control)
- [DrQ-v2 (Data-Regularized Q v2)](https://arxiv.org/abs/2107.09645) (for dm-control)

#### Offline RL
Offline RL algorithms are set of algorithms that can learn from set of environment trajectories.

- Any Percent BC (Behavior Cloning)
- [CQL (Conservative Q-learning)](https://arxiv.org/pdf/2006.04779)

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

```python
from diverserl.algos import DQN
from diverserl.trainers import DeepRLTrainer
from diverserl.common.utils import make_envs

env, eval_env = make_envs(env_id='CartPole-v1')

algo = DQN(env=env, eval_env=eval_env, **config)

trainer = DeepRLTrainer(
    algo=algo,
    **config
)

trainer.run()
```

Or you use [hydra](https://hydra.cc/) by running `run.py`.
```bash
python run.py env=gym_classic_control algo=dqn trainer=deeprl_trainer algo.device=cuda trainer.max_step=10000
```

```bash
python run.py --config-name ppo_gym_atari algo.device=cuda trainer.log_wandb=true
```

```bash
python run.py --config-name dqn_gym_atari algo.device=cuda trainer.log_wandb=true
```
