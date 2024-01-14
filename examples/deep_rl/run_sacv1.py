import argparse
import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from diverserl.algos.deep_rl import SACv1
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="SACv1 Learning Example")

    # env hyperparameters
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Name of the gymnasium environment to run.")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional options to pass into the environment.",
    )

    # deep rl hyperparameters
    parser.add_argument(
        "--network-type", type=str, default="MLP", choices=["MLP"], help="Type of the SACv1 networks to be used."
    )
    parser.add_argument(
        "--network-config",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2...",
        help="Configurations of the SACv1 networks.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="The entropy temperature parameter.")
    parser.add_argument(
        "--tau", type=float, default=0.05, help="Interpolation factor in polyak averaging for target networks."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for optimizer")
    parser.add_argument("--buffer-size", type=int, default=10**6, help="Maximum length of replay buffer")
    parser.add_argument("--actor-lr", type=float, default=0.001, help="Learning rate for actor.")
    parser.add_argument("--actor-optimizer", type=str, default="Adam", help="Optimizer class (or name) for actor")
    parser.add_argument(
        "--actor-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2...",
        help="Parameter dict for actor optimizer.",
    )
    parser.add_argument("--critic-lr", type=float, default=0.001, help="Learning rate of the critic")
    parser.add_argument("--critic-optimizer", type=str, default="Adam", help="Optimizer class (or str) for the critic")
    parser.add_argument(
        "--critic-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2...",
        help="Parameter dict for the critic optimizer",
    )
    parser.add_argument("--v-lr", type=float, default=0.001, help="Learning rate for value network.")
    parser.add_argument("--v-optimizer", type=str, default="Adam", help="Optimizer class (or name) for value network.")
    parser.add_argument(
        "--v-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2",
        help="Parameter dict for value network optimizer",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, ...) on which the code should be run"
    )

    # trainer hyperparameters
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--training-start", type=int, default=1000, help="Number of steps to perform exploartion of environment"
    )
    parser.add_argument(
        "--training_num", type=int, default=1, help="Number of times to run algo.train() in every training iteration"
    )
    parser.add_argument(
        "--train-type",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Type of algorithm training strategy (online, offline)",
    )
    parser.add_argument("--max-step", type=int, default=100000, help="Maximum number of steps to run.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=1000, help="When to run evaulation in every n episodes.")
    parser.add_argument("--eval-ep", type=int, default=10, help="Number of episodes to run evaulation.")
    parser.add_argument(
        "--log-tensorboard", action="store_true", default=False, help="Whether to save the run in tensorboard"
    )
    parser.add_argument("--log-wandb", action="store_true", default=False, help="Whether to save the run in wandb.")
    parser.add_argument("--save-model", action="store_true", default=False, help="Whether to save the model")
    parser.add_argument("--save-freq", type=int, default=100000, help="Frequency of model saving.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.render:
        args.env_option["render_mode"] = "human"

    env = gym.make(id=args.env_id, **args.env_option)
    env.action_space.seed(args.seed)

    eval_env = deepcopy(env)

    algo = SACv1(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network_type=args.network_type,
        network_config=args.network_config,
        gamma=args.gamma,
        alpha=args.alpha,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        actor_optimizer=args.actor_optimizer,
        actor_optimizer_kwargs=args.actor_optimizer_kwargs,
        critic_lr=args.critic_lr,
        critic_optimizer=args.critic_optimizer,
        critic_optimizer_kwargs=args.critic_optimizer_kwargs,
        v_lr=args.v_lr,
        v_optimizer=args.v_optimizer,
        v_optimizer_kwargs=args.v_optimizer_kwargs,
        device=args.device,
    )

    trainer = DeepRLTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
        training_start=args.training_start,
        training_num=args.training_num,
        train_type=args.train_type,
        max_step=args.max_step,
        do_eval=args.do_eval,
        eval_every=args.eval_every,
        eval_ep=args.eval_ep,
        log_tensorboard=args.log_tensorboard,
        log_wandb=args.log_wandb,
        save_model=args.save_model,
        save_freq=args.save_freq,
    )

    trainer.run()
