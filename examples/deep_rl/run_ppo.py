import argparse
import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from diverserl.algos.deep_rl import PPO
from diverserl.trainers import OnPolicyTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="PPO Learning Example")

    # env hyperparameters
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Name of the gymnasium environment to run.")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional options to pass into the environment.",
    )

    # deep rl hyperparameters
    parser.add_argument("--network-type", type=str, default="MLP", choices=["MLP"])
    parser.add_argument(
        "--network-config", default={}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="clip",
        choices=["clip", "adaptive_kl", "fixed_kl"],
        help="Type of surrogate objectives (clip, adaptive_kl, fixed_kl)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.2,
        choices=range(0, 1),
        help='The surrogate clipping value. Only used when the mode is "clip"',
    )
    parser.add_argument(
        "--target-dist",
        type=float,
        default=0.01,
        help='Target KL divergence between the old policy and the current policy. Only used when the mode is "adaptive_kl"',
    )
    parser.add_argument(
        "--beta", type=float, default=3.0, help="Hyperparameter for the KL divergence in the surrogate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")
    parser.add_argument(
        "--lambda-gae", type=float, default=0.96, help="The lambda for the General Advantage Estimation"
    )
    parser.add_argument("--horizon", type=int, default=128, help="The number of steps to gather in each policy rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for optimizer.")
    parser.add_argument("--buffer-size", type=int, default=10**5, help="Maximum length of replay buffer.")
    parser.add_argument("--actor-lr", type=float, default=0.001, help="Learning rate for actor.")
    parser.add_argument("--actor-optimizer", type=str, default="Adam", help="Optimizer class (or name) for actor.")
    parser.add_argument(
        "--actor-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Parameter dict for actor optimizer.",
    )
    parser.add_argument("--critic-lr", type=float, default=0.001, help="Learning rate for critic.")
    parser.add_argument("--critic-optimizer", type=str, default="Adam", help="Optimizer class (or name) for critic.")
    parser.add_argument(
        "--critic-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Parameter dict for critic optimizer.",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, ...) on which the code should be run"
    )

    # trainer hyperparameters
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--max-step", type=int, default=10000, help="Maximum number of steps to run.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=1000, help="When to run evaulation in every n episodes.")
    parser.add_argument("--eval-ep", type=int, default=1, help="Number of episodes to run evaulation.")
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

    algo = PPO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network_type=args.network_type,
        network_config=args.network_config,
        mode=args.mode,
        clip=args.clip,
        target_dist=args.target_dist,
        beta=args.beta,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        horizon=args.horizon,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        actor_optimizer=args.actor_optimizer,
        actor_optimizer_kwargs=args.actor_optimizer_kwargs,
        critic_lr=args.critic_lr,
        critic_optimizer=args.critic_optimizer,
        critic_optimizer_kwargs=args.critic_optimizer_kwargs,
        device=args.device,
    )

    trainer = OnPolicyTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
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
