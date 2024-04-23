import argparse

import numpy as np
import yaml

from diverserl.algos import PPO
from diverserl.common.utils import make_envs, set_seed
from diverserl.trainers import OnPolicyTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="PPO Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

    # env hyperparameters
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Name of the gymnasium environment to run.")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional options to pass into the environment.",
    )

    # deep rl hyperparameters
    parser.add_argument("--network-type", type=str, default="Default", choices=["Default"])
    parser.add_argument(
        "--network-config", default={"Actor":{'mid_activation': 'Tanh', 'kernel_initializer_kwargs': {'gain': np.sqrt(2)}}, "Critic":{'mid_activation': 'Tanh', 'kernel_initializer_kwargs': {'gain': np.sqrt(2)}}}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
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
        "--lambda-gae", type=float, default=0.95, help="The lambda for the General Advantage Estimation"
    )
    parser.add_argument("--vf_coef", type=float, default=0.5, help="The value loss coef")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="The entropy coef")
    parser.add_argument("--horizon", type=int, default=2048, help="The number of steps to gather in each policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size for optimizer.")
    parser.add_argument("--num-epoch", type=int, default=10, help="The K epochs to update the policy")
    parser.add_argument("--actor-lr", type=float, default=0.0003, help="Learning rate for actor.")
    parser.add_argument("--actor-optimizer", type=str, default="Adam", help="Optimizer class (or name) for actor.")
    parser.add_argument(
        "--actor-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Parameter dict for actor optimizer.",
    )
    parser.add_argument("--critic-lr", type=float, default=0.0003, help="Learning rate for critic.")
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
    parser.add_argument("--max-step", type=int, default=1000000, help="Maximum number of steps to run.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=10000, help="When to run evaulation in every n episodes.")
    parser.add_argument("--eval-ep", type=int, default=1, help="Number of episodes to run evaulation.")
    parser.add_argument(
        "--log-tensorboard", action="store_true", default=False, help="Whether to save the run in tensorboard"
    )
    parser.add_argument("--log-wandb", action="store_true", default=False, help="Whether to save the run in wandb.")
    parser.add_argument("--record-video", action="store_true", default=False, help="Whether to record the evaluation.")

    parser.add_argument("--save-model", action="store_true", default=False, help="Whether to save the model")
    parser.add_argument("--save-freq", type=int, default=100000, help="Frequency of model saving.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            config['config_path'] = args.config_path
    else:
        config = vars(args)

    env, eval_env = make_envs(num_envs = 1, **config)

    algo = PPO(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        **config

    )

    trainer = OnPolicyTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
        **config

    )

    trainer.run()
