import argparse

import yaml

from diverserl.algos.classic_rl import MonteCarlo
from diverserl.common.utils import make_envs, set_seed
from diverserl.trainers import ClassicTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="Monte-Carlo Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

    # env setting
    parser.add_argument(
        "--env-id",
        type=str,
        default="CliffWalking-v0",
        choices=["Taxi-v3", "CliffWalking-v0", "FrozenLake-v1"],
        help="Name of the gymnasium environment to run.",
    )
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional options to pass into the environment.",
    )
    parser.add_argument(
        "--wrapper-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional wrappers to be applied to the environment.",
    )
    # algorithm setting
    parser.add_argument("--gamma", type=float, default=0.9, choices=range(0, 1), help="Discount factor.")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        choices=range(0, 1),
        help="Probability to conduct random action during training",
    )
    # trainer setting
    parser.add_argument("--max-episode", type=int, default=1000, help="Maximum number of episodes to run.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=100, help="When to run evaulation in every n episodes.")
    parser.add_argument("--eval-ep", type=int, default=10, help="Number of episodes to run evaulation.")
    parser.add_argument(
        "--log-tensorboard", action="store_true", default=True, help="Whether to save the run in tensorboard"
    )
    parser.add_argument("--log-wandb", action="store_true", default=True, help="Whether to save the run in wandb.")
    parser.add_argument("--record-video", action="store_true", default=True, help="Whether to record the evaluation.")

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

    env = make_envs(sync_vector_env=False, **config)

    algo = MonteCarlo(env, **config)
    trainer = ClassicTrainer(
        algo=algo,
        env=env,
        **config
    )

    trainer.run()
