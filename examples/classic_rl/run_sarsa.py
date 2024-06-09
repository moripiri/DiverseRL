import argparse

from diverserl.algos.classic_rl import SARSA
from diverserl.common.utils import make_envs
from diverserl.trainers import ClassicTrainer
from examples.utils import StoreDictKeyPair, set_config


def get_args():
    parser = argparse.ArgumentParser(description="Dyna-Q Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

    # env setting
    parser.add_argument(
        "--env-id",
        type=str,
        default="FrozenLake-v1",
        choices=["Blackjack-v1", "Taxi-v3", "CliffWalking-v0", "FrozenLake-v1"],
        help="Name of the gymnasium environment to run.",
    )
    parser.add_argument("--render", default=False, action="store_true", help='Render evaluation environment')

    parser.add_argument(
        "--env-option",
        default={"is_slippery": False},
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
    parser.add_argument("--vector-env", type=bool, default=False, help="Whether to make synced vectorized environments")

    # algorithm setting
    parser.add_argument("--gamma", type=float, default=0.9, choices=range(0, 1), help="Discount factor.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Step-size parameter (learning rate)")
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
    parser.add_argument("--eval-ep", type=int, default=1, help="Number of episodes to run evaulation.")

    parser.add_argument(
        "--log-tensorboard", action="store_true", default=False, help="Whether to save the run in tensorboard"
    )
    parser.add_argument("--log-wandb", action="store_true", default=False, help="Whether to save the run in wandb.")
    parser.add_argument("--record-video", action="store_true", default=False, help="Whether to record the evaluation.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    config = set_config(args)

    env = make_envs(**config)
    algo = SARSA(env, **config)

    trainer = ClassicTrainer(
        algo=algo,
        **config
    )

    trainer.run()
