import argparse

from diverserl.algos.classic_rl import QLearning
from diverserl.common.utils import make_env, set_seed
from diverserl.trainers import ClassicTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="Q-Learning Learning Example")
    # env setting
    parser.add_argument(
        "--env-id",
        type=str,
        default="Taxi-v3",
        choices=["Blackjack-v1", "Taxi-v3", "CliffWalking-v0", "FrozenLake-v1"],
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
    parser.add_argument("--eval-ep", type=int, default=10, help="Number of episodes to run evaulation.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    if args.render:
        args.env_option["render_mode"] = "human"

    env, eval_env = make_env(env_id=args.env_id, seed=args.seed, env_option=args.env_option)

    algo = QLearning(env=env, gamma=args.gamma, alpha=args.alpha, eps=args.eps)
    trainer = ClassicTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
        seed=args.seed,
        max_episode=args.max_episode,
        do_eval=args.do_eval,
        eval_every=args.eval_every,
        eval_ep=args.eval_ep,
    )

    trainer.run()
