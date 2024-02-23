import argparse

from diverserl.algos.deep_rl import REINFORCE
from diverserl.common.utils import make_env, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="REINFORCE Learning Example")

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
    parser.add_argument("--network-type", type=str, default="MLP", choices=["MLP"])
    parser.add_argument(
        "--network-config",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Configurations of the REINFORCE networks.",
    )
    parser.add_argument("--buffer-size", type=int, default=int(10**6), help="Maximum length of replay buffer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the network")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer class (or str) of the network")
    parser.add_argument(
        "--optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3....",
        help="Parameter dict for the optimizer",
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

    set_seed(args.seed)

    if args.render:
        args.env_option["render_mode"] = "human"
    config = vars(args)
    env, eval_env = make_env(        **config
)

    algo = REINFORCE(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **config

    )

    trainer = DeepRLTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
        **config

    )

    trainer.run()
