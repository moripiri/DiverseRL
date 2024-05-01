import argparse

import yaml

from diverserl.algos import SACv2
from diverserl.common.utils import make_envs, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="SACv2 Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

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
        "--network-type", type=str, default="Default", choices=["Default"], help="Type of the SACv2 networks to be used."
    )
    parser.add_argument(
        "--network-config",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Configurations of the SACv2 networks.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="The entropy temperature parameter.")
    parser.add_argument("--train-alpha", type=bool, default=True, help="Whether to train the parameter alpha.")
    parser.add_argument("--target-alpha", default=None, help="Target entropy value (Set as -|action_dim| if None).")
    parser.add_argument(
        "--tau", type=float, default=0.05, help="Interpolation factor in polyak averaging for target networks."
    )
    parser.add_argument(
        "--critic-update", type=int, default=2, help="Critic will only be updated once for every critic_update steps."
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
    parser.add_argument("--alpha-lr", type=float, default=0.001, help="Learning rate for alpha.")
    parser.add_argument("--alpha-optimizer", type=str, default="Adam", help="Optimizer class (or name) for alpha.")
    parser.add_argument(
        "--alpha-optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2",
        help="Parameter dict for alpha optimizer",
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
        "--training-freq", type=int, default=1, help="How often in total_step to perform training"
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
    parser.add_argument("--eval-every", type=int, default=10000, help="When to run evaulation in every n episodes.")
    parser.add_argument("--eval-ep", type=int, default=10, help="Number of episodes to run evaulation.")
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

    env = make_envs(**config)

    algo = SACv2(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        **config
    )

    trainer = DeepRLTrainer(
        algo=algo,
        env=env,
        **config
    )

    trainer.run()
