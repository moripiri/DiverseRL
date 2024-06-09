import argparse

from diverserl.algos import DDPG
from diverserl.common.utils import make_envs
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair, set_config


def get_args():
    parser = argparse.ArgumentParser(description="DDPG Learning Example")
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
    parser.add_argument(
        "--wrapper-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional wrappers to be applied to the environment.",
    )
    parser.add_argument("--vector-env", type=bool, default=True, help="Whether to make synced vectorized environments")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    # ddpg hyperparameters
    parser.add_argument("--network-type", type=str, default="Default", choices=["Default", "D2RL"])
    parser.add_argument(
        "--network-config", default={}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
    )
    parser.add_argument("--gamma", type=float, default=0.99, choices=range(0, 1), help="Discount factor")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        choices=range(0, 1),
        help="Interpolation factor in polyak averaging for target networks.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Stddev for Gaussian noise added to policy action at training time.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for optimizer.")
    parser.add_argument("--buffer-size", type=int, default=10**6, help="Maximum length of replay buffer.")

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

    parser.add_argument(
        "--training-start", type=int, default=1000, help="Number of steps to perform exploartion of environment"
    )
    parser.add_argument(
        "--training-freq", type=int, default=1, help="How often in total_step to perform training"
    )
    parser.add_argument(
        "--training_num", type=int, default=1, help="Number of times to run algo.train() in every training iteration"
    )

    parser.add_argument("--max-step", type=int, default=100000, help="Maximum number of steps to run.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=1000, help="When to run evaulation in every n episodes.")
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
    config = set_config(args)

    env = make_envs(**config)

    algo = DDPG(env=env, **config)

    trainer = DeepRLTrainer(
        algo=algo,
        **config
    )

    trainer.run()
