import argparse

import yaml

from diverserl.algos import DQN
from diverserl.common.utils import make_envs, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

    # env hyperparameters
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5", help="Name of the gymnasium environment to run.")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={"frame_skip": 4, "frame_stack": 4, "repeat_action_probability": 0.,
                 "image_size": 84, "noop_max": 30, "terminal_on_life_loss": True, "grayscale_obs": True, },
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
    # dqn hyperparameters
    parser.add_argument("--network-type", type=str, default="D2RL", choices=["Default", "Noisy", "D2RL"])
    parser.add_argument(
        "--network-config", default={}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
    )
    parser.add_argument(
        "--eps-initial",
        type=float,
        default=0.0,
        choices=range(0, 1),
        help="Initial probability to conduct random action during training",
    )
    parser.add_argument(
        "--eps-final",
        type=float,
        default=0.0,
        choices=range(0, 1),
        help="Final probability to conduct random action during training",
    )
    parser.add_argument(
        "--decay-fraction",
        type=float,
        default=0.5,
        choices=range(0, 1),
        help="Fraction of max_step to perform epsilon decay during training.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, choices=range(0, 1), help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for optimizer.")
    parser.add_argument("--buffer-size", type=int, default=10 ** 6, help="Maximum length of replay buffer.")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate of the Q-network")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer class (or str) for the Q-network")
    parser.add_argument(
        "--optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
        help="Parameter dict for the optimizer",
    )
    parser.add_argument("--anneal-lr", type=bool, default=False, help="Linearly decay learning rate.")

    parser.add_argument(
        "--target-copy-freq", type=int, default=1000, help="N step to pass to copy Q-network to target Q-network"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, ...) on which the code should be run"
    )

    # trainer hyperparameters
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")

    parser.add_argument(
        "--training-start", type=int, default=10000, help="Number of steps to perform exploartion of environment"
    )
    parser.add_argument(
        "--training-freq", type=int, default=4, help="How often in total_step to perform training"
    )
    parser.add_argument(
        "--training_num", type=float, default=1, help="Number of times to run algo.train() in every training iteration"
    )
    parser.add_argument("--max-step", type=int, default=3000000, help="Maximum number of steps to run.")
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

    env = make_envs(**config)

    algo = DQN(env=env,
               **config)

    trainer = DeepRLTrainer(
        algo=algo,
        env=env,
        **config
    )

    trainer.run()
