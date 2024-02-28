import argparse

import yaml

from diverserl.algos.dqns import DuelingDQN
from diverserl.common.utils import make_env, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="Double DQN Learning Example")
    parser.add_argument('--config-path', type=str, help="Path to the config yaml file (optional)")

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
    # atari env hyperparameters
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--terminal-on-life-loss", type=bool, default=True)
    parser.add_argument("--grayscale-obs", type=bool, default=True)
    parser.add_argument("--repeat-action-probability", type=float, default=0.)

    # dqn hyperparameters
    parser.add_argument("--network-type", type=str, default="MLP", choices=["MLP"])
    parser.add_argument(
        "--network-config", default={"Q_network": {'estimator': 'mean'}}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
    )
    parser.add_argument(
        "--eps-initial",
        type=float,
        default=0.05,
        choices=range(0, 1),
        help="Initial probability to conduct random action during training",
    )
    parser.add_argument(
        "--eps-final",
        type=float,
        default=0.05,
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
    parser.add_argument("--gamma", type=float, default=0.9, choices=range(0, 1), help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for optimizer.")
    parser.add_argument("--buffer-size", type=int, default=10**6, help="Maximum length of replay buffer.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the Q-network")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer class (or str) for the Q-network")
    parser.add_argument(
        "--optimizer-kwargs",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
        help="Parameter dict for the optimizer",
    )
    parser.add_argument(
        "--target-copy-freq", type=int, default=5, help="N step to pass to copy Q-network to target Q-network"
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
        "--training_num", type=float, default=1, help="Number of times to run algo.train() in every training iteration"
    )
    parser.add_argument(
        "--train-type",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Type of algorithm training strategy (online, offline)",
    )
    parser.add_argument("--max-step", type=int, default=3000000, help="Maximum number of steps to run.")
    parser.add_argument("--do-eval", type=bool, default=True, help="Whether to run evaluation during training.")
    parser.add_argument("--eval-every", type=int, default=10000, help="When to run evaulation in every n episodes.")
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
    set_seed(args.seed)

    if args.render:
        args.env_option["render_mode"] = "human"

    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            config['config_path'] = args.config_path
    else:
        config = vars(args)

    env, eval_env = make_env(**config)

    algo = DuelingDQN(
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
