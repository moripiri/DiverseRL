import argparse

from gymnasium.wrappers import AtariPreprocessing, FrameStack

from diverserl.algos.deep_rl import DQN
from diverserl.common.utils import make_env, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="DQN Learning Example")

    # env hyperparameters
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5", help="Name of the gymnasium environment to run.")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument(
        "--env-option",
        default={},
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3...",
        help="Additional options to pass into the environment.",
    )

    # dqn hyperparameters
    parser.add_argument("--network-type", type=str, default="MLP", choices=["MLP"])
    parser.add_argument(
        "--network-config", default={}, action=StoreDictKeyPair, metavar="KEY1=VAL1 KEY2=VAL2 KEY3=VAL3..."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        choices=range(0, 1),
        help="Probability to conduct random action during training",
    )
    parser.add_argument("--gamma", type=float, default=0.99, choices=range(0, 1), help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for optimizer.")
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
    from copy import deepcopy

    import gymnasium as gym

    # env, eval_env = make_env(env_id=args.env_id, seed=args.seed, env_option=args.env_option)
    env = FrameStack(AtariPreprocessing(gym.make("ALE/Pong-v5", repeat_action_probability=0.0, frameskip=1)), 3)
    eval_env = deepcopy(env)

    algo = DQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network_type=args.network_type,
        network_config=args.network_config,
        eps=args.eps,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        optimizer_kwargs=args.optimizer_kwargs,
        target_copy_freq=args.target_copy_freq,
        device=args.device,
    )

    trainer = DeepRLTrainer(
        algo=algo,
        env=env,
        eval_env=eval_env,
        seed=args.seed,
        training_start=args.training_start,
        training_num=args.training_num,
        train_type=args.train_type,
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
