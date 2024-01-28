import argparse

from diverserl.algos.deep_rl import DDPG
from diverserl.common.utils import make_env, set_seed
from diverserl.trainers import DeepRLTrainer
from examples.utils import StoreDictKeyPair


def get_args():
    parser = argparse.ArgumentParser(description="DDPG Learning Example")

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

    # ddpg hyperparameters
    parser.add_argument("--network-type", type=str, default="MLP", choices=["MLP"])
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

    env, eval_env = make_env(env_id=args.env_id, seed=args.seed, env_option=args.env_option)

    algo = DDPG(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network_type=args.network_type,
        network_config=args.network_config,
        gamma=args.gamma,
        tau=args.tau,
        noise_scale=args.noise_scale,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        actor_optimizer=args.actor_optimizer,
        actor_optimizer_kwargs=args.actor_optimizer_kwargs,
        critic_lr=args.critic_lr,
        critic_optimizer=args.critic_optimizer,
        critic_optimizer_kwargs=args.critic_optimizer_kwargs,
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
