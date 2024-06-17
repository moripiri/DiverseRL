import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="test")
def run(cfg: DictConfig) -> None:
    env, eval_env = instantiate(config=cfg.env)
    algo = instantiate(config=cfg.algo, env=env, eval_env=eval_env)
    trainer = instantiate(config=cfg.trainer, algo=algo, configs=OmegaConf.to_yaml(cfg))
    trainer.run()


if __name__ == "__main__":
    run()
