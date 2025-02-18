import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from diverserl.common.utils import pprint_config


@hydra.main(version_base=None, config_path="diverserl/configs", config_name="test")
def run(cfg: DictConfig) -> None:
    if pprint_config(OmegaConf.to_container(cfg, resolve=True)):
        env, eval_env = instantiate(config=cfg.env)
        algo = instantiate(config=cfg.algo, env=env, eval_env=eval_env)
        trainer = instantiate(cfg.trainer, algo=algo, configs=OmegaConf.to_yaml(cfg))
        trainer.run()

if __name__ == "__main__":
    run()
