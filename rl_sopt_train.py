import hydra
from omegaconf import DictConfig, OmegaConf

from offline_baselines_jax.sopt import utils


@hydra.main(config_path="./offline_baselines_jax/sopt/conf", config_name="sopt_conf")
def main(cfg: DictConfig) -> None:
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':
    class SoptRLWorkSpace(object):
        def __init__(self, cfg: DictConfig):
            self.img_shape = cfg.img_shape

            # self.env = utils.get_env(cfg)
            env = utils.get_env(cfg)
            self.env = env

            self.model, learning_kwargs = self.get_model(cfg)

            self.train(learning_kwargs)

        def train(self, learning_kwargs):
            for i in range(100):
                self.model.learn(**learning_kwargs, reset_num_timesteps=False)
                self.model.save("./sacmaze.zip")

        def get_model(self, cfg):
            if cfg.model_type == "PGIS_SAC":
                model = hydra.utils.instantiate(
                    cfg.pgis_model,
                    env=self.env,
                    replay_buffer_class=hydra.utils.get_class(cfg.replay_buffer_class)
                )
                learning_kwargs = OmegaConf.to_container(cfg.pgis_learning_kwargs, resolve=True)
                model.set_vae(cfg)
                return model, learning_kwargs

            elif cfg.model_type == "SAC":
                model = hydra.utils.instantiate(
                    cfg.sac_model,
                    env=self.env,
                )
                learning_kwargs = OmegaConf.to_container(cfg.sac_learning_kwargs, resolve=True)
                return model, learning_kwargs

    main()
