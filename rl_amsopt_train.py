import hydra
from omegaconf import DictConfig, OmegaConf

import gym
from offline_baselines_jax.sopt import utils
from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy


@hydra.main(config_path="./offline_baselines_jax/sopt/conf", config_name="amsopt_conf")
def main(cfg: DictConfig) -> None:
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':
    class SoptRLWorkSpace(object):
        def __init__(self, cfg: DictConfig):
            self.img_shape = cfg.img_shape

            env = self.get_env(cfg)
            self.env = env

            self.model, learning_kwargs, bc_is_set = self.get_model(cfg)

            if cfg.model_type == "SENSOR" and not bc_is_set:
                self.model.train_bc(bc_path=cfg.bc_path)

            for i in range(100):
                self.model.learn(**learning_kwargs, reset_num_timesteps=False)
                self.model.save("./sacmaze.zip")

        def get_env(self, cfg):
            if "maze" in cfg.env_name:
                env = utils.get_env(cfg)
                env = utils.GoalMDPSensorObservationStackWrapper(env, n_frames=cfg.n_frames)

            else:
                import d4rl_mujoco
                import gym
                env = gym.make(cfg.env_name)
                env = utils.RewardMDPSensorObservationStackWrapper(env, n_frames=cfg.n_frames)
            return env

        def get_model(self, cfg):
            # Get policy first
            observation_space = self.env.observation_space
            if isinstance(observation_space, gym.spaces.Box):
                policy_class = SACPolicy
            elif isinstance(observation_space, gym.spaces.Dict):
                policy_class = MultiInputPolicy
            else:
                raise NotImplementedError()

            if cfg.model_type == "SENSOR":
                # Set model
                model = hydra.utils.instantiate(
                    cfg.sensor_based_model,
                    env=self.env,
                    policy=policy_class
                )

                # Set expert state buffer
                model.set_expert_buffer(cfg.expert_buffer_path, cfg.n_frames)

                # Set behavior cloner
                bc_is_set = model.set_behavior_cloner(cfg.bc_path)
                if bc_is_set:
                    print("Start with trained BC")
                else:
                    print("Start with train the BC")
                # Set learning keward args
                learning_kwargs = OmegaConf.to_container(cfg.sopt_learning_kwargs, resolve=True)
                return model, learning_kwargs, bc_is_set

            elif cfg.model_type == "SAC":
                model = hydra.utils.instantiate(
                    cfg.sac_model,
                    env=self.env,
                    policy=policy_class
                )
                learning_kwargs = OmegaConf.to_container(cfg.sac_learning_kwargs, resolve=True)
                return model, learning_kwargs, None

            elif cfg.model_type == "SACBC":
                model = hydra.utils.instantiate(
                    cfg.sacbc_model,
                    env=self.env,
                    policy=policy_class
                )

                # Set expert state buffer
                model.set_expert_buffer(cfg.expert_buffer_path, cfg.n_frames)

                learning_kwargs = OmegaConf.to_container(cfg.sacbc_learning_kwargs, resolve=True)
                return model, learning_kwargs, None

            else:
                raise NotImplementedError()

    main()
