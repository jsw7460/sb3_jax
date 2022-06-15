import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import colored

from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy
from offline_baselines_jax.sopt import utils
import d4rl_mujoco

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)

@hydra.main(config_path="./offline_baselines_jax/sopt/conf", config_name="amsopt_conf")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':
    class SoptRLWorkSpace(object):
        def __init__(self, cfg: DictConfig):
            self.img_shape = cfg.img_shape
            self.model = None

            env = self.get_env(cfg)
            self.env = env
            self.model, learning_kwargs, pretrained_is_setted = self.get_model(cfg)
            # assert pretrained_is_setted
            # self.model.test_inv_dyna_for_expert_data()

            if cfg.model_type == "SENSOR":
                if not pretrained_is_setted:
                    [print(colored("Prerequisite is not setted. Start with train the prerequisite model.", "red")) for _ in range(100)]
                    # self.model.train_inv_dyna_with_expert_data(path=cfg.prerequisite_path)
                    self.model.prerequisite_ft(path=cfg.prerequisite_path)

                with self.model.warmup_phase():
                    [print(colored("Warmup phase start", "green")) for _ in range(100)]
                    self.model.learn(**learning_kwargs, total_timesteps=cfg.n_warmup_steps)

                with self.model.rl_phase():
                    [print(colored("RL phase start", "green")) for _ in range(100)]
                    self.model.learn(**learning_kwargs, reset_num_timesteps=False, total_timesteps=cfg.n_rl_steps)

                # Done
                exit()

            else:       # SAC, SAC-BC
                print(colored(f"Model type: {cfg.model_type}", "yellow"))
                self.model.learn(**learning_kwargs)

        def get_env(self, cfg):
            if "maze" in cfg.env_name:
                env = utils.get_env(cfg)
                if not cfg.img_based:
                    env = utils.GoalMDPSensorObservationStackWrapper(env, n_frames=cfg.n_frames)
            else:
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

                # Set pretrained component (e.g., behavior cloner, inverse dynamics, ...)
                pretrained_is_setted = model.set_prerequisite_component(cfg.prerequisite_path)

                # Set learning keward args
                learning_kwargs = OmegaConf.to_container(cfg.sopt_learning_kwargs, resolve=True)
                return model, learning_kwargs, pretrained_is_setted

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
