from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf, DictConfig

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.utils import configure_logger, Logger
from offline_baselines_jax.sopt import CondVaeGoalGenerator, cond_vae_goal_generator_update
from offline_baselines_jax.sopt.buffer import CondVaeGoalGeneratorBuffer
import hydra


class Workspace(object):
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = jax.random.PRNGKey(self.seed)

        self.total_timestep_per_batch = cfg.total_timestep_per_batch
        self.log_interval = cfg.log_interval
        self.batch_size = cfg.batch_size
        self.n_subgoal = cfg.n_subgoal

        self.img_shape = cfg.img_shape      # One side length

        self._current_batch = 0
        self._current_trunc = 0
        self.timestep = 0
        self.data_path = cfg.data_path
        self.save_path = cfg.save_path

        self.vae = None         # type: CondVaeGoalGenerator
        self.buffer = None      # type: CondVaeGoalGeneratorBuffer
        self._logger = None     # type: Logger

        self._setup_model(cfg)

        self.buffer = CondVaeGoalGeneratorBuffer(
            data_path=self.data_path + f"Maze2d_img_{self.current_batch}",
            n_subgoal=cfg.n_subgoal
        )
        self._setup_dataset()

        if cfg.load_epoch >= 0:
            self._load(cfg.load_epoch)
            self._render()
        self._train()

    @property
    def target_futue_hop(self):
        return self.buffer.mean_target_future_hop

    @property
    def should_get_next_batch(self):
        return self._current_trunc == 2

    @property
    def current_batch(self):
        return self._current_batch

    @current_batch.setter
    def current_batch(self, value):
        value = (value % 10)
        self._current_batch = value

    @property
    def logger(self):
        return self._logger

    def _render(self):
        import pickle
        dropout_key, sampling_key = jax.random.split(self.rng)
        rngs = {"dropout": dropout_key, "sampling": sampling_key}
        replay_data = self.buffer.sample(100)
        observations = replay_data.observations
        subgoal_observations = replay_data.subgoal_observations
        goal_observations = replay_data.goal_observations

        target_future_hop = np.repeat(
            np.array(self.target_futue_hop),
            repeats=observations.shape[0],
            axis=0
        )[..., np.newaxis]
        print("Target future hop shape", target_future_hop.shape)
        recon, *_ = self.vae(observations, goal_observations, target_future_hop, rngs=rngs)

        imgs = {
            "observations": observations,
            "subgoal": subgoal_observations,
            "recon": recon
        }

        with open ("vae_img2.pkl", "wb") as f:
            pickle.dump(imgs, f)

        print("We saved ~!")
        exit()

    def _setup_dataset(self):
        print(self._current_batch, self._current_trunc)
        if self.should_get_next_batch:
            self.current_batch += 1
            self._current_trunc = 0
            print("Load new buffer", self.current_batch)
            self.buffer = CondVaeGoalGeneratorBuffer(
                data_path=self.data_path + f"Maze2d_img_{self.current_batch}",
                n_subgoal=self.n_subgoal
            )
        else:
            self.buffer.truncation_reset(self._current_trunc)
            # self.buffer.truncation_reset(self._current_trunc)

    def _setup_model(self, cfg: DictConfig) -> None:
        model_def = hydra.utils.instantiate(cfg.model)
        init_state = jnp.zeros((1, self.img_shape, self.img_shape, 3))
        init_goal = jnp.zeros((1, 2))
        init_target_future_hop = jnp.zeros((1, 1))

        param_key, dropout_key, sampling_key = jax.random.split(self.rng, 3)
        model_rngs = {"params": param_key, "dropout": dropout_key, "sampling": sampling_key}
        self.vae = Model.create(
            model_def,
            inputs=[model_rngs, init_state, init_goal, init_target_future_hop, False],
            tx=optax.radam(learning_rate=cfg.learning_rate)
        )
        self._logger = configure_logger(tensorboard_log=cfg.tensorboard_log, tb_log_name="prior")

    def push2logger(self, train_info: Dict):
        for key, value in train_info.items():
            if "loss" in key:
                self.logger.record(f"train/{key}", np.mean(np.array(value)))

    def _save(self, epoch):
        save_path = self.save_path + f"-epoch{epoch}"
        self.vae.save_dict(save_path)

    def _load(self, epoch):
        load_path = self.save_path + f"-epoch{epoch}"
        print("Load from", load_path)
        with open(load_path, "rb") as f:
            params = f.read()
            self.vae = self.vae.load_dict(params)

    def _train(self):
        for epoch in range(10000):
            self.train(epoch, self.timestep + self.total_timestep_per_batch)
            # self.current_batch = (self.current_batch + 1)
            self._save(epoch)
            self._current_trunc += 1
            self._setup_dataset()

    def train(self, epoch, total_timestep):
        # NOTE: Buffer를 truncation해서 학습 시키기 때문에, 주기적으로 다음 truncation을 load해주어야 함.
        while self.timestep < total_timestep:
            self.rng, _ = jax.random.split(self.rng)

            replay_data = self.buffer.sample(self.batch_size)
            observations = replay_data.observations
            subgoal_observations = replay_data.subgoal_observations
            goal_observations = replay_data.goal_observations
            target_future_hop = replay_data.target_future_hop

            new_vae, vae_info = cond_vae_goal_generator_update(
                rng=self.rng,
                vae=self.vae,
                observations=observations,
                subgoal_observations=subgoal_observations,
                goal_observations=goal_observations,
                target_future_hop=target_future_hop
            )

            self.push2logger(vae_info)
            self.vae = new_vae

            self.timestep += 1
            if self.timestep % self.log_interval == 0:
                self.logger.record("time/current_batch", self.current_batch, exclude="tensorboard")
                self.logger.record("time/timestep", self.timestep, exclude="tensorboard")
                self.logger.dump(step=self.timestep)
                self._save(epoch)


@hydra.main(config_path="./offline_baselines_jax/sopt/conf", config_name="prior_conf")
def main(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, "config.yaml")
    Workspace(cfg)

if __name__ == '__main__':
    main()
