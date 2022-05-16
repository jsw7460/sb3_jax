from typing import Any, Dict, List, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import GymEnv, Schedule, Params
from .core import _metla_offline_sac_update, _metla_sac_online_finetune_last_layer
from .metla import METLA
from .policies_mle import SACPolicy

TIMEOUT_LIMIT = 10000000


class LogEntropyCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class METLAMLE(METLA):
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[SACPolicy]] = SACPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = 1.0,
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = {},
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        without_exploration: bool = False,

        dropout: float = 0.0,
        context_len: int = 20,
        future_len: int = 7,
        normalize: bool = True,
    ):
        policy_kwargs.update({"dropout": dropout})
        self.target_entropy = target_entropy
        self.entropy_update = True
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval

        super(METLAMLE, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
            without_exploration=without_exploration,

            dropout=dropout,
            context_len=context_len,
            future_len=future_len,
            normalize=normalize
        )

    def _setup_model(self) -> None:
        super(METLAMLE, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            log_ent_coef_def = LogEntropyCoef(init_value)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key],
                                             tx=optax.adam(learning_rate=self.lr_schedule(1)))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(self.ent_coef)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            self.entropy_update = False

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.latent_dim = self.policy.latent_dim

        super(METLAMLE, self)._create_metla_aliases()

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        # Actor
        bc_losses, sac_actor_losses, goal_reaching_losses = [], [], []

        # Critic
        critic_losses = []

        # AE
        gae_losses, wae_losses, wae_mmd_losses, wae_recon_losses = [], [], [], []

        # SAS
        sas_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.o_history_sample(
                batch_size,
                history_len=self.context_len,
                st_future_len=self.future_len
            )
            self.key, key = jax.random.split(self.key, 2)
            rng, infos, new_models = _metla_offline_sac_update(
                rng=key,
                gamma=self.gamma,
                tau=self.tau,
                actor=self.actor,
                critic=self.critic,
                critic_target=self.critic_target,
                log_ent_coef=self.log_ent_coef,
                ae=self.ae,
                second_ae=self.second_ae,
                sas_predictor=self.sas_predictor,
                history_observations=replay_data.history.observations,
                history_actions=replay_data.history.actions,
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.st_future.observations[:, 0, :],
                dones=jnp.zeros((batch_size, 1)),
                future_observations=replay_data.st_future.observations,
                future_actions=replay_data.st_future.actions
            )

            # self._create_aliases()
            self.apply_update(new_models)

            # Actor
            bc_losses.append(infos["bc_loss"])
            sac_actor_losses.append(infos["sac_actor_loss"])
            goal_reaching_losses.append(infos["goal_reaching_loss"])

            # Critic
            critic_losses.append(infos["critic_loss"])

            # AE
            gae_losses.append(infos["gae_loss"])
            wae_losses.append(infos["wae_loss"])
            wae_mmd_losses.append(infos["wae_mmd_loss"])
            wae_recon_losses.append(infos["wae_recon_loss"])

            # SAS
            sas_losses.append(infos["sas_loss"])

        self._n_updates += gradient_steps
        self.logger.record("config/offline_normalize", self.offline_data_normalizing, exclude="tensorboard")
        self.logger.record("config/context_len", self.context_len, exclude="tensorboard")
        self.logger.record("config/future_len", self.future_len, exclude="tensorboard")
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Actor
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/sac_loss", np.mean(sac_actor_losses))
        self.logger.record("train/goal_loss", np.mean(goal_reaching_losses))

        # Critic
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        # AE
        self.logger.record("train/gae_loss", np.mean(gae_losses))
        self.logger.record("train/wae_loss", np.mean(wae_losses))
        self.logger.record("train/wae_mmd_loss", np.mean(wae_mmd_losses))
        self.logger.record("train/wae_recon_loss", np.mean(wae_recon_losses))

    def get_finetune_loss_fn(self):
        return _metla_sac_online_finetune_last_layer

    def get_finetune_input(self, batch_size: int, context_len: int, future_len: int) -> Dict:
        replay_data = self.replay_buffer.metla_sample(
            batch_size=batch_size,
            history_len=context_len,
            future_len=future_len
        )
        self.key, key = jax.random.split(self.key)
        finetune_input = {
            "rng": self.key,
            "gamma": self.gamma,
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_ent_coef": self.log_ent_coef,
            "second_ae": self.second_ae,
            "finetune_last_layer": self.finetune_last_layer,
            "history_observations": replay_data.history_observations,
            "history_actions": replay_data.history_actions,
            "observations": replay_data.observations,
            "actions": replay_data.actions,
            "next_observations": replay_data.st_future_observations[:, 0, :],
            "rewards": replay_data.rewards,
            "dones": replay_data.dones,
        }
        return finetune_input


    def _excluded_save_params(self) -> List[str]:
        return super(METLAMLE, self)._excluded_save_params() \
               + ["actor", "critic", "critic_target", "log_ent_coef", "ae", "second_ae", "sas_predictor"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {
            "actor": self.actor.params,
            "critic": self.critic.params,
            "critic_target": self.critic_target.params,
            "log_ent_coef": self.log_ent_coef.params,
            "ae": self.ae.params,
            "second_ae": self.second_ae.params,
            "sas_predictor": self.sas_predictor.params,
        }
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ["actor", "critic", "critic_target", "log_ent_coef", "ae", "second_ae", "sas_predictor"]