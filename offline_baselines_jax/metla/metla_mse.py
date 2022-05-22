from typing import Any, Dict, List, Optional, Tuple, Type, Union
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.type_aliases import GymEnv, Schedule, Params
from .core import (
    _metla_offline_td3_update,
    _metla_td3_online_finetune_last_layer,
    _metla_offline_td3_flow_update,
    _metla_online_finetune_warmup_higher_actor,
    _metla_sac_style_higher_actor_finetune,
)
from .metla import METLA
from .policies_mse import TD3Policy

TIMEOUT_LIMIT = 10000000


class METLAMSE(METLA):
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[TD3Policy]] = TD3Policy,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, 'episode'),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = {},
        verbose: int = 0,
        seed: int = 0,
        alpha: int = 2.5,
        _init_setup_model: bool = True,
        without_exploration: bool = False,

        dropout: float = 0.0,
        context_len: int = 20,
        future_len: int = 7,
        normalize: bool = True,
    ):
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.alpha = alpha
        self.bc_losses = deque(maxlen=100)
        self.goal_losses = deque(maxlen=100)

        super(METLAMSE, self).__init__(
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
        super(METLAMSE, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.latent_dim = self.policy.latent_dim

        super(METLAMSE, self)._create_metla_aliases()

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        # Critic
        critic_losses, rand_q_vals = [], []
        target_q_vals, pred_q_vals = [], []

        # AE
        gae_losses = []
        skill_prior_losses = []
        skill_prior_means, skill_prior_log_stds = [], []

        # SAS
        sas_losses = []

        for gradient_step in range(gradient_steps):

            update_flag = 1.0 if (self._n_updates % self.policy_delay == 0) else 0.0

            replay_data = self.replay_buffer.o_history_sample(
                batch_size,
                history_len=self.context_len,
                st_future_len=self.future_len + 1       # Sometimes, we need future for the next step.
            )

            future_obs = replay_data.st_future.observations[:, :self.future_len, ...]
            future_act = replay_data.st_future.actions[:, :self.future_len, ...]

            self.key, key = jax.random.split(self.key, 2)
            rng, infos, new_models = _metla_offline_td3_flow_update(
                rng=key,
                gamma=self.gamma,
                tau=self.tau,
                target_noise=self.target_policy_noise,
                target_noise_clip=self.target_noise_clip,
                update_flag=update_flag,
                actor=self.actor,
                actor_target=self.actor_target,
                critic=self.critic,
                critic_target=self.critic_target,
                ae=self.ae,
                second_ae=self.second_ae,
                sas_predictor=self.sas_predictor,
                history_observations=replay_data.history.observations,
                history_actions=replay_data.history.actions,
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.st_future.observations[:, 0, ...],
                dones=jnp.zeros((batch_size, 1)),
                future_observations=future_obs,
                future_actions=future_act,
            )
            self.apply_update(new_models)
            self._n_updates += 1

            # Actor
            self.bc_losses.append(infos["bc_loss"])
            self.goal_losses.append(infos["goal_reaching_loss"])

            # Critic
            critic_losses.append(infos["critic_loss"])
            rand_q_vals.append(infos["rand_q_val"])
            target_q_vals.append(infos["target_q_val"])
            pred_q_vals.append(infos["pred_q_val"])

            # AE
            gae_losses.append(infos["gae_loss"])
            skill_prior_losses.append(infos["skill_prior_loss"])
            skill_prior_log_stds.append(infos["skill_log_std"])
            skill_prior_means.append(infos["skill_mean"])

            # SAS
            sas_losses.append(infos["sas_loss"])

            self.logger.record("config/normalize", self.offline_data_normalizing, exclude="tensorboard")

            self.logger.record("train/bc_losses", np.mean(self.bc_losses) * self.policy_delay)
            self.logger.record("train/goal_loss", np.mean(self.goal_losses) * self.policy_delay)

            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/target_q_val", np.mean(target_q_vals), exclude="tensorboard")
            self.logger.record("train/pred_q_val", np.mean(pred_q_vals), exclude="tensorboard")
            self.logger.record("train/rand_q_val", np.mean(rand_q_vals), exclude="tensorboard")

            self.logger.record("train/sas_loss", np.mean(sas_losses))
            self.logger.record("train/gae_loss", np.mean(gae_losses))
            self.logger.record("train/skill_prior_loss", np.mean(skill_prior_losses))
            self.logger.record("train/skill_mean", np.mean(skill_prior_means))
            self.logger.record("train/skill_log_std", np.mean(skill_prior_log_stds))

    def get_finetune_loss_fn(self, finetune: str):
        if "last" in finetune:
            raise NotImplementedError()
        elif "higher" in finetune:
            self.finetune_ft = _metla_sac_style_higher_actor_finetune
            self.warmup_ft = _metla_online_finetune_warmup_higher_actor
        else:
            raise NotImplementedError()

    def get_finetune_input(self, batch_size: int, context_len: int, future_len: int) -> Dict:
        replay_data = self.replay_buffer.metla_sample(
            batch_size=batch_size,
            history_len=context_len,
            future_len=future_len
        )
        self.key, key = jax.random.split(self.key)

        finetune_input = {
            "rng": key,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "second_ae": self.second_ae,
            "higher_actor": self.higher_actor,
            "history_observations": replay_data.history_observations,
            "history_actions": replay_data.history_actions,
            "observations": replay_data.observations,
            "actions": replay_data.actions,
            "higher_actions": replay_data.higher_actions,
            "next_observations": replay_data.st_future_observations[:, 0, :],
            "rewards": replay_data.rewards,
            "dones": replay_data.dones,
            "target_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "higher_critic": self.higher_critic,
            "higher_critic_target": self.higher_critic_target,
        }
        return finetune_input

    def _excluded_save_params(self) -> List[str]:
        return super(METLAMSE, self)._excluded_save_params() \
               + ["actor",
                  "actor_target",
                  "critic",
                  "critic_target",
                  "ae",
                  "second_ae",
                  "sas_predictor"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {
            "actor": self.actor.params,
            "actor_target": self.actor_target.params,
            "critic": self.critic.params,
            "critic_target": self.critic_target.params,
            "ae": self.ae.params,
            "second_ae": self.second_ae.params,
            "sas_predictor": self.sas_predictor.params,
        }
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ["actor", "actor_target", "critic", "critic_target", "ae", "second_ae", "sas_predictor"]