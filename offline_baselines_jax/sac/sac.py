from typing import Any, Dict, List, Optional, Tuple, Type, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, Params
from offline_baselines_jax.sac.policies import SACPolicy
from .core import sac_update


class LogEntropyCoef(nn.Module):
    init_value: float = 1.0
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SAC(OffPolicyAlgorithm):
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
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
            without_exploration=without_exploration,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.entropy_update = True

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
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
            self.rng, temp_key = jax.random.split(self.rng, 2)
            self.log_ent_coef = Model.create(
                log_ent_coef_def,
                inputs=[temp_key],
                tx=optax.adam(learning_rate=self.lr_schedule(1))
            )

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(self.ent_coef)
            self.rng, temp_key = jax.random.split(self.rng, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            self.entropy_update = False

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size=batch_size)

            self.rng, key = jax.random.split(self.rng, 2)
            target_update_cond = gradient_step % self.target_update_interval == 0

            self.rng, new_models, info \
                = sac_update(
                    rng=key,
                    actor=self.actor,
                    critic=self.critic,
                    critic_target=self.critic_target,
                    log_ent_coef=self.log_ent_coef,

                    observations=replay_data.observations,
                    actions=replay_data.actions,
                    rewards=replay_data.rewards,
                    next_observations=replay_data.next_observations,
                    dones=replay_data.dones,

                    gamma=self.gamma,
                    target_entropy=self.target_entropy,
                    tau=self.tau,
                    target_update_cond=target_update_cond,
                    entropy_update=self.entropy_update
            )

            ent_coef_losses.append(info['ent_coef_loss'])
            ent_coefs.append(info['ent_coef'])
            critic_losses.append(info['critic_loss'])
            actor_losses.append(info['actor_loss'])

            self.apply_new_models(new_models)

        self._n_updates += gradient_steps
        if self.num_timesteps % 500 == 0:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        raise NotImplementedError()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + SAC_COMPONENTS

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        for comp_str in SAC_COMPONENTS:
            comp = getattr(self, comp_str)
            params_dict[comp_str] = comp.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return SAC_COMPONENTS


SAC_COMPONENTS = ["actor", "critic", "critic_target", "log_ent_coef"]