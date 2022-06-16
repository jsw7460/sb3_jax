from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import functools

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, InfoDict, ReplayBufferSamples, Params
from offline_baselines_jax.td3.policies import TD3Policy

from .core import update_td3



class TD3(OffPolicyAlgorithm):

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
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        alpha: int = 2.5,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
    ):

        super(TD3, self).__init__(
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
            action_noise=action_noise,
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
        if without_exploration and gradient_steps == -1:
            self.gradient_steps = policy_delay

        self.alpha = alpha
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        actor_losses, critic_losses, coef_lambda = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.rng, key = jax.random.split(self.rng, 2)

            actor_update_cond = self._n_updates % self.policy_delay == 0

            self.rng, new_models, info = \
                update_td3(
                    key,
                    actor=self.actor,
                    actor_target=self.actor_target,
                    critic=self.critic,
                    critic_target=self.critic_target,

                    observations=replay_data.observations,
                    actions=replay_data.actions,
                    next_observations=replay_data.next_observations,
                    rewards=replay_data.rewards,
                    dones=replay_data.dones,

                    actor_update_cond=actor_update_cond,
                    tau=self.tau,
                    target_policy_noise=self.target_policy_noise,
                    target_noise_clip=self.target_noise_clip,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    without_exploration=self.without_exploration
                )

            self.apply_new_models(new_models)
            self.actor_target = new_models["actor_target"]
            self.policy.actor_target = new_models["actor_target"]

            actor_losses.append(info['actor_loss'])
            critic_losses.append(info['critic_loss'])
            coef_lambda.append(info['coef_lambda'])


        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/coef", np.mean(coef_lambda))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(TD3, self).learn(
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
        return super(TD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['actor_target'] = self.actor_target.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'actor_target']

    def _load_policy(self) -> None:
        super(TD3, self)._load_policy()
        self.policy.actor_target = self.actor_target
