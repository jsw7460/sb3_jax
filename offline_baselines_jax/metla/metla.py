from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import flax.linen as nn
import gym
from gym.spaces import Box
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import create_mlp, FlattenExtractor
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import GymEnv, Schedule, Params
from offline_baselines_jax.common.type_aliases import MaybeCallback
from offline_baselines_jax.common.type_aliases import (
    TrainFreq,
    TrainFrequencyUnit,
    RolloutReturn
)
from offline_baselines_jax.common.utils import configure_logger
from offline_baselines_jax.common.utils import should_collect_more_steps
from .buffer import TrajectoryBuffer, FinetuneReplayBuffer
from .metla_eval import get_policy_input, get_policy_input_with_last_layer, _predict_mle, _predict_mse
from .networks import SASPredictor, GeneralizedAutoEncoder, WassersteinAutoEncoder, GaussianSkillPrior
from .policies_mle import SACPolicy
from .policies_mse import HigherCritics

TIMEOUT_LIMIT = 10000000


class LogEntropyCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class METLA(OffPolicyAlgorithm):
    policy: Union[str, Type[SACPolicy]]
    env: Union[GymEnv, str]
    learning_rate: Union[float, Schedule] = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = 1
    gradient_steps: int = 1
    action_noise: Optional[ActionNoise] = None
    replay_buffer_class: Optional[ReplayBuffer] = None
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: bool = False
    tensorboard_log: Optional[str] = None
    create_eval_env: bool = False
    policy_kwargs: Optional[Dict[str, Any]] = {}
    verbose: int = 0
    seed: int = 0
    _init_setup_model: bool = True
    without_exploration: bool = False

    dropout: float = 0.0
    context_len: int = 20
    future_len: int = 7
    normalize: bool = True

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
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
        # self.latent_dim = policy_kwargs["latent_dim"]
        super(METLA, self).__init__(
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


        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale

        try:
            self.env_name = self.env.get_attr("unwrapped", 0)[0].spec.id
        except AttributeError:
            self.env_name = None

        self.latent_dim = None

        self.observation_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.context_len = context_len
        self.future_len = future_len
        self.normalize = normalize
        self.offline_data_normalizing = None

        self.online_finetune_init = True
        self.online_finetune_step = None
        self.online_learning_starts = 20
        self._initial_rewards = None
        self.warmup = None

        self._history_observations = []
        self._history_actions = []

        self.ae = None
        self.sas_predictor = None
        self.second_ae = None
        self.critic = None
        self.critic_target = None
        self.actor = None
        self.actor_target = None
        self.finetune = None
        self.warmup_ft = None
        self.finetune_ft = None
        self.higher_actor = None     # Finetune last layr시에 Goal generator 뒤쪽에 하나 붙이게 될 MLP
        self.action_predict = None
        self.higher_critic = None
        self.higher_critic_target = None

        if _init_setup_model:
            self._setup_model()

    def load_data(self, data_path: str):
        self.replay_buffer = TrajectoryBuffer(
            data_path=data_path,
            observation_dim=self.observation_space.shape[0],
            action_dim=self.action_space.shape[0],
            normalize=self.normalize
        )
        self.offline_data_normalizing = self.replay_buffer.normalizing_factor

    def _setup_model(self) -> None:
        super(METLA, self)._setup_model()

    def _create_metla_aliases(self) -> None:
        init_observation = self.observation_space.sample()[jnp.newaxis, ...]
        init_action = self.action_space.sample()[jnp.newaxis, ...]
        init_latent = jnp.zeros((1, self.latent_dim))
        init_history = jnp.hstack((init_observation, init_action))
        init_history = jnp.repeat(init_history, repeats=self.context_len, axis=0)
        init_future = jnp.hstack((init_observation, init_action))
        init_future = jnp.repeat(init_future, repeats=self.future_len, axis=0)

        sas_predictor_def = SASPredictor(
            state_dim=self.latent_dim,
            net_arch=[256, 256],
            dropout=self.dropout,
            squash_output=False
        )
        param_key, dropout_key = jax.random.split(self.key)
        sas_rngs = {"params": param_key, "dropout": dropout_key}
        self.sas_predictor = Model.create(
            sas_predictor_def,
            inputs=[sas_rngs, init_latent, init_action],
            tx=optax.radam(learning_rate=self.learning_rate),
        )

        gae_def = GeneralizedAutoEncoder(
            recon_dim=self.observation_space.shape[0],
            latent_dim=self.latent_dim,
            squashed_out=self.normalize,
            dropout=self.dropout,
            n_nbd=5
        )
        param_key, dropout_key = jax.random.split(param_key)
        ae_rngs = {"params": param_key, "dropout": dropout_key}
        self.ae = Model.create(
            gae_def,
            inputs=[ae_rngs, init_history, init_observation, init_future],
            tx=optax.radam(learning_rate=self.learning_rate)
        )

        second_ae_def = GaussianSkillPrior(
            recon_dim=self.latent_dim,      # Dimension of skill
            dropout=self.dropout,
        )
        param_key, dropout_key, noise_key = jax.random.split(param_key, 3)
        second_ae_rngs = {"params": param_key, "dropout": dropout_key}
        self.second_ae = Model.create(
            second_ae_def,
            inputs=[second_ae_rngs, init_observation],
            tx=optax.radam(learning_rate=self.learning_rate)
        )

    def apply_update(self, new_models: Dict):
        for k, v in new_models.items():
            assert hasattr(self, k), f"unexpected attribute {k}"
            if k == "actor":
                self.policy.actor = v
                self.actor = v
            elif k == "actor_target":
                self.actor_target = v
                self.policy.actor_target = v
            elif k == "critic":
                self.policy.critic = v
                self.critic = v
            elif k == "critic_target":
                self.policy.critic_target = v
                self.critic_target = v
            else:
                setattr(self, k, v)

    def online_finetune_setup(
        self,
        finetune: str,
        initial_rewards: float,
        warmup: int = 2,
    ):
        if self.observation_dim == -1:
            self.observation_dim = self.observation_space.shape[0]
        if self.action_dim == -1:
            self.action_dim = self.action_space.shape[0]

        self.without_exploration = False
        self._initial_rewards = initial_rewards
        self.warmup = warmup
        self.online_finetune_step = 0
        self.batch_size = 256
        self.verbose = 1
        self.gradient_steps = 1
        self.finetune = finetune

        init_history = jnp.zeros((1, self.observation_dim + self.action_space.shape[0]))
        init_history = jnp.repeat(init_history, repeats=self.context_len, axis=0)
        init_observation = self.observation_space.sample()[jnp.newaxis, ...]
        init_latent = jnp.zeros((1, self.latent_dim))

        higher_features_extractor = FlattenExtractor(
            _observation_space=None,
            _feature_dim=0
        )

        param_key, dropout_key = jax.random.split(self.key)

        higher_critic_def = HigherCritics(
            features_extractor=higher_features_extractor,
            net_arch=[256, 256],
            dropout=self.dropout,
        )
        param_key, dropout_key = jax.random.split(param_key)
        higher_critic_rngs = {"params": param_key, "dropout": dropout_key}
        self.higher_critic = Model.create(
            higher_critic_def,
            inputs=[higher_critic_rngs, init_observation, init_latent, False],
            tx=optax.radam(learning_rate=self.learning_rate)
        )
        self.higher_critic_target = Model.create(
            higher_critic_def,
            inputs=[higher_critic_rngs, init_observation, init_latent, False],
            tx=optax.radam(learning_rate=self.learning_rate)
        )

        try:
            getattr(self, "actor_target")
            self.action_predict = _predict_mse

        except AttributeError:
            self.action_predict = _predict_mle

        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            reset_num_timesteps=False
        )

        higher_action_space = Box(low=-10.0, high=10.0, shape=(self.latent_dim, ))
        self.replay_buffer = FinetuneReplayBuffer(
            buffer_size=5000,
            observations_space=self.env.observation_space,
            lower_action_space=self.action_space,
            higher_action_space=higher_action_space,
            n_envs=1
        )

        self._history_observations = []
        self._history_actions = []
        self.observation_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        if "finetune_generator_flow" in finetune:
            raise NotImplementedError()

        elif "finetune_only_generator" in finetune:
            raise NotImplementedError()

        elif "last" in finetune:
            raise NotImplementedError()

        elif "higher" in finetune:
            higher_actor_def = GaussianSkillPrior(
                recon_dim=self.latent_dim,
                dropout=self.dropout
            )
            param_key, dropout_key, noise_key = jax.random.split(self.key, 3)
            higher_actor_rngs = {"params": param_key, "dropout": dropout_key}
            self.higher_actor = Model.create(
                higher_actor_def,
                inputs=[higher_actor_rngs, init_observation],
                tx=optax.radam(learning_rate=self.learning_rate)
            )
            self.get_finetune_loss_fn("higher")
            self.tensorboard_log += "-ft_higher_actor"

            # Copy the parameter for initialization
            self.higher_actor = self.higher_actor.replace(params=self.second_ae.params)

        else:
            raise NotImplementedError()

    def _get_policy_input(self, observation: jnp.ndarray):
        self.key, dropout_key, noise_key = jax.random.split(self.key, 3)
        if observation.ndim == 3:
            observation = observation.squeeze(1)
        elif observation.ndim == 1:
            observation = observation[jnp.newaxis, ...]

        # if len(self._history_observations) == 0:
        #     history = jnp.zeros((1, self.observation_dim + self.action_dim))
        #     history = jnp.repeat(history, repeats=self.context_len, axis=0)
        #     conditioning_latent, *_ = self.second_ae(
        #         history,
        #         observation,
        #         deterministic=True,
        #         rngs={"dropout_key": dropout_key, "noise": noise_key}
        #     )
        #     return observation, conditioning_latent, history, None

        # else:
        # history_observation = np.vstack(self._history_observations)[-self.context_len:, ...]
        # history_action = np.vstack(self._history_actions)[-self.context_len:, ...]
        # cur_hist_len = len(history_observation)
        # hist_padding_obs = jnp.zeros((self.context_len - cur_hist_len, self.observation_dim))
        # hist_padding_act = jnp.zeros((self.context_len - cur_hist_len, self.action_dim))
        if self.higher_actor is None:
            return get_policy_input(
                key=self.key,
                vae=self.second_ae,
                observation=observation,
            )

        else:
            if "last" in self.finetune:
                raise NotImplementedError()

            elif "higher" in self.finetune:
                return get_policy_input(
                    key=self.key,
                    higher_actor=self.higher_actor,
                    observation=observation,
                )
            else:
                raise NotImplementedError()

    def metla_sample_action(self, observations: jnp.ndarray):
        self.key, sampling_key, dropout_key = jax.random.split(self.key, 3)
        observations = deepcopy(observations)
        policy_observation, conditioned_latent, *_ = self._get_policy_input(observations)

        action = self.action_predict(
            key=self.key,
            actor=self.actor,
            observations=observations,
            conditioned_latent=conditioned_latent
        )

        return action, conditioned_latent

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        prior_losses = []

        finetune_input = self.get_finetune_input(
            batch_size=batch_size,
            context_len=self.context_len,
            future_len=self.future_len
        )

        if self.warmup > 0:
            raise NotImplementedError()
        else:
            rng, infos, new_models = self.finetune_ft(**finetune_input)

            actor_losses.append(infos["actor_loss"])
            critic_losses.append(infos["critic_loss"])
            prior_losses.append(infos["prior_loss"])

        # print("RuN", self.higher_actor.params["latent_pi"]["Dense_0"])
        self.apply_update(new_models)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        exit()

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(critic_losses) > 0:
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(prior_losses) > 0:
            self.logger.record("train/prior_loss", np.mean(prior_losses))

        if len(ent_coefs) > 0:
            self.logger.record("train/ent_coef", np.mean(ent_coefs))

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(critic_losses) > 0:
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: FinetuneReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            # Select action randomly or according to policy
            actions, higher_actions = self.metla_sample_action(self._last_obs.copy() / self.offline_data_normalizing)
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self._history_observations.append(self._last_obs.copy() / self.offline_data_normalizing)
            self._history_actions.append(actions.copy())
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition_traj(
                replay_buffer,
                actions,
                higher_actions,
                new_obs,
                rewards,
                dones,
                infos,
                metla_normalizing=self.offline_data_normalizing
            )

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    self._history_observations = []
                    self._history_actions = []
                    self.online_finetune_step += 1

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        if self.env_name is not None:
                            self.logger.record("config/env_name", self.env_name)
                        self.logger.record("config/normalize", self.offline_data_normalizing, exclude="tensorboard")
                        self.logger.record("config/init_reward", self._initial_rewards, exclude="tensorboard")
                        self.logger.record("train/finetue_step", self.online_finetune_step)
                        self._dump_logs()
                    if self.online_finetune_step == 100000:
                        print("Train done")
                        exit()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition_traj(
        self,
        replay_buffer: FinetuneReplayBuffer,
        lower_action: np.ndarray,
        higher_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        metla_normalizing: float = 1.0
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add_traj(
            self._last_original_obs / metla_normalizing,
            next_obs / metla_normalizing,
            lower_action,
            higher_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs.copy()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "METLA",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(METLA, self).learn(
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
        return super(METLA, self)._excluded_save_params()

    @abstractmethod
    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_finetune_loss_fn(self, finetune: str):
        raise NotImplementedError()

    @abstractmethod
    def _create_aliases(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_finetune_input(self, batch_size: int, context_len: int, future_len: int):
        raise NotImplementedError()

    @abstractmethod
    def _get_jax_save_params(self) -> Dict[str, Params]:
        raise NotImplementedError()

    @abstractmethod
    def _get_jax_load_params(self) -> List[str]:
        raise NotImplementedError()