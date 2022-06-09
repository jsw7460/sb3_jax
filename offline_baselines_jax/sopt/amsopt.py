import functools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from collections import deque

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
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    Params,
    RolloutReturn
)
from offline_baselines_jax.common.utils import should_collect_more_steps
from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy
from .buffer import SensorBasedExpertBuffer
from . import core

from .networks import (
    SensorBasedSingleStateActionMatcherFromHighToLow,
    SensorBasedSingleStateDiscriminator,
    SensorBasedDoubleStateDiscriminator,
    SensorBasedInverseDynamics,
    NaiveSensorBasedBehaviorCloner
)

from offline_baselines_jax.common.jax_layers import (
    CombinedExtractor,
    FlattenExtractor,
)

ENV_MAX_LEN = 2000


@functools.partial(jax.jit, static_argnames=('actor_apply_fn',))
def sample_actions(
    rng: int,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray
) -> Tuple[int, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=("disc_apply_fn", ))        # Forward path of observations to discriminator
def sample_intrinsic_rewards(
    disc_apply_fn: Callable[..., Any],
    disc_params: Params,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray
):
    disc_score = disc_apply_fn({"params": disc_params}, observations, next_observations)
    return jnp.mean(disc_score)


class LogEntropyCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SensorBasedActionMatcherSoptSAC(OffPolicyAlgorithm):
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[SACPolicy]] = MultiInputPolicy,
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
        dropout: float = 0.0,
        intrinsic_rew_coef: float = 0.0,

        n_stack: int = 0,
        update_ft_str: str = None,
    ):

        super(SensorBasedActionMatcherSoptSAC, self).__init__(
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
            dropout=dropout
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.update_entropy = True

        self.bc_timestep = 0
        self.behavior_cloner = None
        self.n_stack = n_stack
        self.so_buffer = None # type: StateOnlyBuffer

        if _init_setup_model:
            self._setup_model()

        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval

        self.dropout = dropout
        self.intrinsic_rew_coef = intrinsic_rew_coef

        self.update_ft_str = update_ft_str
        self.update_ft = getattr(core, update_ft_str)

    def set_expert_buffer(self, path: str, n_frames: int):
        self.so_buffer = SensorBasedExpertBuffer(path, n_frames)
        self.so_buffer.relabel_action_by_obs_difference()

    def _setup_model(self) -> None:
        super(SensorBasedActionMatcherSoptSAC, self)._setup_model()
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
            self.log_ent_coef = Model.create(
                log_ent_coef_def, inputs=[temp_key],
                tx=optax.adamw(learning_rate=self.lr_schedule(1))
            )

        else:
            # Force conversion to float
            # this will throw an erwror if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(self.ent_coef)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            self.update_entropy = False

        # Define init observation and lower action dimension
        features_extractor_class = None
        init_obs = None
        low_action_dim = None
        highaction_dim = self.action_space.shape[0]
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            features_extractor_class = CombinedExtractor
            low_action_dim = self.observation_space.sample()["observations"].shape[-1]
            init_obs = self.observation_space.sample()
            for k, v in init_obs.items():
                v = v[np.newaxis, ...]
                init_obs.update({k: v})

        elif isinstance(self.env.observation_space, gym.spaces.Box):
            features_extractor_class = FlattenExtractor
            low_action_dim = self.observation_space.sample().shape[-1]
            init_obs = self.observation_space.sample()[np.newaxis, ...]

        # NOTE: Action matcher
        self.key, params_key, dropout_key = jax.random.split(self.key, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        action_matcher_def = SensorBasedSingleStateActionMatcherFromHighToLow(
            features_extractor=features_extractor,
            dropout=self.dropout,
            highaction_dim=highaction_dim,
            squash_output=True
        )
        self.action_matcher = Model.create(
            action_matcher_def,
            inputs=[rngs, init_obs, np.zeros((1, low_action_dim))],
            tx=optax.adamw(learning_rate=self.learning_rate)
        )

        # NOTE: Single state based discriminator
        self.key, params_key, dropout_key = jax.random.split(self.key, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        discriminator_def = SensorBasedSingleStateDiscriminator(
            features_extractor=features_extractor,
            dropout=self.dropout
        )
        self.single_state_discriminator = Model.create(
            discriminator_def,
            inputs=[rngs, init_obs],
            tx=optax.adamw(learning_rate=self.learning_rate)
        )

        # NOTE: Double state based discriminator
        self.key, params_key, dropout_key = jax.random.split(self.key, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        discriminator_def = SensorBasedDoubleStateDiscriminator(
            features_extractor=features_extractor,
            dropout=self.dropout
        )
        self.double_state_discriminator = Model.create(
            discriminator_def,
            inputs=[rngs, init_obs, init_obs],
            tx=optax.adamw(learning_rate=self.learning_rate)
        )

        # NOTE: Behavior cloner
        self.key, params_key, dropout_key = jax.random.split(self.key, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        bc_def = NaiveSensorBasedBehaviorCloner(
            features_extractor=features_extractor,
            lowaction_dim=low_action_dim,                        # Test: = Sensor data dimension
            dropout=self.dropout
        )
        self.behavior_cloner = Model.create(
            bc_def,
            inputs=[rngs, init_obs],
            tx=optax.adamw(learning_rate=self.learning_rate)
        )

        # NOTE: Inverse dynamics
        self.key, params_key, dropout_key = jax.random.split(self.key, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        inv_dyna_def = SensorBasedInverseDynamics(
            features_extractor=features_extractor,
            dropout=self.dropout,
            highaction_dim=highaction_dim,
            squash_output=True
        )
        self.inverse_dynamics = Model.create(
            inv_dyna_def,
            inputs=[rngs, init_obs, init_obs],
            tx=optax.adamw(learning_rate=self.learning_rate)
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def set_behavior_cloner(self, path: str):
        try:
            with open(path, "rb") as f:
                params = f.read()
                self.behavior_cloner = self.behavior_cloner.load_dict(params)
            return True

        except:
            return False

    def train_bc(self, num_timesteps: int = 500_000, batch_size: int = 64, bc_path: str = None):
        bc_losses = deque(maxlen=5000)
        while self.bc_timestep < num_timesteps:
            rng, _ = jax.random.split(self.key)
            observations, actions, _ = self.so_buffer.sample(batch_size)

            behavior_cloner, bc_info = core.behavior_cloner_update(
                rng=rng,
                behavior_cloner=self.behavior_cloner,
                expert_observations=observations,
                expert_actions=actions
            )
            self.behavior_cloner = behavior_cloner
            bc_losses.append(np.mean(bc_info["bc_loss"]))

            self.bc_timestep += 1
            if self.bc_timestep % 5000 == 0:
                print("*" * 10 + "BC" + "*" * 10)
                print("Timestep", self.bc_timestep)
                print("Loss", np.mean(bc_losses))
                save_path = bc_path + f"-bc{self.bc_timestep}"
                self.behavior_cloner.save_dict(save_path)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        actor_losses = []

        ent_coef_losses =[]
        ent_coefs = []

        critic_losses =[]

        single_disc_losses, single_expert_disc_scores, single_policy_disc_scores = [], [], []
        double_disc_losses, double_expert_disc_scores, double_policy_disc_scores = [], [], []

        inverse_dynamics_losses = []

        action_matcher_losses = []

        training_info = None
        for train_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size=batch_size)
            expert_observation, _, expert_next_observation = self.so_buffer.sample(batch_size=batch_size)
            # self.key, new_models, training_info = self.update_ft(
            self.key, new_models, training_info = core.dummy_sac_update(
                rng=self.key,
                log_ent_coef=self.log_ent_coef,
                actor=self.actor,
                critic=self.critic,
                critic_target=self.critic_target,
                behavior_cloner=self.behavior_cloner,
                single_state_discriminator=self.single_state_discriminator,
                double_state_discriminator=self.double_state_discriminator,
                action_matcher=self.action_matcher,
                inverse_dynamics=self.inverse_dynamics,

                observations=replay_data.observations,
                actions=replay_data.actions,
                rewards=replay_data.rewards,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,

                expert_observation=expert_observation,
                expert_next_observation=expert_next_observation,

                update_entropy=self.update_entropy,
                target_entropy=self.target_entropy,
                gamma=self.gamma,
                tau=self.tau

            )
            self.apply_new_models(new_models)

        if self.num_timesteps % 1000 == 0:
            actor_losses.append(training_info["actor_loss"])
            critic_losses.append(training_info["critic_loss"])
            ent_coef_losses.append(training_info["ent_coef_loss"])
            ent_coefs.append(training_info["ent_coef"])

            self.logger.record_mean("train/ent_coef", np.mean(np.array(ent_coefs)))
            self.logger.record_mean("train/actor_loss", np.mean(np.array(actor_losses)))
            self.logger.record_mean("train/critic_loss", np.mean(np.array(critic_losses)))
            self.logger.record_mean("train/ent_coef_loss", np.mean(np.array(ent_coef_losses)))

            action_matcher_losses.append(training_info["action_matcher_loss"])

            single_disc_losses.append(training_info["single_discriminator_loss"])
            single_expert_disc_scores.append(training_info["single_expert_disc_score"])
            single_policy_disc_scores.append(training_info["single_policy_disc_score"])

            double_disc_losses.append(training_info["double_discriminator_loss"])
            double_expert_disc_scores.append(training_info["double_expert_disc_score"])
            double_policy_disc_scores.append(training_info["double_policy_disc_score"])

            inverse_dynamics_losses.append(training_info["inverse_dynamics_loss"])

            self.logger.record_mean("train/action_matcher_loss", np.mean(np.array(action_matcher_losses)))

            self.logger.record_mean("train/single_disc_loss", np.mean(np.array(single_disc_losses)))
            self.logger.record_mean("train/single_expert_disc_score", np.mean(np.array(single_expert_disc_scores)))
            self.logger.record_mean("train/single_policy_disc_score", np.mean(np.array(single_policy_disc_scores)))

            self.logger.record_mean("train/double_disc_loss", np.mean(np.array(double_disc_losses)))
            self.logger.record_mean("train/double_expert_disc_score", np.mean(np.array(double_expert_disc_scores)))
            self.logger.record_mean("train/double_policy_disc_score", np.mean(np.array(double_policy_disc_scores)))

            self.logger.record_mean("train/inv_dyna_loss", np.mean(np.array(inverse_dynamics_losses)))


    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        num_collected_steps, num_collected_episodes = 0, 0

        callback.on_rollout_start()
        continue_training = True

        intrinsic_rewards = 0
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            new_obs, original_rewards, dones, infos = env.step(actions)

            # intrinsic_reward = sample_intrinsic_rewards(
            #     self.discriminator.apply_fn,
            #     self.discriminator.params,
            #     self._last_obs,
            #     new_obs
            # )
            # rewards = original_rewards + self.intrinsic_rew_coef * intrinsic_reward
            rewards = original_rewards

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            # Store data in replay buffer (normalized action and unnormalized observation)

            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    current_ep_len = infos[0].get("episode", None)["l"]
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if ENV_MAX_LEN is not None:
                        current_ep_len = infos[0].get("episode", None)["l"]
                        is_success = int(current_ep_len < ENV_MAX_LEN)
                        self.ep_success_buffer.append(is_success)

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        if intrinsic_rewards != 0:
                            self.logger.record("config/use_intrinsic_rew", 1, exclude="tensorboard")
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
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

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs.copy()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> jnp.ndarray:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.policy.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action


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
        tb_log_name: str = "SensorBasedActionMatcherSoptSAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SensorBasedActionMatcherSoptSAC, self).learn(
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
        return super(SensorBasedActionMatcherSoptSAC, self)._excluded_save_params() \
               + ["actor", "critic", "critic_target", "log_ent_coef"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['log_ent_coef'] = self.log_ent_coef.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'log_ent_coef']
