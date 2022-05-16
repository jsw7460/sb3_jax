import functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Schedule

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):            # NOTE: 현재 Policy는 state와 latent의 concat을 input으로 받도록 할 것이다.
    features_extractor: nn.Module
    action_space: gym.spaces.Space
    latent_dim: int
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.relu

    latent_pi = None
    mu = None
    log_std = None

    def setup(self):            # NOTE: 현재 Policy는 state와 latent의 concat을 input으로 받도록 할 것이다.
        action_dim = self.action_space.shape[0]
        self.latent_pi = create_mlp(
            output_dim=self.latent_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False,
        )
        self.mu = create_mlp(
            output_dim=action_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False,
        )
        self.log_std = create_mlp(
            output_dim=action_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, latent: jnp.ndarray, deterministic: bool = False):
        mean_actions, log_stds = self.get_action_dist_params(observations, latent, deterministic=deterministic)
        return self.actions_from_params(mean_actions, log_stds)

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        latent: jnp.ndarray,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.hstack((observations, latent))
        x = self.features_extractor(x)

        latent_pi = self.latent_pi(x, deterministic=deterministic)
        mean_actions = self.mu(latent_pi, deterministic=deterministic)
        log_stds = self.log_std(latent_pi, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_stds

    def actions_from_params(
        self,
        mean_actions: jnp.ndarray,
        log_std: jnp.ndarray,
    ):
        # From mean and log std, return the actions by applying the tanh nonlinear transformation.
        base_dist = tfd.MultivariateNormalDiag(loc=mean_actions, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist

    # This is for deterministic action: no sampling, just return "mean"
    def deterministic_action(self, x: jnp.ndarray, deterministic: bool = False):
        mean_actions, *_ = self.get_action_dist_params(x, deterministic=deterministic)
        return mean_actions

    def _predict(self, observation: jnp.ndarray, conditioned_latent: jnp.ndarray, deterministic: bool) -> np.ndarray:
        # 여기서의 observation은 latent가 concatenate되어있는 말 그대로 input 그 자체
        rng, dropout_key, sample_key = jax.random.split(self.key, 3)
        action_dist = self.forward(observations=observation, latent=conditioned_latent, deterministic=deterministic)
        action = action_dist.sample(seed=sample_key)
        return action


class SingleCritic(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        latent: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool
    ) -> jnp.ndarray:
        q_net_input = jnp.concatenate((observations, latent, actions), axis=1)
        return self.q_net(q_net_input, deterministic=deterministic)


class Critics(nn.Module):           # n_critics mode.
    features_extractor: nn.Module
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            SingleCritic,
            in_axes=None,
            out_axes=1,
            variable_axes={"params": 1},
            split_rngs={"params": True},
            axis_size=self.n_critics,
        )
        self.q_networks = batch_qs(self.features_extractor, self.net_arch, self.activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        latent: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False
    ) -> jnp.ndarray:

        return self.q_networks(observations, latent, actions, deterministic)


class SACPolicy(object):
    def __init__(
        self,
        key: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        latent_dim: int,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        n_critics: int = 2,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.key = key

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        features_extractor = features_extractor_class(
            _observation_space=observation_space,
            **features_extractor_kwargs
        )

        init_observation = observation_space.sample()[jnp.newaxis, ...]
        init_latent = jax.random.normal(self.key, shape=(1, latent_dim))
        init_action = self.action_space.sample()[jnp.newaxis, ...]

        param_key, dropout_key, action_key = jax.random.split(self.key, 3)
        actor_rngs = {"params": param_key, "dropout": dropout_key, "action_sample": action_key}
        actor_def = Actor(
            features_extractor=features_extractor,
            action_space=action_space,
            latent_dim=latent_dim,
            net_arch=actor_arch,
            activation_fn=activation_fn,
            dropout=dropout,
        )
        actor = Model.create(
            actor_def,
            inputs=[actor_rngs, init_observation, init_latent],
            tx=optax.adam(learning_rate=lr_schedule)
        )

        critic_rngs = {"params": param_key, "dropout": dropout_key, "action_sample": action_key}
        critic_def = Critics(
            features_extractor=features_extractor,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout=dropout,
            n_critics=n_critics
        )
        critic = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_latent, init_action],
            tx=optax.adam(learning_rate=lr_schedule)
        )
        critic_target = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_latent, init_action],
            tx=optax.adam(learning_rate=lr_schedule)
        )

        self.n_critics = n_critics
        self.latent_dim = latent_dim
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target

    def _predict(self, observation: jnp.ndarray, conditioned_latent: jnp.ndarray, deterministic: bool) -> np.ndarray:
        raise NotImplementedError()

    def predict(self, observation: jnp.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError()

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


MlpPolicy = SACPolicy
