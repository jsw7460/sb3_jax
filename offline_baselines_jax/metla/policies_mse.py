from typing import Any, Dict, List, Optional, Type, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Schedule


CLIPPING_CONST = 10.0


class HigherActor(nn.Module):
    features_extractor: nn.Module
    action_dim: int
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu

    mu = None

    def setup(self):
        self.mu = create_mlp(
            output_dim=self.action_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=True
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False):
        x = self.features_extractor(observations)
        return CLIPPING_CONST * jnp.tanh(self.mu(x, deterministic=deterministic))


class Actor(nn.Module):
    features_extractor: nn.Module
    action_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu

    mu = None

    def setup(self):
        action_dim = self.action_space.shape[0]
        self.mu = create_mlp(
            output_dim=action_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=True
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, latent: jnp.ndarray, deterministic: bool = False):
        x = jnp.hstack((observations, latent))
        x = self.features_extractor(x)
        return self.mu(x, deterministic=deterministic)


class HigherSingleCritic(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, actions: jnp.ndarray, deterministic: bool = False):
        q_net_input = jnp.concatenate((observations, actions), axis=1)
        return self.q_net(q_net_input, deterministic=deterministic)


class HigherCritics(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            HigherSingleCritic,
            in_axes=None,
            out_axes=1,
            variable_axes={"params": 1},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_critics
        )
        self.q_networks = batch_qs(self.features_extractor, self.net_arch, self.dropout, self.activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            deterministic: bool
    ) -> jnp.ndarray:
        return self.q_networks(observations, actions, deterministic)


class SingleCritic(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout,
            squash_output=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, latent: jnp.ndarray, actions: jnp.ndarray, deterministic: bool):
        q_net_input = jnp.concatenate((observations, latent, actions), axis=1)
        return self.q_net(q_net_input, deterministic=deterministic)


class Critics(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.leaky_relu
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            SingleCritic,
            in_axes=None,
            out_axes=1,
            variable_axes={"params": 1},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_critics
        )
        self.q_networks = batch_qs(self.features_extractor, self.net_arch, self.dropout, self.activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self,
            observations: jnp.ndarray,
            latent: jnp.ndarray,
            actions: jnp.ndarray,
            deterministic: bool
    ) -> jnp.ndarray:
        return self.q_networks(observations, latent, actions, deterministic)


class TD3Policy(object):
    def __init__(
        self,
        key,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        latent_dim: int,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.leaky_relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        dropout: int = 2,
        n_critics: int = 2,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.key = key
        self.latent_dim = latent_dim

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [400, 300]
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        features_extractor = features_extractor_class(
            _observation_space=observation_space,
            **features_extractor_kwargs
        )

        init_observation = observation_space.sample()[jnp.newaxis, ...]
        init_latent = jnp.zeros((1, latent_dim))
        init_action = self.action_space.sample()[jnp.newaxis, ...]

        param_key, dropout_key, action_key = jax.random.split(self.key, 3)
        actor_rngs = {"params": param_key, "dropout": dropout_key}
        actor_def = Actor(
            features_extractor=features_extractor,
            action_space=action_space,
            net_arch=actor_arch,
            activation_fn=activation_fn,
            dropout=dropout
        )
        actor = Model.create(
            actor_def,
            inputs=[actor_rngs, init_observation, init_latent],
            tx=optax.radam(learning_rate=lr_schedule)
        )
        actor_target = Model.create(
            actor_def,
            inputs=[actor_rngs, init_observation, init_latent],
            tx=optax.radam(learning_rate=lr_schedule)
        )

        param_key, dropout_key = jax.random.split(param_key, 2)
        critic_rngs = {"params": param_key, "dropout": dropout_key}
        critic_def = Critics(
            features_extractor=features_extractor,
            net_arch=critic_arch,
            activation_fn=activation_fn,
            dropout=dropout,
            n_critics=n_critics
        )
        critic = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_latent, init_action, False],
            tx=optax.radam(learning_rate=lr_schedule)
        )
        critic_target = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_latent, init_action, False],
            tx=optax.radam(learning_rate=lr_schedule)
        )

        self.actor, self.actor_target = actor, actor_target
        self.critic, self.critic_target = critic, critic_target

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

MlpPolicy = TD3Policy
