import functools
from typing import Any, Dict, List, Optional, Type, Union, Callable, Tuple

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import get_action_dim, preprocess_obs
from offline_baselines_jax.common.type_aliases import Schedule, Params


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "deterministic"))
def sample_actions(
    rng: jnp.ndarray,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: [np.ndarray, Dict],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    rng, dropout_key = jax.random.split(rng)
    rngs = {"dropout": dropout_key}
    action = actor_apply_fn({'params': actor_params}, observations, deterministic=deterministic, rngs=rngs)
    return rng, action


class Actor(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.0

    mu = None

    def setup(self):
        action_dim = get_action_dim(self.action_space)
        self.mu = create_mlp(action_dim, self.net_arch, self.activation_fn, self.dropout, squash_output=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False):
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        mu = self.mu(features, deterministic=deterministic)
        return mu


class SingleCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.relu

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False
    ):
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        q_input = jnp.concatenate((features, actions), axis=1)
        return self.q_net(q_input, deterministic=deterministic)


class Critic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float = 0.0
    activation_fn: Type[nn.Module] = nn.relu
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
        self.q_networks = batch_qs(
            self.features_extractor,
            self.observation_space,
            self.net_arch,
            self.dropout,
            self.activation_fn
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, actions: jnp.ndarray, deterministic: bool = False):
        return self.q_networks(observations, actions, deterministic)


class TD3Policy(object):
    def __init__(
        self,
        key,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.rng, actor_key, critic_key, features_key, dropout_key = jax.random.split(key, 5)

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        features_extractor_def = features_extractor_class(
            _observation_space=observation_space,
            **features_extractor_kwargs
        )

        actor_def = Actor(
            features_extractor=features_extractor_def,
            observation_space=observation_space,
            action_space=action_space,
            net_arch=actor_arch,
            activation_fn=activation_fn,
            dropout=dropout
        )

        critic_def = Critic(
            features_extractor=features_extractor_def,
            observation_space=observation_space,
            net_arch=critic_arch,
            activation_fn=activation_fn,
            n_critics=n_critics
        )

        # Init dummy inputs
        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = observation_space[key][np.newaxis, ...]
        else:
            observation = observation_space.sample()[np.newaxis, ...]

        actor_rngs = {"params": actor_key, "dropout": dropout_key}
        actor = Model.create(
            actor_def,
            inputs=[actor_rngs, observation],
            tx=optax.adam(learning_rate=lr_schedule)
        )
        actor_target = Model.create(
            actor_def,
            inputs=[actor_rngs, observation],
            tx=optax.adam(learning_rate=lr_schedule)
        )

        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = np.expand_dims(observation[key], axis=0)
        else:
            observation = np.expand_dims(observation_space.sample(), axis=0)
        action = np.expand_dims(action_space.sample(), axis=0)

        critic_rngs = {"params": critic_key, "dropout": dropout_key}
        critic = Model.create(
            critic_def,
            inputs=[critic_rngs, observation, action],
            tx=optax.adam(learning_rate=lr_schedule)
        )
        critic_target = Model.create(critic_def, inputs=[critic_key, observation, action])

        self.actor, self.actor_target = actor, actor_target
        self.critic, self.critic_target = critic, critic_target

    def _predict(self, observation: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observation, deterministic)
        self.rng = rng
        return np.asarray(actions)

    def predict(self, observation: jnp.ndarray, deterministic: bool = True) -> np.ndarray:
        actions = self._predict(observation, deterministic)
        if isinstance(self.action_space, gym.spaces.Box):
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions, None

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


class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        key,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super(CnnPolicy, self).__init__(
            key,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            n_critics,
        )


class MultiInputPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        key,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super(MultiInputPolicy, self).__init__(
            key,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            n_critics,
        )