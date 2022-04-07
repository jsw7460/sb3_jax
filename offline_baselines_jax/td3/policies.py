from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax
import functools
import optax

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import get_action_dim
from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
    default_init,
)
from offline_baselines_jax.common.type_aliases import Schedule, Params


@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def sample_actions(actor_apply_fn: Callable[..., Any], actor_params: Params,
                    observations: np.ndarray,) -> jnp.ndarray:
    action = actor_apply_fn({'params': actor_params}, observations)
    return action

class Actor(nn.Module):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    features_extractor: nn.Module
    action_space: gym.spaces.Space
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        # Save arguments to re-create object at loading
        features = self.features_extractor(observations)
        action_dim = get_action_dim(self.action_space)
        latent_pi = create_mlp(-1, self.net_arch, self.activation_fn, squash_output=True)(features)

        mu = nn.Dense(action_dim, kernel_init=default_init())(latent_pi)
        return mu

class Critic(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        features = self.features_extractor(observations)
        inputs = jnp.concatenate([features, actions], -1)
        q_value = create_mlp(1, self.net_arch, self.activation_fn)(inputs)
        return q_value

class DoubleCritic(nn.Module):
    features_extractor: nn.Module
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    n_critics: int = 2

    @nn.compact
    def __call__(self, states, actions):
        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.n_critics)
        qs = VmapCritic(self.features_extractor, self.net_arch, self.activation_fn)(states, actions)
        return qs


class TD3Policy(object):
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
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.rng, actor_key, critic_key, features_key = jax.random.split(key, 4)

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        features_extractor_def = features_extractor_class(_observation_space=observation_space, **features_extractor_kwargs)
        actor_def = Actor(features_extractor=features_extractor_def, action_space=action_space,
                          net_arch=actor_arch, activation_fn=activation_fn)
        critic_def = DoubleCritic(features_extractor=features_extractor_def, net_arch=critic_arch,
                                  activation_fn=activation_fn, n_critics=n_critics)

        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = np.expand_dims(observation[key], axis=0)
        else:
            observation = np.expand_dims(observation_space.sample(), axis=0)

        actor = Model.create(actor_def, inputs=[actor_key, observation], tx=optax.adam(learning_rate=lr_schedule))
        actor_target = Model.create(actor_def, inputs=[actor_key, observation], tx=optax.adam(learning_rate=lr_schedule))

        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = np.expand_dims(observation[key], axis=0)
        else:
            observation = np.expand_dims(observation_space.sample(), axis=0)
        action = np.expand_dims(action_space.sample(), axis=0)

        critic = Model.create(critic_def, inputs=[critic_key, observation, action],
                              tx=optax.adam(learning_rate=lr_schedule))
        critic_target = Model.create(critic_def, inputs=[critic_key, observation, action])


        self.actor, self.actor_target = actor, actor_target
        self.critic, self.critic_target = critic, critic_target

    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return np.asarray(sample_actions(self.actor.apply_fn, self.actor.params, observation))

    def predict(self, observation: jnp.ndarray, deterministic: bool = False) -> np.ndarray:
        actions = self._predict(observation)
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