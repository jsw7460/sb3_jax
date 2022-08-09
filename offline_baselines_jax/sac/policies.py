import functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

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
from offline_baselines_jax.common.utils import get_basic_rngs

tfd = tfp.distributions
tfb = tfp.bijectors
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "deterministic"))
def sample_actions(
    rng: jnp.ndarray,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: Union[np.ndarray, Dict],
    deterministic: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rng, dropout_key = jax.random.split(rng)
    rngs = {"dropout": dropout_key}
    dist = actor_apply_fn({'params': actor_params}, observations, deterministic=deterministic, rngs=rngs)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=rng)


class Actor(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.0

    latent_pi = None
    mu = None
    log_std = None

    def setup(self):
        self.latent_pi = create_mlp(-1, self.net_arch, self.activation_fn, self.dropout)
        action_dim = get_action_dim(self.action_space)
        self.mu = create_mlp(action_dim, self.net_arch, self.activation_fn, self.dropout)
        self.log_std = create_mlp(action_dim, self.net_arch, self.activation_fn, self.dropout)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False, **kwargs) -> jnp.ndarray:
        mean_actions, log_stds = self.get_action_dist_params(observations, deterministic=deterministic, **kwargs)
        return self.actions_from_params(mean_actions, log_stds)

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations, **kwargs)

        latent_pi = self.latent_pi(features, deterministic=deterministic)
        mean_actions = self.mu(latent_pi, deterministic=deterministic)
        log_stds = self.log_std(latent_pi, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_stds

    def actions_from_params(self, mean: jnp.ndarray, log_std: jnp.ndarray):
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist


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
            dropout=self.dropout,
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
            variable_axes={"params": 1, "batch_stats": 1},
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



class SACPolicy(object):
    def __init__(
        self,
        rng,
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

        self.rng, actor_key, critic_key, features_key, dropout_key = jax.random.split(rng, 5)

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        features_extractor_def = features_extractor_class(
            _observation_space=observation_space,
            **features_extractor_kwargs
        )
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
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
            n_critics=n_critics,
            dropout=dropout
        )

        # Init dummy inputs
        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = observation[key][np.newaxis, ...]
        else:
            observation = observation_space.sample()[np.newaxis, ...]
        action = action_space.sample()[np.newaxis, ...]

        self.rng, actor_rngs = get_basic_rngs(self.rng)
        actor = Model.create(actor_def, inputs=[actor_rngs, observation], tx=optax.adam(learning_rate=lr_schedule))

        self.rng, critic_rngs = get_basic_rngs(self.rng)
        critic = Model.create(
            critic_def,
            inputs=[critic_rngs, observation, action],
            tx=optax.adam(learning_rate=lr_schedule)
        )
        critic_target = Model.create(critic_def, inputs=[critic_rngs, observation, action])
        self.actor = actor
        self.critic, self.critic_target = critic, critic_target

    def _predict(self, observation: jnp.ndarray, deterministic: bool) -> np.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observation, deterministic)

        self.rng = rng
        return np.asarray(actions)

    def predict(self, observation: jnp.ndarray, deterministic: bool = True, **kwargs) -> np.ndarray:
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


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
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
        rng,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(CnnPolicy, self).__init__(
            rng,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            n_critics,
            dropout
        )


class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
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
        rng,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(MultiInputPolicy, self).__init__(
            rng,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            n_critics,
            dropout
        )