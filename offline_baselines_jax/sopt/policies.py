from functools import partial
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
    create_mlp,
    get_actor_critic_arch,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Schedule, Params
from offline_baselines_jax.common.preprocessing import preprocess_obs

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 2
LOG_STD_MIN = -20


@partial(jax.jit, static_argnames=("actor_apply_fn", ))
def sample_actions(
    rng: jnp.ndarray,
    actor_apply_fn: Callable,
    actor_params: Params,
    observations: jnp.ndarray,
    subgoals: jnp.ndarray,
    goals: jnp.ndarray,
    deterministic: bool = False
):
    dist = actor_apply_fn({"params": actor_params}, observations, subgoals, goals, deterministic)
    return dist.sample(seed=rng)


class CNNExtractor(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    feature_dim: int = 512

    @nn.compact
    def __call__(self, observations: jnp.array) -> jnp.array:
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(observations)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # flatten

        x = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(x)
        return x


class PosGoalImgSubgoalActor(nn.Module):            # NOTE: 현재 Policy는 state와 latent의 concat을 input으로 받도록 할 것이다.
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
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

    def forward(
        self,
        observations: jnp.ndarray,
        subgoal: jnp.ndarray,
        goal: jnp.ndarray,
        deterministic: bool = False):
        mean_actions, log_stds = self.get_action_dist_params(
            observations=observations,
            subgoal=subgoal,
            goal=goal,
            deterministic=deterministic
        )
        return self.actions_from_params(mean_actions, log_stds)

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        subgoal: jnp.ndarray,
        goal: jnp.ndarray,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        observations = preprocess_obs(observations, self.observation_space, normalize_images=True)
        x = jnp.concatenate((observations, subgoal), axis=-1)       # Subgoal: Image
        x = self.features_extractor(x)
        x = jnp.hstack((x, goal))                                     # Concatenate with goal
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

    def _predict(
        self,
        observation: jnp.ndarray,
        subgoal: jnp.ndarray,
        goal: jnp.ndarray,
        deterministic: bool
    ) -> np.ndarray:

        rng, dropout_key, sample_key = jax.random.split(self.rng, 3)
        action_dist = self.forward(
            observations=observation,
            subgoal=subgoal,
            goal=goal,
            deterministic=deterministic)

        action = action_dist.sample(seed=sample_key)
        return action


class SinglePosGoalImgSubgoalCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.Space
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
        subgoal: jnp.ndarray,
        goal: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False
    ) -> jnp.ndarray:

        observations = preprocess_obs(observations, self.observation_space)
        x = jnp.concatenate((observations, subgoal), axis=-1)
        x = self.features_extractor(x)
        q_net_input = jnp.hstack((x, goal, actions))
        return self.q_net(q_net_input, deterministic=deterministic)


class PosGoalImgSubgoalCritics(nn.Module):           # n_critics mode.
    features_extractor: nn.Module
    observation_space: gym.Space
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            SinglePosGoalImgSubgoalCritic,
            in_axes=None,
            out_axes=1,
            variable_axes={"params": 1},
            split_rngs={"params": True},
            axis_size=self.n_critics,
        )
        self.q_networks = batch_qs(self.features_extractor, self.observation_space, self.net_arch, self.activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        subgoal: jnp.ndarray,
        goal: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False
    ) -> jnp.ndarray:
        return self.q_networks(
            observations,
            subgoal,
            goal,
            actions,
            deterministic
        )


class PosGoalImgSubgoalPolicy(object):      # SAC Style
    def __init__(
        self,
        key: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        latent_dim: int,                    # Dimension of latent embedding of policy
        goal_dim: int,                      # XY - pos
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = CNNExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        n_critics: int = 2,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = key

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
        init_subgoal = observation_space.sample()[jnp.newaxis, ...]
        init_goal = jnp.zeros((1, goal_dim))
        init_action = self.action_space.sample()[jnp.newaxis, ...]

        param_key, dropout_key, action_key = jax.random.split(self.rng, 3)
        actor_rngs = {"params": param_key, "dropout": dropout_key, "action_sample": action_key}
        actor_def = PosGoalImgSubgoalActor(
            features_extractor=features_extractor,
            observation_space=observation_space,
            action_space=action_space,
            latent_dim=latent_dim,
            net_arch=actor_arch,
            activation_fn=activation_fn,
            dropout=dropout,
        )
        actor = Model.create(
            actor_def,
            inputs=[actor_rngs, init_observation, init_subgoal, init_goal],
            tx=optax.adam(learning_rate=lr_schedule)
        )

        critic_rngs = {"params": param_key, "dropout": dropout_key, "action_sample": action_key}
        critic_def = PosGoalImgSubgoalCritics(
            features_extractor=features_extractor,
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout=dropout,
            n_critics=n_critics
        )
        critic = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_subgoal, init_goal, init_action, False],
            tx=optax.adam(learning_rate=lr_schedule)
        )
        critic_target = Model.create(
            critic_def,
            inputs=[critic_rngs, init_observation, init_subgoal, init_goal, init_action, False],
            tx=optax.adam(learning_rate=lr_schedule)
        )

        self.n_critics = n_critics
        self.latent_dim = latent_dim
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target

    def _predict(self, observation: jnp.ndarray, conditioned_latent: jnp.ndarray, deterministic: bool) -> np.ndarray:
        raise NotImplementedError()

    def predict(
        self,
        observation: jnp.ndarray,
        subgoals: jnp.ndarray,
        goals: jnp.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        self.rng, _ = jax.random.split(self.rng)
        actions = sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observation,
            subgoals,
            goals,
            deterministic
        )
        if isinstance(self.action_space, gym.spaces.Box):
            actions = jnp.clip(actions, self.action_space.low, self.action_space.high)

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
