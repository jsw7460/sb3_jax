import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax
import functools
import optax

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

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
from offline_baselines_jax.common.jax_layers import SoftModule
from offline_baselines_jax.sac.policies import SACPolicy
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def sample_actions(rng: int, actor_apply_fn: Callable[..., Any], actor_params: Params,
                    observations: np.ndarray, module_select:List[List[int]]) -> Tuple[int, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations, **{'module_select': module_select})
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)

class SoftModulePolicy(SACPolicy):
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
        key,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = [64],
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = SoftModule,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super(SoftModulePolicy, self).__init__(
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

    def predict(self, observation: jnp.ndarray, deterministic: bool = False, module_select: List[List[int]]=None, ) -> np.ndarray:
        actions = self._predict(observation, module_select)
        if isinstance(self.action_space, gym.spaces.Box):
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions, None

    def _predict(self, observation: jnp.ndarray, module_select: List[List[int]]=None,) -> np.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observation, module_select)
        self.rng = rng
        actions = np.asarray(actions)
        return actions
