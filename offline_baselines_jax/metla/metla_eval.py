import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np

from offline_baselines_jax.common.policies import Model

# from offline_baselines_jax import METLA


HISTORY_OBSERVATION = []
HISTORY_ACTION = []


class METLASampler(object):
    def __init__(
        self,
        seed: int,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        vae: Model,
        normalizing_factor: float,
        history_len: int = 30,
    ):
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        self.latent_dim = latent_dim
        self.vae: Model = vae
        self.normalizing_factor = normalizing_factor
        self.history_len = history_len
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def __len__(self):
        return len(HISTORY_OBSERVATION)

    def normalize_obs(self, observation: jnp.ndarray):
        return observation.copy() / self.normalizing_factor

    def append(self, observation: jnp.ndarray, action: jnp.ndarray) -> None:
        self.observation_dim = observation.shape[-1]
        self.action_dim = action.shape[-1]

        if observation.ndim == 1:
            observation = observation[jnp.newaxis, ...]
        HISTORY_OBSERVATION.append(observation.copy())
        HISTORY_ACTION.append(action.copy())

    def _get_policy_input(self, observation: jnp.ndarray):
        self.key, dropout_key, noise_key = jax.random.split(self.key, 3)

        if observation.ndim == 3:
            observation = observation.squeeze(1)
        elif observation.ndim == 1:
            observation = observation[jnp.newaxis, ...]

        if len(HISTORY_OBSERVATION) == 0:
            history = jnp.zeros((1, self.observation_dim + self.action_dim))
            history = jnp.repeat(history, repeats=self.history_len, axis=0)
            conditioning_latent, *_ = self.vae(
                history,
                observation,
                deterministic=True,
                rngs={"dropout_key": dropout_key, "noise": noise_key}
            )
            return observation, conditioning_latent, history, None

        else:
            history_observation = np.vstack(HISTORY_OBSERVATION)[-self.history_len:, ...]
            history_action = np.vstack(HISTORY_ACTION)[-self.history_len:, ...]
            cur_hist_len = len(history_observation)
            hist_padding_obs = jnp.zeros((self.history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = jnp.zeros((self.history_len - cur_hist_len, self.action_dim))
            return get_policy_input(
                key=self.key,
                vae=self.vae,
                hist_padding_obs=hist_padding_obs,
                hist_padding_act=hist_padding_act,
                observation=observation,
                history_observation=history_observation,
                history_action=history_action
            )


@jax.jit
def get_policy_input(
        key: jnp.ndarray,
        vae: Model,
        hist_padding_obs: jnp.ndarray,
        hist_padding_act: jnp.ndarray,
        observation: jnp.ndarray,
        history_observation: jnp.ndarray,
        history_action: jnp.ndarray,
):
    key, dropout_key, noise_key = jax.random.split(key, 3)

    history_obs = history_observation
    history_act = history_action

    history_obs = jnp.vstack((hist_padding_obs, history_obs))
    history_act = jnp.vstack((hist_padding_act, history_act))
    history = jnp.hstack((history_obs, history_act))
    conditioning_latent, _ = vae(
        history,
        observation,
        deterministic=True,
        rngs={"dropout_key": dropout_key, "noise": noise_key}
    )
    return observation, conditioning_latent, history


@jax.jit
def get_policy_input_with_last_layer(
        key: jnp.ndarray,
        vae: Model,
        last_layer: Model,
        hist_padding_obs: jnp.ndarray,
        hist_padding_act: jnp.ndarray,
        observation: jnp.ndarray,
        history_observation: jnp.ndarray,
        history_action: jnp.ndarray,
):
    key, dropout_key, noise_key = jax.random.split(key, 3)

    history_obs = history_observation
    history_act = history_action

    history_obs = jnp.vstack((hist_padding_obs, history_obs))
    history_act = jnp.vstack((hist_padding_act, history_act))
    history = jnp.hstack((history_obs, history_act))
    conditioning_latent_, *_ = vae(
        history,
        observation,
        deterministic=True,
        rngs={"dropout_key": dropout_key, "noise": noise_key}
    )

    conditioning_latent = last_layer(
        conditioning_latent_,
        deterministic=False,
        rngs={"dropout_key": dropout_key}
    )
    return observation, conditioning_latent, history


@partial(jax.jit)
def _predict_mle(
    key: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    conditioned_latent: jnp.ndarray
) -> jnp.ndarray:
    rng, dropout_key, action_sample_key = jax.random.split(key, 3)
    if observations.ndim == 3:
        observations = observations.squeeze(1)
    elif observations.ndim == 1:
        observations = observations[jnp.newaxis, ...]
    action_dist = actor(
        observations,
        conditioned_latent,
        deterministic=True,
        rngs={"dropout": dropout_key}
    )
    action = action_dist.sample(seed=rng)

    return action


@partial(jax.jit)
def _predict_mse(
    key: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    conditioned_latent: jnp.ndarray
) -> jnp.ndarray:
    rng, dropout_key, action_sample_key = jax.random.split(key, 3)
    if observations.ndim == 3:
        observations = observations.squeeze(1)
    elif observations.ndim == 1:
        observations = observations[jnp.newaxis, ...]
    action = actor(
        observations,
        conditioned_latent,
        deterministic=True,
        rngs={"dropout": dropout_key}
    )

    return action


def evaluate_metla(
    seed: int,
    env: gym.Env,
    model,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    mse: bool = True,
):
    env.seed(seed)
    if mse:
        predictor = _predict_mse
    else:
        predictor = _predict_mle
    key = jax.random.PRNGKey(seed)
    history_len = model.context_len
    normalizing_factor = model.offline_data_normalizing
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = model.latent_dim
    vae = model.second_ae
    sampler = METLASampler(seed, observation_dim, action_dim, latent_dim, vae, normalizing_factor, history_len)

    episodic_rewards = []
    episodic_lengths = []

    for i in range(n_eval_episodes):
        global HISTORY_OBSERVATION
        global HISTORY_ACTION
        HISTORY_OBSERVATION = []
        HISTORY_ACTION = []
        observation = env.reset()
        observation = sampler.normalize_obs(observation)
        dones = False
        current_rewards = 0
        current_lengths = 0
        while not dones:
            current_lengths += 1
            key, *_ = jax.random.split(key)
            observation, conditioned_latent, *_ = sampler._get_policy_input(observation)

            action = predictor(key, model.actor, observation, conditioned_latent)

            if action.ndim == 0:
                action = action[jnp.newaxis, ...]
            next_observation, rewards, dones, infos = env.step(action)

            sampler.append(observation, action)
            current_rewards += rewards
            observation = sampler.normalize_obs(next_observation)

        episodic_rewards.append(current_rewards)
        episodic_lengths.append(current_lengths)

    return jnp.mean(jnp.array(episodic_rewards)), jnp.mean(jnp.array(episodic_lengths))
