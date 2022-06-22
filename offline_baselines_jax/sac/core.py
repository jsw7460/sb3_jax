import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    InfoDict,
    Params
)
from offline_baselines_jax.common.jax_layers import polyak_update


def log_ent_coef_update(
    rng:Any,
    log_ent_coef: Model,
    actor: Model,

    observations: jnp.ndarray,

    target_entropy: float,
) -> Tuple[Model, InfoDict]:
    dropout_key, _ = jax.random.split(rng)

    def temperature_loss_fn(ent_params: Params):
        dist = actor(observations, deterministic=False, rngs={"dropout": dropout_key})
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})

        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(
    rng: int,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
):
    ent_coef = jnp.exp(log_ent_coef())
    dropout_key, _ = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn(
            {'params': actor_params},
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        q_values_pi = critic(observations, actions_pi, deterministic=False, rngs={"dropout": dropout_key})
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(
    rng:Any,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma:float
):
    dropout_key, _ = jax.random.split(rng)

    dist = actor(next_observations, deterministic=False, rngs={"dropout": dropout_key})
    next_actions = dist.sample(seed=rng)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(next_observations, next_actions, deterministic=False, rngs={"dropout": dropout_key})
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values= critic.apply_fn(
            {'params': critic_params},
            observations,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean(), "n_qs": n_qs}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info



@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update'))
def sac_update(
    rng: int,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
    target_entropy: float,
    tau: float,
    target_update_cond: bool,
    entropy_update: bool
):

    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = sac_critic_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma
    )

    if target_update_cond:
        new_critic_target = polyak_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, key = jax.random.split(rng, 2)
    new_actor, actor_info = sac_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations
    )

    rng, key = jax.random.split(rng, 2)
    if entropy_update:
        new_temp, ent_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        new_temp, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    new_models = {
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor,
        "log_ent_coef": new_temp
    }
    return rng, new_models, {**critic_info, **actor_info, **ent_info}
