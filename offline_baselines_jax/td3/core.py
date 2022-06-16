import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.jax_layers import polyak_update
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import InfoDict, Params

STATIC_ARGNAMES = (
    "gamma",
    "tau",
    "target_policy_noise",
    "target_noise_clip",
    "alpha",
    "without_exploration",
    "actor_update_cond"
)

def td3_critic_update(
    rng:Any,
    critic: Model,
    critic_target: Model,
    actor_target: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma:float,
    target_policy_noise: float,
    target_noise_clip: float
):

    dropout_key, _ = jax.random.split(rng)

    # Select action according to policy and add clipped noise
    noise = jax.random.normal(rng) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = actor_target(next_observations, rngs={"dropout": dropout_key})
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    # Compute the next Q-values: min over all critics targets
    next_q_values = critic_target(next_observations, next_actions, deterministic=False, rngs={"dropout": dropout_key})
    next_q_values = jnp.min(next_q_values, axis=1)
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        q_values = critic.apply_fn(
            {"params": critic_params},
            observations, actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs
        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def td3_actor_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,

    alpha: float,
    without_exploration: bool
):
    dropout_key, _ = jax.random.split(rng)

    if without_exploration:
        _actions_pi = actor(observations, deterministic=False, rngs={"dropout": dropout_key})
        q1 = critic(observations, _actions_pi, deterministic=False, rngs={"dropout": dropout_key})[0]
        coef_lambda = alpha / (jnp.mean(jnp.abs(q1)))
    else:
        coef_lambda = 1

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Compute actor loss
        actions_pi = actor.apply_fn(
            {"params": actor_params},
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        q_value = critic(
            observations,
            actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[0].mean()

        actor_loss = - q_value

        if without_exploration:
            bc_loss = jnp.mean(jnp.square(actions_pi - actions))
            actor_loss = coef_lambda * actor_loss + bc_loss

        return actor_loss, {'actor_loss': actor_loss, 'q_value': q_value, 'coef_lambda': coef_lambda}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


@functools.partial(jax.jit, static_argnames=tuple(STATIC_ARGNAMES))
def update_td3(
    rng: int,
    actor: Model,
    critic: Model,
    actor_target: Model,
    critic_target: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    actor_update_cond: bool,
    tau: float,
    target_policy_noise: float,
    target_noise_clip: float,
    gamma: float,
    alpha: float,
    without_exploration: bool
):

    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = td3_critic_update(
        rng=rng,
        critic=critic,
        critic_target=critic_target,
        actor_target=actor_target,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip
    )

    if actor_update_cond:
        new_actor, actor_info = td3_actor_update(
            rng=rng,
            actor=actor,
            critic=critic,
            observations=observations,
            actions=actions,
            alpha=alpha,
            without_exploration=without_exploration
        )
        new_actor_target = polyak_update(new_actor, actor_target, tau)
        new_critic_target = polyak_update(new_critic, critic_target, tau)
    else:
        new_actor, actor_info = actor, {'actor_loss': 0, 'q_value': 0, 'coef_lambda': 1}
        new_actor_target = actor_target
        new_critic_target = critic_target

    new_models = {
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor,
        "actor_target": new_actor_target
    }

    return rng, new_models, {**critic_info, **actor_info, "actor_update_cond": actor_update_cond}