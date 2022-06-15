import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    InfoDict,
    ReplayBufferSamples,
    Params
)


def log_ent_coef_update(
    key:Any,
    log_ent_coef: Model,
    actor: Model,
    target_entropy: float,
    replay_data:ReplayBufferSamples
) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(ent_params: Params):
        dist = actor(replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(key: int, actor: Model, critic: Model, log_ent_coef: Model, replay_data: ReplayBufferSamples):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(replay_data.observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(
    key:Any,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    replay_data: ReplayBufferSamples,
    gamma:float
):
    dist = actor(replay_data.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(replay_data.next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values= critic.apply_fn({'params': critic_params}, replay_data.observations, replay_data.actions)

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def target_update(critic: Model, critic_target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic_target.params)
    return critic_target.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update'))
def sac_update(
    rng: int,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    replay_data: ReplayBufferSamples,
    gamma: float,
    target_entropy: float,
    tau: float,
    target_update_cond: bool,
    entropy_update: bool
):

    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = sac_critic_update(key, actor, critic, critic_target, log_ent_coef, replay_data, gamma)
    if target_update_cond:
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, key = jax.random.split(rng, 2)
    new_actor, actor_info = sac_actor_update(key, actor, new_critic, log_ent_coef, replay_data)

    rng, key = jax.random.split(rng, 2)
    if entropy_update:
        new_temp, ent_info = log_ent_coef_update(key, log_ent_coef, new_actor, target_entropy, replay_data)
    else:
        new_temp, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    new_models = {
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor,
        "log_ent_coef": new_temp
    }
    return rng, new_models, {**critic_info, **actor_info, **ent_info}
