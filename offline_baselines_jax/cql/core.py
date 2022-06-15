import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import InfoDict, ReplayBufferSamples, Params


def log_alpha_update(log_alpha_coef: Model, conservative_loss: float) -> Tuple[Model, InfoDict]:
    def alpha_loss_fn(alpha_params: Params):
        alpha_coef = jnp.exp(log_alpha_coef.apply_fn({'params': alpha_params}))
        alpha_coef_loss = -alpha_coef * conservative_loss

        return alpha_coef_loss, {'alpha_coef': alpha_coef, 'alpha_coef_loss': alpha_coef_loss}

    new_alpha_coef, info = log_alpha_coef.apply_gradient(alpha_loss_fn)
    new_alpha_coef = param_clip(new_alpha_coef, 1e+6)
    return new_alpha_coef, info

def log_ent_coef_update(key:Any, log_ent_coef: Model, actor:Model , target_entropy: float, replay_data:ReplayBufferSamples) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(ent_params: Params):
        dist = actor(replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(key: int, actor: Model, critic:Model, log_ent_coef: Model, replay_data:ReplayBufferSamples):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(replay_data.observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=0)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(key:Any, actor: Model, critic: Model, critic_target: Model, log_ent_coef: Model,
                                log_alpha_coef: Model, replay_data: ReplayBufferSamples, gamma:float, conservative_weight:float,
                                lagrange_thresh:float,):
    next_dist = actor(replay_data.next_observations)
    next_actions = next_dist.sample(seed=key)
    next_log_prob = next_dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(replay_data.next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=0)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

    batch_size, action_dim = replay_data.actions.shape
    alpha_coef = jnp.exp(log_alpha_coef())
    ###############################
    ## For CQL Conservative Loss ##
    ###############################

    cql_dist = actor(replay_data.observations)
    cql_actions = cql_dist.sample(seed=key)
    cql_log_prob = cql_dist.log_prob(cql_actions)

    repeated_observations = jnp.repeat(replay_data.observations, repeats=10, axis=0)
    key, subkey = jax.random.split(key, 2)
    random_actions = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(batch_size * 10, action_dim))

    random_density = jnp.log(0.5 ** action_dim)
    alpha_coef = jnp.clip(jnp.exp(log_alpha_coef()), 0, 1e+6)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q = critic.apply_fn({'params': critic_params}, replay_data.observations, replay_data.actions)
        cql_q = critic.apply_fn({'params': critic_params}, replay_data.observations, cql_actions)
        random_q = critic.apply_fn({'params': critic_params}, repeated_observations, random_actions)
        conservative_loss = 0
        for idx in range(len(cql_q)):
            conservative_loss += jax.scipy.special.logsumexp(jnp.ndarray([jnp.repeat(cql_q[idx], repeats=10, axis=0) - cql_log_prob, random_q[idx] - random_density])).mean() - current_q[idx].mean()
        conservative_loss = (conservative_weight * ((conservative_loss) / len(cql_q)) - lagrange_thresh)
        # Compute critic loss
        critic_loss = 0
        for q in current_q:
            critic_loss = critic_loss + jnp.mean(jnp.square(q - target_q_values))
        critic_loss = critic_loss / len(current_q) + alpha_coef * conservative_loss

        return critic_loss, {'critic_loss': critic_loss, 'current_q': current_q.mean(), 'conservative_loss': conservative_loss}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info

def param_clip(log_alpha_coef: Model, a_max: float) -> Model:
    new_log_alpha_params = jax.tree_multimap(lambda p: jnp.clip(p, a_max=jnp.log(a_max)), log_alpha_coef.params)
    return log_alpha_coef.replace(params=new_log_alpha_params)

def target_update(critic: Model, critic_target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic_target.params)
    return critic_target.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update',
                                             'alpha_update', 'conservative_weight', 'lagrange_thresh'))
def update_cql(
    rng: int, actor: Model, critic: Model, critic_target: Model, log_ent_coef: Model, log_alpha_coef: Model, replay_data: ReplayBufferSamples,
        gamma: float, target_entropy: float, tau: float, target_update_cond: bool, entropy_update: bool, alpha_update: bool,
        conservative_weight:float, lagrange_thresh:float,
) -> Tuple[int, Model, Model, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = sac_critic_update(key, actor, critic, critic_target, log_ent_coef, log_alpha_coef, replay_data,
                                                gamma, conservative_weight, lagrange_thresh)
    if target_update_cond:
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, key = jax.random.split(rng)
    new_actor, actor_info = sac_actor_update(key, actor, new_critic, log_ent_coef, replay_data)
    rng, key = jax.random.split(rng)
    if entropy_update:
        new_temp, ent_info = log_ent_coef_update(key, log_ent_coef, new_actor, target_entropy, replay_data)
    else:
        new_temp, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    if alpha_update:
        new_alpha, alpha_info = log_alpha_update(log_alpha_coef, critic_info['conservative_loss'])
    else:
        new_alpha, alpha_info = log_alpha_coef, {'alpha_coef': jnp.exp(log_alpha_coef()), 'alpha_coef_loss': 0}

    return rng, new_actor, new_critic, new_critic_target, new_temp, new_alpha, {**critic_info, **actor_info, **ent_info, **alpha_info}
