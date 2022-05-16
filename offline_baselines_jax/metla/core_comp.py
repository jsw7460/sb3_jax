from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import InfoDict, Params, ReplayBufferSamples
from .networks import WassersteinAutoEncoder
from gym.spaces import Box

EPS = 1e-5
USE_GOAL_REACHING_LOSS = 1.0
GAE_LE_COEFS = 1.0
RAND_QVAL_COEF = 1.0


def update_td3_higher_actor(
    gamma: float,
    rng: jnp.ndarray,
    higher_actor: Model,             # View it as an actor
    second_ae: Model,
    higher_critic: Model,
    higher_critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    higher_actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)
    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    last_dropout_key, actor_dropout_key = jax.random.split(dropout_key, 2)
    history_for_next_timestep = history_input[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timestep, current), axis=1)

    def finetune_layer_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        higher_actions_pi, _ = higher_actor.apply_fn(
            {"params": params},
            history_input,
            observations,
            deterministic=False,
            rngs={"dropout": last_dropout_key, "noise": noise_key}
        )

        prior_loss = jnp.mean((higher_actions_pi - higher_actions) ** 2) * 0

        next_higher_action, _ = higher_actor.apply_fn(
            {"params": params},
            history_for_next_timestep,
            next_observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )

        next_q_values = higher_critic_target(
            next_observations,
            next_higher_action,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        next_q_values = jnp.min(next_q_values, axis=1)
        target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

        q_values = higher_critic(
            observations,
            higher_actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        n_qs = q_values.shape[1]
        critic_loss = sum([(target_q_values - q_values[:, i, ...].squeeze()) ** 2 for i in range(n_qs)])
        critic_loss = jnp.mean(critic_loss)

        critic_loss = (critic_loss / len(q_values))

        actor_loss = -higher_critic(
            observations,
            higher_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        actor_loss = jnp.mean(actor_loss)

        finetune_layer_loss = critic_loss + actor_loss + prior_loss

        return finetune_layer_loss, {"finetune_layer_loss": finetune_layer_loss, "prior_loss": prior_loss}

    new_finetune_layer, finetune_info = higher_actor.apply_gradient(finetune_layer_loss_fn)
    return new_finetune_layer, finetune_info


def update_td3_finetune_layer(
    gamma: float,
    rng: jnp.ndarray,
    finetune_layer: Model,             # View it as an actor
    second_ae: Model,
    higher_critic: Model,
    higher_critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)
    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    current_first_pred_goal, _ = second_ae(
        history_input,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    last_dropout_key, actor_dropout_key = jax.random.split(dropout_key, 2)

    history_for_next_timestep = history_input[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timestep, current), axis=1)

    next_first_pred_goal, _ = second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    def finetune_layer_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        current_conditioned_latent = finetune_layer.apply_fn(
            {"params": params},
            current_first_pred_goal,
            deterministic=False,
            rngs={"dropout": last_dropout_key, "noise": noise_key}
        )

        prior_loss = jnp.mean((current_conditioned_latent - current_first_pred_goal) ** 2) * 100

        next_conditioned_latent = finetune_layer.apply_fn(
            {"params": params},
            next_first_pred_goal,
            deterministic=False,
            rngs={"dropout": last_dropout_key, "noise": noise_key}
        )

        next_q_values = higher_critic_target(
            next_observations,
            next_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        next_q_values = jnp.min(next_q_values, axis=1)
        target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

        q_values = higher_critic(
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        n_qs = q_values.shape[1]

        critic_loss = sum([(target_q_values - q_values[:, i, ...].squeeze()) ** 2 for i in range(n_qs)])
        critic_loss = jnp.mean(critic_loss)

        critic_loss = (critic_loss / len(q_values))

        actor_loss = -higher_critic(
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        actor_loss = jnp.mean(actor_loss)

        finetune_layer_loss = critic_loss + actor_loss + prior_loss * 100

        return finetune_layer_loss, {"finetune_layer_loss": finetune_layer_loss, "prior_loss": prior_loss}

    new_finetune_layer, finetune_info = finetune_layer.apply_gradient(finetune_layer_loss_fn)
    return new_finetune_layer, finetune_info


def update_td3_higher_actor_critic_higher_actor(
    gamma: float,
    rng: jnp.ndarray,
    higher_actor: Model,             # View it as an actor
    higher_critic: Model,
    higher_critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    higher_actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)
    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    history_for_next_timestep = history_input[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timestep, current), axis=1)

    next_higher_actions, _ = higher_actor(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    next_q_values = higher_critic_target(
        next_observations,
        next_higher_actions,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_q_values = jnp.min(next_q_values, axis=1)
    target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

    def higher_critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_values = higher_critic.apply_fn(
            {"params": params},
            observations,
            higher_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        n_qs = q_values.shape[1]
        higher_critic_loss \
            = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs
        higher_critic_loss = (higher_critic_loss / len(q_values))

        return higher_critic_loss, {"critic_loss": higher_critic_loss}

    new_higher_critic, higher_critic_info = higher_critic.apply_gradient(higher_critic_loss_fn)

    def higher_actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        higher_actions_pi, _ = higher_actor.apply_fn(
            {"params": params},
            history_input,
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )
        higher_actor_loss = -new_higher_critic(
            observations,
            higher_actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        higher_actor_loss = jnp.mean(higher_actor_loss)

        return higher_actor_loss, {"actor_loss": higher_actor_loss}

    new_finetune_layer, finetune_layer_actor_info = higher_actor.apply_gradient(higher_actor_loss_fn)

    info = {**higher_critic_info, **finetune_layer_actor_info}
    return new_finetune_layer, new_higher_critic, info


def update_td3_higher_actor_critic_last_layer(
    gamma: float,
    rng: jnp.ndarray,
    finetune_layer: Model,             # View it as an actor
    second_ae: Model,
    higher_critic: Model,
    higher_critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)
    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    current_first_pred_goal, _ = second_ae(
        history_input,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    history_for_next_timestep = history_input[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timestep, current), axis=1)

    next_first_pred_goal, _ = second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    current_conditioned_latent = finetune_layer(current_first_pred_goal, deterministic=False)
    next_conditioned_latent = finetune_layer(next_first_pred_goal, deterministic=False)

    next_q_values = higher_critic_target(
        next_observations,
        next_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_q_values = jnp.min(next_q_values, axis=1)
    target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

    def higher_critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_values = higher_critic.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        n_qs = q_values.shape[1]
        higher_critic_loss \
            = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs
        higher_critic_loss = (higher_critic_loss / len(q_values))

        return higher_critic_loss, {"critic_loss": higher_critic_loss}
    new_higher_critic, higher_critic_info = higher_critic.apply_gradient(higher_critic_loss_fn)

    def higher_actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        current_conditioned_latent_grad = finetune_layer.apply_fn(
            {"params": params},
            current_first_pred_goal,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        higher_actor_loss = -higher_critic(
            observations,
            current_conditioned_latent_grad,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        higher_actor_loss = jnp.mean(higher_actor_loss)

        return higher_actor_loss, {"actor_loss": higher_actor_loss}

    new_finetune_layer, finetune_layer_actor_info = finetune_layer.apply_gradient(higher_actor_loss_fn)

    info = {**higher_critic_info, **finetune_layer_actor_info}
    return new_finetune_layer, new_higher_critic, info


def update_sac_last_layer(
    gamma: float,
    key: jnp.ndarray,
    log_ent_coef: Model,
    finetune_last_layer: Model,
    second_ae: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
):
    rng, dropout_key, noise_key = jax.random.split(key, 3)
    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    current_first_pred_goal, _ = second_ae(
        history_input,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    last_dropout_key, actor_dropout_key = jax.random.split(dropout_key, 2)

    history_for_next_timestep = history_input[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timestep, current), axis=1)

    next_first_pred_goal, _ = second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    def last_layer_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        current_conditioned_latent = finetune_last_layer.apply_fn(
            {"params": params},
            current_first_pred_goal,
            deterministic=False,
            rngs={"dropout": last_dropout_key}
        )

        prior_loss = jnp.mean((current_conditioned_latent - current_first_pred_goal) ** 2)

        next_conditioned_latent = finetune_last_layer.apply_fn(
            {"params": params},
            next_first_pred_goal,
            deterministic=False,
            rngs={"dropout": last_dropout_key}
        )

        action_dist = actor(
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": actor_dropout_key}
        )
        actions_pi = action_dist.sample(seed=key)
        log_prob = action_dist.log_prob(actions_pi)

        next_action_dist = actor(
            next_observations,
            next_conditioned_latent,
            deterministic=False,
            rngs={"dropout": actor_dropout_key}
        )
        next_action = next_action_dist.sample(seed=key)
        next_log_prob = next_action_dist.log_prob(next_action)

        next_q_values = critic_target(
            next_observations,
            next_conditioned_latent,
            next_action,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        next_q_values = jnp.min(next_q_values, axis=1)
        ent_coef = jnp.exp(log_ent_coef())
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

        target_q_values = jnp.squeeze(rewards) + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

        q_values = critic(
            observations,
            current_conditioned_latent,
            actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        n_qs = q_values.shape[1]

        critic_loss = sum([(target_q_values - q_values[:, i, :].squeeze()) ** 2 for i in range(n_qs)])
        critic_loss = jnp.mean(critic_loss)

        critic_loss = (critic_loss / len(q_values))

        min_qf_pi = jnp.min(q_values, axis=1)
        actor_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)

        last_layer_loss = critic_loss + actor_loss + (prior_loss * 100)
        return last_layer_loss, {
                "last_layer_loss": last_layer_loss,
                "next_q_values": next_q_values,
                "prior_loss": prior_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            }

    new_finetune_last_layer, last_layer_info = finetune_last_layer.apply_gradient(last_layer_loss_fn)

    return new_finetune_last_layer, last_layer_info


def log_ent_coef_update(
        key: Any,
        log_ent_coef: Model,
        actor: Model,
        target_entropy: float,
        replay_data: ReplayBufferSamples
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
        min_qf_pi = jnp.min(q_values_pi, axis=0)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(
        key: Any,
        actor: Model,
        critic: Model,
        critic_target: Model,
        log_ent_coef: Model,
        replay_data: ReplayBufferSamples,
        gamma: float
):
    dist = actor(replay_data.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(replay_data.next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=0)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q = critic.apply_fn({'params': critic_params}, replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0
        for q in current_q:
            critic_loss = critic_loss + jnp.mean(jnp.square(q - target_q_values))
        critic_loss = critic_loss / len(current_q)
        return critic_loss, {'critic_loss': critic_loss, 'current_q': current_q.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_target(critic: Model, critic_target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic_target.params)
    return critic_target.replace(params=new_target_params)


def update_entropy(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    target_entropy: float,
    observations: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray
) -> Tuple[Model, InfoDict]:
    dropout_key, sample_key = jax.random.split(rng, 2)

    def temperature_loss_fn(ent_params: Params):
        dist = actor(observations, current_conditioned_latent, deterministic=False, rngs={"dropout": dropout_key})
        actions_pi = dist.sample(seed=sample_key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def update_gae(
    rng: jnp.ndarray,
    ae: Model,
    history_observations: jnp.ndarray,
    observations: jnp.ndarray,
    future_observations: jnp.ndarray,
    n_nbd: int = 5,
    **kwargs,
) -> Tuple[Model, Dict]:            # Observation으로 현재 timestep observation의 neighborhood를 reconstruct

    batch_size = observations.shape[0]
    future_len = future_observations.shape[1]

    assert future_len > n_nbd
    rng, dropout_key = jax.random.split(rng)

    # Coefficients: [batch_size, 2 * n_nbd + 1]
    _observations = observations[:, jnp.newaxis, :]                                     # [batch_size, 1, state_dim]
    dist_to_history = jnp.sum((_observations - history_observations) ** 2, axis=2)      # [batch_size, history_len]
    dist_to_future = jnp.sum((_observations - future_observations) ** 2, axis=2)        # [batch_size, future_len]

    dist_to_history = dist_to_history[:, -n_nbd:]
    dist_to_future = dist_to_future[:, :n_nbd]

    history_coefs = jnp.exp(-dist_to_history / GAE_LE_COEFS)
    current_coefs = jnp.ones((batch_size, 1))
    future_coefs = jnp.exp(-dist_to_future / GAE_LE_COEFS)

    coefs = jnp.concatenate((history_coefs, current_coefs, future_coefs), axis=1)      # [batch_size, 2 * n_nbd + 1]

    history_nbd = history_observations[:, -n_nbd:, :]                                  # [batch_size, n_nbd, state_dim]
    future_nbd = future_observations[:, :n_nbd, :]

    # [batch_size, 2 * n_nbd + 1, state_dim]
    nbds = jnp.concatenate((history_nbd, observations[:, jnp.newaxis, :], future_nbd), axis=1)

    def gae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent = ae.apply_fn(
            {"params": params},
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        gae_loss = jnp.sum((target_pred - nbds) ** 2, axis=2)                          # [batch_size, 2 * n_nbd + 1]
        gae_loss = jnp.mean(coefs * gae_loss)

        infos = {"gae_loss": gae_loss, "latent_tensor": latent}
        return gae_loss, infos

    new_gae, info = ae.apply_gradient(gae_loss_fn)
    return new_gae, info


def update_sas(
    key: int,
    sas_predictor: Model,
    observations: jnp.ndarray,      # 여기서 observation은 latent space에 있다.
    actions: jnp.ndarray,
    next_state: jnp.ndarray,
):
    # predictor_input = jnp.hstack((replay_data.observations, replay_data.actions))
    # next_state = replay_data.st_future.observations[:, 0, :]

    def sas_predictor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        next_state_pred = sas_predictor.apply_fn(
            {"params": params},
            observations,
            actions,
            deterministic=False,
            rngs={"dropout": key}
        )
        sas_loss = jnp.mean((next_state_pred - next_state) ** 2)
        return sas_loss, {"sas_loss": sas_loss}

    new_actor, info = sas_predictor.apply_gradient(sas_predictor_loss_fn)
    return new_actor, info


def update_wae(
    rng: jnp.ndarray,
    ae: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    recon_target: jnp.ndarray,
    **kwargs
) -> Tuple[Model, Dict]:

    history = jnp.concatenate((history_observations, history_actions), axis=2)
    rng, key = jax.random.split(rng)
    dropout_key, mmd_key, decoder_key, noise_key = jax.random.split(rng, 4)

    def wae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent = ae.apply_fn(
            {"params": params},
            history,
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
        )

        mmd_loss = ae.apply_fn(
            {"params": params},
            z=latent,
            key=key,
            rngs={"dropout": dropout_key},
            method=WassersteinAutoEncoder.rbf_mmd_loss
        )

        recon_loss = jnp.mean((target_pred - recon_target) ** 2)

        wae_loss = (recon_loss + mmd_loss)
        infos = {
            "wae_loss": wae_loss,
            "wae_recon_loss": recon_loss,
            "wae_mmd_loss": mmd_loss,
            "wae_recon": target_pred
        }
        return wae_loss, infos

    new_wae, info = ae.apply_gradient(wae_loss_fn)
    return new_wae, info


@jax.jit
def update_wae_by_td3_policy_grad_flow(
    gamma: float,
    rng: jnp.ndarray,
    second_ae: Model,
    actor_target: Model,
    critic: Model,
    critic_target: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    latent_next_observation: jnp.ndarray,           # 이거 fix 시키자
    latent_goal_observation: jnp.ndarray,
    dones: jnp.ndarray,
    target_noise: float,
    target_noise_clip: float,
    **kwargs,
):
    rng, dropout_key, sampling_key, noise_key, decoder_key = jax.random.split(rng, 5)

    history = jnp.concatenate((history_observations, history_actions), axis=2)
    history_cat = history[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_cat, current), axis=1)

    def wae_grad_flow_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        current_conditioned_latent, _ = second_ae.apply_fn(
            {"params": params},
            history,
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )
        next_conditioned_latent, _ = second_ae.apply_fn(
            {"params": params},
            history_for_next_timestep,
            next_observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )

        next_actions = actor_target(
            next_observations,
            next_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        noise = jax.random.normal(noise_key, shape=next_actions.shape) * (target_noise ** 2)
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        next_q_values = critic_target(
            next_observations,
            next_conditioned_latent,
            next_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        next_q_values = jnp.min(next_q_values, axis=1)

        rewards = - jnp.sum((latent_next_observation - latent_goal_observation) ** 2, axis=-1)
        target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

        q_values = critic(
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        n_qs = q_values.shape[1]
        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs

        critic_loss = (critic_loss / len(q_values))

        actor_loss = -critic(
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        actor_loss = jnp.mean(actor_loss)

        wae_loss = actor_loss + critic_loss

        return wae_loss, {
            "wae_actor_loss": actor_loss,
            "wae_critic_loss": critic_loss,
        }

    new_second_ae, second_ae_info = second_ae.apply_gradient(wae_grad_flow_loss_fn)
    return new_second_ae, second_ae_info


@jax.jit
def update_gae_by_td3_policy_grad_flow(
    gamma: float,
    rng: jnp.ndarray,
    ae: Model,
    actor_target: Model,
    critic: Model,
    critic_target: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    latent_next_observation: jnp.ndarray,           # 이거 fix 시키자
    latent_goal_observation: jnp.ndarray,
    dones: jnp.ndarray,
    target_noise: float,
    target_noise_clip: float,
    **kwargs,
):
    rng, dropout_key, sampling_key, noise_key = jax.random.split(rng, 4)

    def gae_grad_flow_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, current_conditioned_latent = ae.apply_fn(
            {"params": params},
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        _, next_conditioned_latent = ae.apply_fn(
            {"params": params},
            next_observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        next_actions = actor_target(
            next_observations,
            next_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        noise = jax.random.normal(noise_key, shape=next_actions.shape) * (target_noise ** 2)
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        next_q_values = critic_target(
            next_observations,
            next_conditioned_latent,
            next_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        next_q_values = jnp.min(next_q_values, axis=1)

        rewards = - jnp.sum((latent_next_observation - latent_goal_observation) ** 2, axis=-1)
        target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

        q_values = critic(
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        n_qs = q_values.shape[1]
        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs

        critic_loss = (critic_loss / len(q_values))

        actor_loss = -critic(
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )[:, 0, ...]
        actor_loss = jnp.mean(actor_loss)

        gae_loss = actor_loss + critic_loss

        return gae_loss, {
            "wae_actor_loss": actor_loss,
            "wae_critic_loss": critic_loss,
        }

    new_ae, ae_info = ae.apply_gradient(gae_grad_flow_loss_fn)
    return new_ae, ae_info



@jax.jit
def update_wae_by_policy_grad_flow(
    rng: jnp.ndarray,
    second_ae: Model,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
):
    rngs, ae_dropout_key, ae_noise_key, actor_dropout, q_dropout, sample_key = jax.random.split(rng, 6)

    ent_coef = jnp.exp(log_ent_coef())

    history_input = jnp.concatenate((history_observations, history_actions), axis=2)

    def wae_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        current_conditioned_latent, _ = second_ae.apply_fn(
            {"params": params},
            history_input,
            observations,
            deterministic=False,
            rngs={"dropout": ae_dropout_key, "noise": ae_noise_key}
        )
        action_dist = actor(
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": actor_dropout}
        )
        actions_pi = action_dist.sample(seed=sample_key)
        log_prob = action_dist.log_prob(actions_pi)

        q_values_pi = critic(observations, current_conditioned_latent, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        wae_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)
        return wae_loss, {"wae_loss": wae_loss}

    new_second_ae, second_ae_loss_info = second_ae.apply_gradient(wae_loss_fn)

    return new_second_ae, second_ae_loss_info


def update_sac_critic(      # 우선 그냥 SAC로 구현한다. 이건 Reward가 없는 상태에서 goal-conditioned RL 로 formulation해서 optimize 함
    rng: jnp.ndarray,
    gamma: float,
    actor: Model,
    sas_predictor: Model,                       # To compute the next state
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    latent_observation: jnp.ndarray,
    latent_goal_observation: jnp.ndarray,       #
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
    next_conditioned_latent: jnp.ndarray,
):
    rng, dropout_key, sampling_key = jax.random.split(rng, 3)
    next_action_dist = actor(
        next_observations,
        next_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_action = next_action_dist.sample(seed=rng)
    next_log_prob = next_action_dist.log_prob(next_action)

    next_q_values = critic_target(
        next_observations,
        next_conditioned_latent,
        next_action,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_q_values = jnp.min(next_q_values, axis=1)
    ent_coef = jnp.exp(log_ent_coef())
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

    # Compute the reward using second auto encoder and sas predictor
    rng, sampling_key = jax.random.split(rng)
    pred_actions = actor(
        observations,
        current_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    pred_actions = pred_actions.sample(seed=sampling_key)
    latent_next_observation = sas_predictor(
        latent_observation,
        pred_actions,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    rewards = - jnp.sum((latent_next_observation - latent_goal_observation) ** 2, axis=-1)
    target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_values = critic.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        n_qs = q_values.shape[1]
        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs

        critic_loss = (critic_loss / len(q_values))
        return critic_loss, {"critic_loss": critic_loss}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_sac_critic_with_reward(      # Online finetuning 당시에 사용함
    rng: jnp.ndarray,
    gamma: float,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
    next_conditioned_latent: jnp.ndarray,
):
    rng, dropout_key, sampling_key = jax.random.split(rng, 3)
    next_action_dist = actor(
        next_observations,
        next_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_action = next_action_dist.sample(seed=rng)
    next_log_prob = next_action_dist.log_prob(next_action)

    next_q_values = critic_target(
        next_observations,
        next_conditioned_latent,
        next_action
    )
    next_q_values = jnp.min(next_q_values, axis=1)
    ent_coef = jnp.exp(log_ent_coef())
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

    target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_values = critic.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        n_qs = q_values.shape[1]
        critic_loss = sum([(target_q_values - q_values[:, i, :].squeeze()) ** 2 for i in range(n_qs)])
        critic_loss = jnp.mean(critic_loss)

        critic_loss = (critic_loss / len(q_values))
        return critic_loss, {"critic_loss": critic_loss}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_sac_actor(
    rng: jnp.ndarray,
    actor: Model,
    sas_predictor: Model,                       # To compute the next state
    critic: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    latent_observation: jnp.ndarray,
    latent_goal_observation: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
):
    # NOTE 1. SAC Loss
    rngs, dropout_key, sample_key = jax.random.split(rng, 3)
    ent_coef = jnp.exp(log_ent_coef())

    def sac_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        action_dist = actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = action_dist.sample(seed=sample_key)
        log_prob = action_dist.log_prob(actions_pi)

        q_values_pi = critic(
            observations,
            current_conditioned_latent,
            actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        coef_lambda = 2.5 / jnp.mean((jnp.abs(min_qf_pi)))
        sac_actor_loss = coef_lambda * jnp.mean(ent_coef * log_prob - min_qf_pi)
        return sac_actor_loss, {"sac_actor_loss": sac_actor_loss}

    new_actor, sac_loss_info = actor.apply_gradient(sac_loss_fn)

    # NOTE 2. Behavior Cloning Loss
    rngs, dropout_key, sample_key = jax.random.split(rngs, 3)
    actions = jnp.clip(actions, -1+EPS, 1-EPS)        # Avoid Nan due to +-1.0 actions.

    def bc_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        action_dist = new_actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        bc_log_prob = jnp.mean(action_dist.log_prob(actions))

        bc_loss = -bc_log_prob
        return bc_loss, {"bc_loss": bc_loss, "log_prob": bc_log_prob}

    new_actor, bc_loss_info = new_actor.apply_gradient(bc_loss_fn)

    # NOTE 3. Goal Reaching Loss
    rngs, dropout_key, sample_key = jax.random.split(rngs, 3)

    def goal_reaching_loss_fn(params: Params):
        action_dist = new_actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = action_dist.sample(seed=sample_key)
        latent_next_observation = sas_predictor(latent_observation, actions_pi)

        goal_reaching_loss = jnp.mean((latent_next_observation - latent_goal_observation) ** 2)
        return goal_reaching_loss, {"goal_reaching_loss": goal_reaching_loss}

    new_actor, goal_reaching_loss_info = new_actor.apply_gradient(goal_reaching_loss_fn)

    return new_actor, {**sac_loss_info, **bc_loss_info, **goal_reaching_loss_info}


def update_sac_actor_with_reward(                   # For online finetuning!
    rng: jnp.ndarray,
    actor: Model,
    sas_predictor: Model,                       # To compute the next state
    critic: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    latent_observation: jnp.ndarray,
    latent_goal_observation: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
):
    # NOTE 1. SAC Loss
    rngs, dropout_key, sample_key = jax.random.split(rng, 3)
    ent_coef = jnp.exp(log_ent_coef())

    def sac_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:        # Reward를 높이는 방향으로 학습을 진행
        action_dist = actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = action_dist.sample(seed=sample_key)
        log_prob = action_dist.log_prob(actions_pi)

        q_values_pi = critic(observations, current_conditioned_latent, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        sac_actor_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)
        return sac_actor_loss, {"sac_actor_loss": sac_actor_loss}

    new_actor, sac_loss_info = actor.apply_gradient(sac_loss_fn)

    # NOTE 2. Goal Reaching Loss                # Landmark 주변으로 가도록 Policy를 restriction
    rngs, dropout_key, sample_key = jax.random.split(rngs, 3)

    def goal_reaching_loss_fn(params: Params):
        action_dist = actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = action_dist.sample(seed=sample_key)
        latent_next_observation = sas_predictor(latent_observation, actions_pi)

        goal_reaching_loss \
            = jnp.mean((latent_next_observation - latent_goal_observation) ** 2) * USE_GOAL_REACHING_LOSS
        return goal_reaching_loss, {"goal_reaching_loss": goal_reaching_loss}

    new_actor, goal_reaching_loss_info = new_actor.apply_gradient(goal_reaching_loss_fn)

    return new_actor, {**sac_loss_info, **goal_reaching_loss_info}


def update_td3_critic(
    rng: jnp.ndarray,
    gamma: float,
    actor_target: Model,
    critic: Model,
    critic_target: Model,
    sas_predictor: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
    next_conditioned_latent: jnp.ndarray,
    target_noise: float,
    target_noise_clip: float,
):
    rng, dropout_key, sampling_key, noise_key = jax.random.split(rng, 4)

    next_actions = actor_target(
        next_observations,
        next_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )

    noise = jax.random.normal(noise_key, shape=next_actions.shape) * (target_noise ** 2)
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    next_q_values = critic_target(
        next_observations,
        next_conditioned_latent,
        next_actions,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_q_values = jnp.min(next_q_values, axis=1)

    current_action = actor_target(
        observations,
        current_conditioned_latent,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_latent_observation = sas_predictor(current_conditioned_latent, current_action)
    rewards = jnp.sum((next_latent_observation - current_conditioned_latent) ** 2, axis=1)

    target_q_values = rewards + gamma * (1 - jnp.squeeze(dones)) * jnp.squeeze(next_q_values)

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_values = critic.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        n_qs = q_values.shape[1]
        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, :].squeeze())) ** 2 for i in range(n_qs)]) / n_qs
        critic_loss = (critic_loss / len(q_values))

        rand_actions = jax.random.uniform(key=rng, shape=actions.shape)
        rand_q_val = critic.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            rand_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        rand_q_val = jnp.mean(rand_q_val)

        total_loss = critic_loss + RAND_QVAL_COEF * (jnp.mean(q_values) ** 2)

        return total_loss, {"critic_loss": critic_loss, "rand_q_val": rand_q_val,
                            "target_q_val": jnp.mean(target_q_values),
                            "pred_q_val": jnp.mean(q_values),
                            "next_q_val": jnp.mean(next_q_values)}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_td3_actor(
    rng: jnp.ndarray,
    update_flag: float,
    actor: Model,
    sas_predictor: Model,                       # To compute the next state
    critic: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    latent_observation: jnp.ndarray,
    latent_goal_observation: jnp.ndarray,
    current_conditioned_latent: jnp.ndarray,    # Q and Actor are conditioned on latent vec. == Output of second ae.
):
    # NOTE 1. TD3 Loss
    rngs, dropout_key, sample_key = jax.random.split(rng, 3)

    def td3_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        actions_pi = actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        q_values_pi = critic(
            observations,
            current_conditioned_latent,
            actions_pi,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        q_values_pi = q_values_pi[:, 0, ...]

        coef_lambda = 2.5 / jnp.mean((jnp.abs(q_values_pi)))

        td3_actor_loss = coef_lambda * jnp.mean(-q_values_pi) * update_flag
        return td3_actor_loss, {"td3_actor_loss": td3_actor_loss, "coef_lambda": coef_lambda}

    new_actor, td3_loss_info = actor.apply_gradient(td3_loss_fn)

    # NOTE 2. Behavior Cloning Loss
    rngs, dropout_key, sample_key = jax.random.split(rngs, 3)

    def bc_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        actions_pi = new_actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        bc_loss = jnp.mean((actions_pi - actions) ** 2) * update_flag
        return bc_loss, {"bc_loss": bc_loss}

    new_actor, bc_loss_info = new_actor.apply_gradient(bc_loss_fn)

    # NOTE 3. Goal Reaching Loss
    rngs, dropout_key, sample_key = jax.random.split(rngs, 3)

    def goal_reaching_loss_fn(params: Params):
        actions_pi = new_actor.apply_fn(
            {"params": params},
            observations,
            current_conditioned_latent,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        latent_next_observation = sas_predictor(latent_observation, actions_pi)

        goal_reaching_loss = jnp.mean((latent_next_observation - latent_goal_observation) ** 2) * update_flag
        return goal_reaching_loss, {"goal_reaching_loss": goal_reaching_loss}

    new_actor, goal_reaching_loss_info = new_actor.apply_gradient(goal_reaching_loss_fn)

    return new_actor, {**td3_loss_info, **bc_loss_info, **goal_reaching_loss_info}