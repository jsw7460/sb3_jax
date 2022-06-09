from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Params, InfoDict


EPS = 1e-6
ALPHA = 2.5
POSITIVE_TARGET = 1
NEGATIVE_TARGET = 0


def log_ent_coef_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    observations: jnp.ndarray,
    target_entropy: float,
) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(ent_params: Params):
        dist = actor(observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_with_bc_actor_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    sac_coef: float,
    bc_coef: float,

    observations: jnp.ndarray,

    expert_observations: jnp.ndarray,
    expert_actions: jnp.ndarray
):
    expert_actions = jnp.clip(expert_actions, -1.0 + EPS, 1.0 - EPS)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        sac_loss = (ent_coef * log_prob - min_qf_pi).mean() * sac_coef

        expert_dist = actor.apply_fn({"params": actor_params}, expert_observations)
        expert_log_prob = expert_dist.log_prob(expert_actions)

        bc_loss = - jnp.mean(expert_log_prob) * bc_coef
        actor_loss = sac_loss + bc_loss

        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob, "bc_loss": bc_loss, "sac_loss": sac_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(
    rng: jnp.ndarray,
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
    dist = actor(next_observations)
    next_actions = dist.sample(seed=rng)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values= critic.apply_fn({'params': critic_params}, observations, actions)

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_cond_vae(
    rng: jnp.ndarray,
    vae: Model,
    observations: jnp.ndarray,
    subgoal_observations: jnp.ndarray,
    goal_observations: jnp.ndarray,
    target_future_hop: jnp.ndarray
):
    dropout_key, sampling_key = jax.random.split(rng)
    def vae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        recon, latent, stats = vae.apply_fn(
            {"params": params},
            observations,
            goal_observations,
            target_future_hop,
            deterministic=False,
            rngs={"dropout": dropout_key, "sampling": sampling_key}
        )
        mu, log_std = stats
        # Reconstruction loss
        recon_loss = jnp.mean((recon - subgoal_observations) ** 2)

        # KL prior loss
        std = jnp.exp(log_std)
        dim = std.shape[1]
        kl_loss = 0.5 * (
            - jnp.log(jnp.prod(std ** 2 + 1e-8, axis=1, keepdims=True))
            - dim
            + jnp.sum(std ** 2, axis=1, keepdims=True)
            + jnp.sum(mu ** 2, axis=1, keepdims=True)
        )
        kl_loss = jnp.mean(kl_loss)

        vae_loss = recon_loss + 0.01 * kl_loss

        infos = {"kl_loss": kl_loss, "recon_loss": recon_loss, "vae_loss": vae_loss}

        return vae_loss, infos

    vae, vae_info = vae.apply_gradient(vae_loss_fn)
    return vae, vae_info


def pgis_log_ent_coef_update(
    rng:jnp.ndarray,
    target_entropy: float,
    log_ent_coef: Model,
    actor: Model,

    observations: jnp.ndarray,
    subgoals: jnp.ndarray,
    goals: jnp.ndarray
) -> Tuple[Model, InfoDict]:

    dist = actor(observations, subgoals, goals, deterministic=False)
    actions_pi = dist.sample(seed=rng)
    log_prob = dist.log_prob(actions_pi)

    def temperature_loss_fn(params: Params):
        ent_coef = log_ent_coef.apply_fn({'params': params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def pgis_actor_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    subgoals: jnp.ndarray,
    goals: jnp.ndarray,
):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn(
            {'params': actor_params},
            observations,
            subgoals,
            goals,
            deterministic=False
        )
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(
            observations,
            subgoals,
            goals,
            actions_pi,
            deterministic=False
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def pgis_critic_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    subgoals: jnp.ndarray,
    goals: jnp.ndarray,
    dones: jnp.ndarray,
    gamma:float
):
    dist = actor(next_observations, subgoals, goals, deterministic=False)
    next_actions = dist.sample(seed=rng)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(
        next_observations,
        subgoals,
        goals,
        next_actions,
        deterministic=False
    )
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
            subgoals,
            goals,
            actions,
            deterministic=False
        )

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[: ,i, ...].squeeze()) ** 2)  for i in range(n_qs)])
        critic_loss = critic_loss / n_qs
        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def naivebc_update(
    rng: jnp.ndarray,
    behavior_cloner: Model,
    expert_observations: jnp.ndarray,
    expert_actions: jnp.ndarray
):
    dropout_key, _ = jax.random.split(rng)
    def bc_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_actions = behavior_cloner.apply_fn(
            {"params": params},
            expert_observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        bc_loss = jnp.mean((pred_actions - expert_actions) ** 2)
        return bc_loss, {"bc_loss": bc_loss}

    behavior_cloner, bc_info = behavior_cloner.apply_gradient(bc_loss_fn)
    return behavior_cloner, bc_info


def sensor_based_single_state_discriminator_update(
    rng: jnp.ndarray,
    discriminator: Model,
    expert_observation: jnp.ndarray,
    observation: jnp.ndarray,
):
    dropout_key1, dropout_key2 = jax.random.split(rng)
    def discriminator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        expert_score = discriminator.apply_fn(
            {"params": params},
            expert_observation,
            deterministic=False,
            rngs={"dropout": dropout_key1}
        )

        policy_score = discriminator.apply_fn(
            {"params": params},
            observation,
            deterministic=False,
            rngs={"dropout": dropout_key2}
        )

        # Bernoulli: Expert --> 1, Policy --> 0
        loss = - jnp.mean(jnp.log(expert_score) + jnp.log(1 - policy_score))

        return loss, {"single_discriminator_loss": loss, "single_expert_disc_score": expert_score, "single_policy_disc_score": policy_score}

    discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)
    return discriminator, info


def sensor_based_double_state_discriminator_update(
    rng: jnp.ndarray,
    discriminator: Model,
    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,
    observation: jnp.ndarray,
    next_observation: jnp.ndarray,
):
    dropout_key1, dropout_key2 = jax.random.split(rng)
    def discriminator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        expert_score = discriminator.apply_fn(
            {"params": params},
            expert_observation,
            expert_next_observation,
            deterministic=False,
            rngs={"dropout": dropout_key1}
        )

        policy_score = discriminator.apply_fn(
            {"params": params},
            observation,
            next_observation,
            deterministic=False,
            rngs={"dropout": dropout_key2}
        )

        # Bernoulli: Expert --> 1, Policy --> 0
        loss = - jnp.mean(jnp.log(expert_score) + jnp.log(1 - policy_score))

        return loss, {"double_discriminator_loss": loss, "double_expert_disc_score": expert_score, "double_policy_disc_score": policy_score}

    discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)
    return discriminator, info


def sensor_based_action_matcher_update(
    rng: jnp.ndarray,
    action_matcher: Model,
    behavior_cloner: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,

    action_matcher_coefs: jnp.ndarray,      # Policy score of double state discriminator
):
    """
    :param rng:
    :param action_matcher:
    :param behavior_cloner:
    :param observations:
    :param actions:
    :param next_observations:
    :param action_matcher_coefs: Output of double state discriminator. Decide whether we have to match the actions.
    :return:
    """
    dropout_key, _ = jax.random.split(rng)
    low_actions = behavior_cloner(observations)

    def action_matcher_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_actions = action_matcher.apply_fn(
            {"params": params},
            observations,
            low_actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        action_matcher_loss = jnp.mean(action_matcher_coefs * jnp.mean((pred_actions - actions) ** 2, axis=1))

        return action_matcher_loss, {"action_matcher_loss": action_matcher_loss}

    action_matcher, info = action_matcher.apply_gradient(action_matcher_loss_fn)
    return action_matcher, info


def disc_weighted_critic_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
):
    next_dist = actor(next_observations)
    next_actions = next_dist.sample(seed=rng)
    next_log_prob = next_dist.log_prob(next_actions)

    # Compute the next Q values
    next_q_values = critic_target(next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values = critic.apply_fn({'params': params}, observations, actions)

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    critic, info = critic.apply_gradient(critic_loss_fn)
    return critic, info


def disc_weighted_actor_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    log_ent_coef: Model,
    behavior_cloner: Model,
    action_matcher: Model,

    policy_disc_score: jnp.ndarray,
    observations: jnp.ndarray,
):
    low_actions = behavior_cloner(observations, deterministic=True)

    pred_high_actions = action_matcher(observations, low_actions)
    pred_high_actions = jnp.clip(pred_high_actions, -1 + EPS, 1 - EPS)

    ent_coef = jnp.exp(log_ent_coef())

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        sac_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)
        supervised_loss = - jnp.mean(policy_disc_score * dist.log_prob(pred_high_actions))

        actor_loss = sac_loss + supervised_loss
        return actor_loss, {"actor_loss": actor_loss, "sac_loss": sac_loss, "supervised_loss": supervised_loss}

    actor, info = actor.apply_gradient(actor_loss_fn)
    return actor, info


def bc_included_actor_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    expert_observations: jnp.ndarray,
    expert_actions: jnp.ndarray         # Target action of behavior cloning
):

    expert_actions = jnp.clip(expert_actions, -1.0 + EPS, 1.0 - EPS)
    ent_coef = jnp.exp(log_ent_coef())

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        coef_lambda = ALPHA / jnp.mean(jnp.abs(min_qf_pi))

        sac_loss = coef_lambda * jnp.mean(ent_coef * log_prob - min_qf_pi)

        expert_dist = actor.apply_fn({"params": params}, expert_observations)
        supervised_loss = -jnp.mean(expert_dist.log_prob(expert_actions))

        actor_loss = sac_loss + supervised_loss
        return actor_loss, {"actor_loss": actor_loss, "sac_loss": sac_loss, "supervised_loss": supervised_loss, "coef_lambda": coef_lambda}

    actor, info = actor.apply_gradient(actor_loss_fn)
    return actor, info

def inverse_dynamics_update(
    rng: jnp.ndarray,

    inverse_dynamics: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
):
    rng, dropout_key = jax.random.split(rng)
    def inverse_dynamics_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_actions = inverse_dynamics.apply_fn(
            {"params": params},
            observations,
            next_observations,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        inv_dyna_loss = jnp.mean((pred_actions - actions) ** 2)
        return inv_dyna_loss, {"inverse_dynamics_loss": inv_dyna_loss}

    inverse_dynamics, info = inverse_dynamics.apply_gradient(inverse_dynamics_loss_fn)
    return inverse_dynamics, info

def target_update(critic: Model, critic_target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic_target.params)
    return critic_target.replace(params=new_target_params)


def dummy_actor_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
):

    ent_coef = jnp.exp(log_ent_coef())

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        sac_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)

        actor_loss = sac_loss
        return actor_loss, {"actor_loss": actor_loss, "sac_loss": sac_loss}

    actor, info = actor.apply_gradient(actor_loss_fn)
    return actor, info