from collections import defaultdict
import functools
import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from .core_comp import (

    sac_actor_update,
    sac_critic_update,
    sac_with_bc_actor_update,

    update_cond_vae,
    pgis_actor_update,
    pgis_critic_update,
    pgis_log_ent_coef_update,

    log_ent_coef_update,
    sensor_based_single_state_discriminator_update,
    sensor_based_double_state_discriminator_update,
    sensor_based_action_matcher_update,
    disc_weighted_actor_update,
    disc_weighted_critic_update,

    inverse_dynamics_update,
    bc_included_actor_update,
    naivebc_update,

    target_update,

    dummy_actor_update,
)


@jax.jit
def cond_vae_goal_generator_update(
    rng: jnp.ndarray,
    vae: Model,
    observations: jnp.ndarray,
    subgoal_observations: jnp.ndarray,
    goal_observations: jnp.ndarray,          # x, y pos. Not image
    target_future_hop: jnp.ndarray
):
    new_vae, vae_info = update_cond_vae(
        rng=rng,
        vae=vae,
        observations=observations,
        subgoal_observations=subgoal_observations,
        goal_observations=goal_observations,
        target_future_hop=target_future_hop
    )

    return new_vae, vae_info


@jax.jit
def posgoal_imgsubgoal_sac_update(      # pgis
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    subgoal_generator: Model,           # == vae

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    goals: jnp.ndarray,

    target_entropy: float,              # For SAC
    gamma: float,                       # Discount factor
    tau: float                          # Soft target update
):
    rng, sampling_key= jax.random.split(rng)

    subgoals, *_ = subgoal_generator(        # Fix the subgoal. We don't train subgoal generator anymore.
        observations,
        goals,
        deterministic=False,
        rngs={"sampling": sampling_key}
    )

    rng, _ = jax.random.split(rng)
    critic, critic_info = pgis_critic_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        subgoals=subgoals,
        goals=goals,
        dones=dones,
        gamma=gamma
    )

    critic_target = target_update(critic, critic_target, tau)

    rng, _ = jax.random.split(rng)

    ent_coef, ent_coef_info = pgis_log_ent_coef_update(
        rng=rng,
        target_entropy=target_entropy,
        log_ent_coef=log_ent_coef,
        actor=actor,
        observations=observations,
        subgoals=subgoals,
        goals=goals
    )

    rng, _ = jax.random.split(rng)
    actor, actor_info = pgis_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations,
        subgoals=subgoals,
        goals=goals
    )

    new_models = {
        "actor": actor,
        "critic": critic,
        "critic_target": critic_target,
        "ent_coef": ent_coef
    }

    return new_models, {**critic_info, **actor_info, **ent_coef_info}


@jax.jit
def prerequisite_behavior_cloner_update(
    rng: jnp.ndarray,
    behavior_cloner: Model,
    expert_observations: jnp.ndarray,
    expert_actions: jnp.ndarray            # Relabled actions
):
    behavior_cloner, bc_info = naivebc_update(
        rng=rng,
        behavior_cloner=behavior_cloner,
        expert_observations=expert_observations,
        expert_actions=expert_actions
    )
    return behavior_cloner, bc_info


@jax.jit
def prerequisite_inverse_dynamics_update(
    rng: jnp.ndarray,
    inverse_dynamics: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray
):
    inverse_dynamics, inv_dyna_info = inverse_dynamics_update(
        rng=rng,
        inverse_dynamics=inverse_dynamics,
        observations=observations,
        actions=actions,
        next_observations=next_observations
    )
    return inverse_dynamics, inv_dyna_info


@functools.partial(jax.jit, static_argnames=("update_entropy", ))
def warmup_bc_types(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    behavior_cloner: Model,
    single_state_discriminator: Model,
    double_state_discriminator: Model,          # Action matcher 때문에, 어쩔 수 없이 필요함...
    action_matcher: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,

    update_entropy: bool,                  # For SAC
    target_entropy: float,                 # For SAC
    gamma: float,                          # Discount factor
    tau: float,                            # Soft target update,

    **kwargs
):
    single_state_discriminator, single_disc_info = sensor_based_single_state_discriminator_update(
        rng=rng,
        discriminator=single_state_discriminator,
        expert_observation=expert_observation,
        observation=observations,
    )

    double_state_discriminator, double_disc_info = sensor_based_double_state_discriminator_update(
        rng=rng,
        discriminator=double_state_discriminator,
        expert_observation=expert_observation,
        expert_next_observation=expert_next_observation,
        observation=observations,
        next_observation=next_observations
    )
    action_matcher_coefs = double_disc_info["double_policy_disc_score"]

    _, rng = jax.random.split(rng)
    action_matcher, action_matcher_info = sensor_based_action_matcher_update(
        rng=rng,
        action_matcher=action_matcher,
        behavior_cloner=behavior_cloner,
        observations=observations,
        actions=actions,
        action_matcher_coefs=action_matcher_coefs
    )

    _, rng = jax.random.split(rng)
    critic, critic_info = disc_weighted_critic_update(
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

    if update_entropy:
        _, rng = jax.random.split(rng)
        log_ent_coef, ent_coef_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        ent_coef_info = {"ent_coef": jnp.exp(log_ent_coef()), "ent_coef_loss": 0}

    _, rng = jax.random.split(rng)
    actor, actor_info = disc_weighted_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        behavior_cloner=behavior_cloner,
        action_matcher=action_matcher,
        policy_disc_score=0.0,
        observations=observations,
    )

    critic_target = target_update(critic, critic_target, tau=tau)

    models = {
        "single_state_discriminator": single_state_discriminator,
        "double_state_discriminator": double_state_discriminator,
        "action_matcher": action_matcher,
        "critic": critic,
        "critic_target": critic_target,
        "actor": actor,
        "log_ent_coef": log_ent_coef,
    }

    infos = defaultdict(int)
    infos.update({**single_disc_info, **double_disc_info, **action_matcher_info, **critic_info, **actor_info, **ent_coef_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=("update_entropy", "target_entropy", "gamma", "tau"))
def sensor_based_single_state_amsopt_sac_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    behavior_cloner: Model,
    single_state_discriminator: Model,
    double_state_discriminator: Model,          # Action matcher 때문에, 어쩔 수 없이 필요함...
    action_matcher: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,

    update_entropy: bool,                  # For SAC
    target_entropy: float,                 # For SAC
    gamma: float,                          # Discount factor
    tau: float,                            # Soft target update,

    **kwargs
):
    single_state_discriminator, single_disc_info = sensor_based_single_state_discriminator_update(
        rng=rng,
        discriminator=single_state_discriminator,
        expert_observation=expert_observation,
        observation=observations,
    )
    policy_disc_score = single_disc_info["single_policy_disc_score"]

    double_state_discriminator, double_disc_info = sensor_based_double_state_discriminator_update(
        rng=rng,
        discriminator=double_state_discriminator,
        expert_observation=expert_observation,
        expert_next_observation=expert_next_observation,
        observation=observations,
        next_observation=next_observations
    )
    action_matcher_coefs = double_disc_info["double_policy_disc_score"]

    _, rng = jax.random.split(rng)
    action_matcher, action_matcher_info = sensor_based_action_matcher_update(
        rng=rng,
        action_matcher=action_matcher,
        behavior_cloner=behavior_cloner,
        observations=observations,
        actions=actions,
        action_matcher_coefs=action_matcher_coefs
    )

    _, rng = jax.random.split(rng)
    critic, critic_info = disc_weighted_critic_update(
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

    if update_entropy:
        _, rng = jax.random.split(rng)
        log_ent_coef, ent_coef_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        ent_coef_info = {"ent_coef": jnp.exp(log_ent_coef()), "ent_coef_loss": 0}

    _, rng = jax.random.split(rng)
    actor, actor_info = disc_weighted_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        behavior_cloner=behavior_cloner,
        action_matcher=action_matcher,
        policy_disc_score=policy_disc_score,
        observations=observations,
    )

    critic_target = target_update(critic, critic_target, tau=tau)

    models = {
        "single_state_discriminator": single_state_discriminator,
        "double_state_discriminator": double_state_discriminator,
        "action_matcher": action_matcher,
        "critic": critic,
        "critic_target": critic_target,
        "actor": actor,
        "log_ent_coef": log_ent_coef,
    }

    infos = defaultdict(int)
    infos.update({**single_disc_info, **double_disc_info, **action_matcher_info, **critic_info, **actor_info, **ent_coef_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=("target_entropy", "gamma", "tau"))
def sensor_based_double_state_amsopt_sac_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    behavior_cloner: Model,
    discriminator: Model,
    action_matcher: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,

    target_entropy: float,              # For SAC
    gamma: float,                       # Discount factor
    tau: float,                          # Soft target update

    **kwargs
):
    discriminator, disc_info = sensor_based_double_state_discriminator_update(
        rng=rng,
        discriminator=discriminator,
        expert_observation=expert_observation,
        expert_next_observation=expert_next_observation,
        observation=observations,
        next_observation=next_observations
    )
    policy_disc_score = disc_info["policy_disc_score"]

    _, rng = jax.random.split(rng)
    action_matcher, action_matcher_info = sensor_based_action_matcher_update(
        rng=rng,
        action_matcher=action_matcher,
        behavior_cloner=behavior_cloner,
        observations=observations,
        actions=actions,
        action_matcher_coefs=policy_disc_score
    )

    _, rng = jax.random.split(rng)
    critic, critic_info = disc_weighted_critic_update(
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

    _, rng = jax.random.split(rng)
    log_ent_coef, ent_coef_info = log_ent_coef_update(
        rng=rng,
        log_ent_coef=log_ent_coef,
        actor=actor,
        observations=observations,
        target_entropy=target_entropy
    )

    _, rng = jax.random.split(rng)
    actor, actor_info = disc_weighted_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        behavior_cloner=behavior_cloner,
        action_matcher=action_matcher,
        policy_disc_score=policy_disc_score,
        observations=observations,
    )

    critic_target = target_update(critic, critic_target, tau=tau)

    models = {
        "discriminator": discriminator,
        "action_matcher": action_matcher,
        "critic": critic,
        "critic_target": critic_target,
        "actor": actor,
        "log_ent_coef": log_ent_coef,
    }
    infos = defaultdict(int)
    infos.update({**disc_info, **action_matcher_info, **critic_info, **actor_info, **ent_coef_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=("update_entropy", "target_entropy", "gamma", "tau"))
def sensor_based_inverse_dynamics_without_disc_amsopt_sac_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    inverse_dynamics: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,

    update_entropy: bool,               # For SAC
    target_entropy: float,              # For SAC
    gamma: float,                       # Discount factor
    tau: float,                         # Soft target update,

    **kwargs
):
    """
    Discriminator 없이, RL을 하면서 추가적인 term으로 Behavior cloning term을 줄 뿐이다.
    이 때, behavior cloning할 action은 inverse dynamics로 예측한다.
    """

    _, rng = jax.random.split(rng)
    inverse_dynamics, inv_dyna_info = inverse_dynamics_update(
        rng=rng,
        inverse_dynamics=inverse_dynamics,
        observations=observations,
        actions=actions,
        next_observations=next_observations
    )

    _, rng = jax.random.split(rng)
    critic, critic_info = disc_weighted_critic_update(
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

    if update_entropy:
        _, rng = jax.random.split(rng)
        log_ent_coef, ent_coef_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        ent_coef_info = {"ent_coef": jnp.exp(log_ent_coef()), "ent_coef_loss": 0}

    expert_actions = inverse_dynamics(expert_observation, expert_next_observation)
    actor, actor_info = bc_included_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations,
        expert_observations=expert_observation,
        expert_actions=expert_actions
    )

    critic_target = target_update(critic, critic_target, tau=tau)

    models = {
        "inverse_dynamics": inverse_dynamics,
        "critic": critic,
        "critic_target": critic_target,
        "actor": actor,
        "log_ent_coef": log_ent_coef,
    }
    infos = defaultdict(int)
    infos.update({**inv_dyna_info, **critic_info, **ent_coef_info, **actor_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=("update_entropy", "target_entropy", "gamma", "tau"))
def dummy_sac_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    update_entropy: bool,               # For SAC
    target_entropy: float,              # For SAC
    gamma: float,                       # Discount factor
    tau: float,                         # Soft target update,

    **kwargs
):
    """
    Discriminator 없이, RL을 하면서 추가적인 term으로 Behavior cloning term을 줄 뿐이다.
    이 때, behavior cloning할 action은 inverse dynamics로 예측한다.
    """

    _, rng = jax.random.split(rng)
    critic, critic_info = disc_weighted_critic_update(
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

    if update_entropy:
        _, rng = jax.random.split(rng)
        log_ent_coef, ent_coef_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        ent_coef_info = {"ent_coef": jnp.exp(log_ent_coef()), "ent_coef_loss": 0}

    actor, actor_info = dummy_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations,
    )

    critic_target = target_update(critic, critic_target, tau=tau)

    models = {
        "critic": critic,
        "critic_target": critic_target,
        "actor": actor,
        "log_ent_coef": log_ent_coef,
    }
    infos = defaultdict(int)
    infos.update({**critic_info, **ent_coef_info, **actor_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update'))
def sac_update(
    rng: jnp.ndarray,
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

    rng, _ = jax.random.split(rng, 2)
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
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, _ = jax.random.split(rng, 2)
    new_actor, actor_info = sac_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations
    )

    rng, _ = jax.random.split(rng, 2)
    if entropy_update:
        new_log_ent_coef, ent_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        new_log_ent_coef, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    new_models = {
        "actor": new_actor,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "log_ent_coef": new_log_ent_coef
    }

    return rng, new_models, {**critic_info, **actor_info, **ent_info}


@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update'))
def sac_update_with_bc(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observations: jnp.ndarray,
    expert_actions: jnp.ndarray,

    sac_coef: float,
    bc_coef: float,

    gamma: float,
    target_entropy: float,
    tau: float,
    target_update_cond: bool,
    entropy_update: bool
):

    rng, _ = jax.random.split(rng, 2)
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
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, _ = jax.random.split(rng, 2)
    new_actor, actor_info = sac_with_bc_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,

        sac_coef=sac_coef,
        bc_coef=bc_coef,

        observations=observations,
        expert_observations=expert_observations,
        expert_actions=expert_actions
    )

    rng, _ = jax.random.split(rng, 2)
    if entropy_update:
        new_log_ent_coef, ent_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        new_log_ent_coef, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    new_models = {
        "actor": new_actor,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "log_ent_coef": new_log_ent_coef
    }

    return rng, new_models, {**critic_info, **actor_info, **ent_info}