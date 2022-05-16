import jax
import jax.numpy as jnp

from offline_baselines_jax.common.policies import Model
from collections import defaultdict
from .core_comp import (
    # Common
    update_target,
    update_gae,
    update_wae,
    update_sas,

    # SAC
    update_entropy,
    update_sac_critic,
    update_sac_critic_with_reward,
    update_sac_actor,
    update_sac_actor_with_reward,
    update_wae_by_policy_grad_flow,
    update_sac_last_layer,

    # TD3
    update_td3_critic,
    update_td3_actor,
    update_td3_finetune_layer,
    update_td3_higher_actor_critic_last_layer,
    update_wae_by_td3_policy_grad_flow,
    update_td3_higher_actor,
    update_td3_higher_actor_critic_higher_actor,
    update_gae_by_td3_policy_grad_flow,
)


@jax.jit
def _metla_online_finetune_warmup_last_layer(
    rng: jnp.ndarray,
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    finetune_layer: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    **kwargs
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)

    history_input = jnp.concatenate((history_observations, history_actions), axis=2)
    label, _ = second_ae(
        history_input,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    def warmup_loss_fn(params):
        pred = finetune_layer.apply_fn(
            {"params": params},
            label,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        warmup_loss = jnp.mean((pred - label) ** 2) * 1000
        return warmup_loss, {"warmup_loss": warmup_loss}

    new_finetune_layer, finetune_layer_info = finetune_layer.apply_gradient(warmup_loss_fn)
    return rng, finetune_layer_info, {"finetune_layer": new_finetune_layer}


@jax.jit
def _metla_online_finetune_warmup_higher_actor(
    rng: jnp.ndarray,
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    higher_actor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    **kwargs
):
    rng, dropout_key, noise_key = jax.random.split(rng, 3)

    history_input = jnp.concatenate((history_observations, history_actions), axis=2)
    label, _ = second_ae(
        history_input,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    def warmup_loss_fn(params):
        pred, _ = higher_actor.apply_fn(
            {"params": params},
            history_input,
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )
        warmup_loss = jnp.mean((pred - label) ** 2)
        return warmup_loss, {"warmup_loss": warmup_loss}

    new_finetune_layer, finetune_layer_info = higher_actor.apply_gradient(warmup_loss_fn)
    return rng, finetune_layer_info, {"finetune_layer": new_finetune_layer}


@jax.jit
def _metla_td3_online_finetune_higher_actor(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    higher_critic: Model,
    higher_critic_target: Model,
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    higher_actor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    higher_actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,           # NOTE: In online finetuning, we access to reward signal
    dones: jnp.ndarray,
    **kwargs
):
    # new_finetune_layer, finetune_info = update_td3_higher_actor(
    #     gamma=gamma,
    #     rng=rng,
    #     higher_actor=higher_actor,
    #     second_ae=second_ae,
    #     higher_critic=higher_critic,
    #     higher_critic_target=higher_critic_target,
    #     history_observations=history_observations,
    #     history_actions=history_actions,
    #     observations=observations,
    #     actions=actions,
    #     higher_actions=higher_actions,
    #     next_observations=next_observations,
    #     rewards=rewards,
    #     dones=dones,
    # )

    new_finetune_layer, new_higher_critic, actor_critic_info = update_td3_higher_actor_critic_higher_actor(
        gamma=gamma,
        rng=rng,
        higher_actor=higher_actor,
        higher_critic=higher_critic,
        higher_critic_target=higher_critic_target,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
        higher_actions=higher_actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
    )

    new_higher_critic_target = update_target(new_higher_critic, higher_critic_target, tau=tau)

    infos = defaultdict(int)
    # infos.update({**actor_critic_info, **finetune_info})
    infos.update({**actor_critic_info})
    new_models = {
        "finetune_layer": new_finetune_layer,
        "higher_critic": new_higher_critic,
        "higher_critic_target": new_higher_critic_target
    }
    return rng, infos, new_models


@jax.jit
def _metla_td3_online_finetune_last_layer(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    higher_critic: Model,
    higher_critic_target: Model,
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    finetune_layer: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,           # NOTE: In online finetuning, we access to reward signal
    dones: jnp.ndarray,
    **kwargs
):
    new_finetune_layer, finetune_info = update_td3_finetune_layer(
        gamma=gamma,
        rng=rng,
        finetune_layer=finetune_layer,
        second_ae=second_ae,
        higher_critic=higher_critic,
        higher_critic_target=higher_critic_target,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
    )

    new_finetune_layer, new_higher_critic, actor_critic_info = update_td3_higher_actor_critic_last_layer(
        gamma=gamma,
        rng=rng,
        finetune_layer=new_finetune_layer,
        second_ae=second_ae,
        higher_critic=higher_critic,
        higher_critic_target=higher_critic_target,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
    )
    new_higher_critic_target = update_target(new_higher_critic, higher_critic_target, tau=tau)

    infos = defaultdict(int)
    infos.update({**actor_critic_info, **finetune_info})
    new_models = {
        "finetune_layer": new_finetune_layer,
        "higher_critic": new_higher_critic,
        "higher_critic_target": new_higher_critic_target
    }
    return rng, infos, new_models


@jax.jit
def _metla_sac_online_finetune_last_layer(
    rng: jnp.ndarray,
    gamma: float,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    finetune_last_layer: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,           # NOTE: In online finetuning, we access to reward signal
    dones: jnp.ndarray,
    **kwargs
):
    new_finetune_last_layer, last_layer_info = update_sac_last_layer(
        gamma=gamma,
        key=rng,
        log_ent_coef=log_ent_coef,
        finetune_last_layer=finetune_last_layer,
        second_ae=second_ae,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones
    )
    infos = defaultdict(int)
    infos.update({**last_layer_info})
    new_models = {"finetune_last_layer": new_finetune_last_layer}
    return rng, infos, new_models


@jax.jit
def _metla_online_finetue_generator_flow(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,
    second_ae: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    **kwargs
):
    """
    학습 시키는 네트워크: Second AE(Goal generator)
    Goal generator를 학습 시키는데, Policy에 gradient를 걸어서 policy를 통해 reward를 maximizing.
    """
    new_second_ae, second_ae_info = update_wae_by_policy_grad_flow(
        rng=rng,
        second_ae=second_ae,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations
    )
    new_models = {"second_ae": new_second_ae}
    return rng, {**second_ae_info}, new_models


@jax.jit
def _metla_online_finetune_only_generator(
    rng: jnp.ndarray,
    ae: Model,                  # Generalized AE for state embedding
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
    **kwargs,
):
    """
    학습시키는 network: SAS predictor && Second AE (Goal generator)

    Goal generator를 더 좋은 goal을 찾도록 학습시킴. policy랑 Q함수는 학습 안한다
    이걸 위해서는 우선 SAS loss를 줄이는것이 선행되어야 하고, 그 다음에 SAS에 의해 예측되어지는
    landmark를 학습 시킨다.
    """
    batch_size = observations.shape[0]

    # NOTE: SAS train
    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)
    _, latent_observation = ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    new_sas, sas_info \
        = update_sas(rng, sas_predictor, latent_observation, actions, latent_next_observation)

    rng, dropout_key = jax.random.split(rng)

    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = new_sas(_latent_observations, _actions, rngs={"dropout": dropout_key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)  # [batch_size, future_len, action_dim]
    state_grads = jnp.linalg.norm(state_grads, axis=2)  # [batch_size, future_len]
    max_indices = jnp.argmax(state_grads, axis=1)  # [batch_size]

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    # NOTE: Second AE train == goal generator train
    rng, key, dropout_key, noise_key = jax.random.split(rng, 4)
    new_second_ae, second_ae_info = update_wae(
        rng=rng,
        ae=second_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        recon_target=latent_goal_observation,
    )

    infos = {**sas_info, **second_ae_info}
    new_models = {
        "sas_predictor": new_sas,
        "second_ae": new_second_ae
    }
    return rng, infos, new_models


@jax.jit
def _metla_online_finetune_actor_critic(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    target_entropy: float,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    ae: Model,                  # Generalized AE for state embedding
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,           # NOTE: In online finetuning, we access to reward signal
    dones: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
    **kwargs
):
    """
    완전 Online finetuning.
    모든 component는 각 loss를 줄이고,
    policy와 Q 함수는 모두 reward를 maximize하는 방향으로 학습을 진행한다.
    Goal generator (Second ae)는 원래 하던대로 더 좋은 goal을 찾는 방향으로만 추가 학습한다.
    """
    batch_size = observations.shape[0]
    rng, key = jax.random.split(rng)

    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)

    _, latent_observation = ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = sas_predictor(_latent_observations, _actions, rngs={"dropout": dropout_key_1})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)  # [batch_size, future_len, action_dim]
    state_grads = jnp.linalg.norm(state_grads, axis=2)  # [batch_size, future_len]
    max_indices = jnp.argmax(state_grads, axis=1)  # [batch_size]

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    rng, key, dropout_key, noise_key, decoder_key = jax.random.split(rng, 5)

    history = jnp.concatenate((history_observations, history_actions), axis=2)
    current_conditioned_latent, _ = second_ae(
        history,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
    )

    # current_conditioned_latent = second_ae_info["wae_recon"]
    history = history[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history, current), axis=1)
    next_conditioned_latent, _ = second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    # NOTE: Entropy coef train
    rng, key = jax.random.split(rng)
    new_log_ent_coef, log_ent_info = update_entropy(
        rng=rng,
        log_ent_coef=log_ent_coef,
        actor=actor,
        target_entropy=target_entropy,
        observations=observations,
        current_conditioned_latent=current_conditioned_latent
    )

    # NOTE: Critic train __________with reward__________
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_sac_critic_with_reward(
        rng=rng,
        gamma=gamma,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=new_log_ent_coef,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        current_conditioned_latent=current_conditioned_latent,
        next_conditioned_latent=next_conditioned_latent,
    )
    new_critic_target = update_target(new_critic, critic_target, tau=tau)

    # NOTE: Actor train
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_sac_actor_with_reward(
        rng=rng,
        actor=actor,
        sas_predictor=sas_predictor,
        critic=new_critic,
        log_ent_coef=new_log_ent_coef,
        observations=observations,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent
    )

    infos = defaultdict(int)
    infos.update({**log_ent_info, **critic_info, **actor_info})
    new_models = {
        "log_ent_coef": new_log_ent_coef,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor
    }
    return rng, infos, new_models


@jax.jit
def _metla_online_finetune(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    target_entropy: float,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    ae: Model,                  # Generalized AE for state embedding
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,           # NOTE: In online finetuning, we access to reward signal
    dones: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
):
    """
    완전 Online finetuning.
    모든 component는 각 loss를 줄이고,
    policy와 Q 함수는 모두 reward를 maximize하는 방향으로 학습을 진행한다.
    Goal generator (Second ae)는 원래 하던대로 더 좋은 goal을 찾는 방향으로만 추가 학습한다.
    """
    batch_size = observations.shape[0]

    # NOTE: AE train: Goal generation model training
    rng, key = jax.random.split(rng)
    new_ae, ae_info \
        = update_gae(rng, ae, history_observations, observations, future_observations, n_nbd=5)

    # NOTE: SAS train: Note that SAS is latent model. That is, input state is embedded in the latent space.
    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)

    _, latent_observation = new_ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = new_ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = new_ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    new_sas, sas_info \
        = update_sas(rng, sas_predictor, latent_observation, actions, latent_next_observation)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    rng, dropout_key = jax.random.split(rng)

    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = new_sas(_latent_observations, _actions, rngs={"dropout": dropout_key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)            # [batch_size, future_len, action_dim]
    state_grads = jnp.linalg.norm(state_grads, axis=2)              # [batch_size, future_len]
    max_indices = jnp.argmax(state_grads, axis=1)                   # [batch_size]

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    # NOTE: Second AE train         # Make a conditioning vector
    rng, key, dropout_key, noise_key = jax.random.split(rng, 4)
    new_second_ae, second_ae_info = update_wae(
        rng=key,
        ae=second_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        recon_target=latent_goal_observation
    )

    current_conditioned_latent = second_ae_info["wae_recon"]
    history = jnp.concatenate((history_observations, history_actions), axis=2)[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history, current), axis=1)
    next_conditioned_latent, _ = new_second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    # NOTE: Entropy coef train
    rng, key = jax.random.split(rng)
    new_log_ent_coef, log_ent_info = update_entropy(
        rng=rng,
        log_ent_coef=log_ent_coef,
        actor=actor,
        target_entropy=target_entropy,
        observations=observations,
        current_conditioned_latent=current_conditioned_latent
    )

    # NOTE: Critic train __________with reward__________
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_sac_critic_with_reward(
        rng=rng,
        gamma=gamma,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=new_log_ent_coef,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        current_conditioned_latent=current_conditioned_latent,
        next_conditioned_latent=next_conditioned_latent,
    )
    new_critic_target = update_target(new_critic, critic_target, tau=tau)

    # NOTE: Actor train
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_sac_actor_with_reward(
        rng=rng,
        actor=actor,
        sas_predictor=new_sas,
        critic=new_critic,
        log_ent_coef=new_log_ent_coef,
        observations=observations,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent
    )

    infos = {**log_ent_info, **ae_info, **sas_info, **second_ae_info, **critic_info, **actor_info}
    new_models = {
        "log_ent_coef": new_log_ent_coef,
        "ae": new_ae,
        "sas_predictor": new_sas,
        "second_ae": new_second_ae,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor
    }
    return rng, infos, new_models


@jax.jit
def _metla_offline_sac_update(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,
    ae: Model,                  # Generalized AE for state embedding
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
    # goal_observations: jnp.ndarray,     # Goal states for conditioned on policy and goal-conditioned-rl.
):
    batch_size = observations.shape[0]
    # NOTE: AE train
    rng, key = jax.random.split(rng)
    new_ae, ae_info \
        = update_gae(rng, ae, history_observations, observations, future_observations, n_nbd=5)

    # NOTE: SAS train: Note that SAS is latent model. That is, input state is embedded in the latent space.
    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)

    _, latent_observation = new_ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = new_ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = new_ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    new_sas, sas_info \
        = update_sas(rng, sas_predictor, latent_observation, actions, latent_next_observation)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    rng, dropout_key = jax.random.split(rng)

    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = new_sas(_latent_observations, _actions, rngs={"dropout": dropout_key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    # NOTE: Second AE train
    rng, key, dropout_key, noise_key = jax.random.split(rng, 4)
    new_second_ae, second_ae_info = update_wae(
        rng=key,
        ae=second_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        recon_target=latent_goal_observation
    )

    current_conditioned_latent = second_ae_info["wae_recon"]
    history = jnp.concatenate((history_observations, history_actions), axis=2)[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history, current), axis=1)
    next_conditioned_latent, _ = new_second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    # NOTE: Critic train
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_sac_critic(
        rng=rng,
        gamma=gamma,
        actor=actor,
        sas_predictor=new_sas,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent,
        next_conditioned_latent=next_conditioned_latent,
    )
    new_critic_target = update_target(new_critic, critic_target, tau=tau)

    # NOTE: Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_sac_actor(
        rng=rng,
        actor=actor,
        sas_predictor=new_sas,
        critic=new_critic,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent
    )

    infos = {**ae_info, **sas_info, **second_ae_info, **critic_info, **actor_info}

    new_models = {
        "ae": new_ae,
        "sas_predictor": new_sas,
        "second_ae": second_ae,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor
    }

    return rng, infos, new_models


@jax.jit
def _metla_offline_td3_update(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    target_noise: float,
    target_noise_clip: float,
    update_flag: float,
    actor: Model,
    actor_target: Model,
    critic: Model,
    critic_target: Model,
    ae: Model,                       # Generalized AE for state embedding
    second_ae: Model,                # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
):
    batch_size = observations.shape[0]
    # NOTE: AE train
    rng, key = jax.random.split(rng)
    new_ae, ae_info \
        = update_gae(rng, ae, history_observations, observations, future_observations, n_nbd=5)

    # NOTE: SAS train: Note that SAS is latent model. That is, input state is embedded in the latent space.
    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)

    _, latent_observation = new_ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = new_ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = new_ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    new_sas, sas_info \
        = update_sas(rng, sas_predictor, latent_observation, actions, latent_next_observation)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    rng, dropout_key = jax.random.split(rng)

    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = new_sas(_latent_observations, _actions, rngs={"dropout": dropout_key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    # NOTE: Second AE train
    rng, key, dropout_key, noise_key = jax.random.split(rng, 4)
    new_second_ae, second_ae_info = update_wae(
        rng=key,
        ae=second_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        recon_target=latent_goal_observation
    )

    current_conditioned_latent = second_ae_info["wae_recon"]
    history = jnp.concatenate((history_observations, history_actions), axis=2)[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history, current), axis=1)
    next_conditioned_latent, _ = new_second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    rng, key = jax.random.split(rng)

    new_critic, critic_info = update_td3_critic(
        rng=rng,
        gamma=gamma,
        actor_target=actor_target,
        sas_predictor=new_sas,
        critic=critic,
        critic_target=critic_target,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        current_conditioned_latent=current_conditioned_latent,
        next_conditioned_latent=next_conditioned_latent,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip,
    )

    new_actor, actor_info = update_td3_actor(
        rng=rng,
        update_flag=update_flag,
        actor=actor,
        sas_predictor=new_sas,
        critic=new_critic,
        observations=observations,
        actions=actions,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent
    )

    new_critic_target = update_target(new_critic, critic_target, tau=tau)
    new_actor_target = update_target(new_actor, actor_target, tau=tau)

    infos = {**ae_info, **sas_info, **second_ae_info, **critic_info, **actor_info}
    new_models = {
        "actor": new_actor,
        "actor_target": new_actor_target,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "ae": new_ae,
        "second_ae": new_second_ae,
        "sas_predictor": new_sas,
    }

    return rng, infos, new_models


@jax.jit
def _metla_offline_td3_flow_update(
    rng: jnp.ndarray,
    gamma: float,
    tau: float,
    target_noise: float,
    target_noise_clip: float,
    update_flag: float,
    actor: Model,
    actor_target: Model,
    critic: Model,
    critic_target: Model,
    ae: Model,                  # Generalized AE for state embedding
    second_ae: Model,           # Wasserstein AE for reconstruct latent goal using the history
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    future_observations: jnp.ndarray,
    future_actions: jnp.ndarray,
):
    batch_size = observations.shape[0]
    # NOTE: AE train
    rng, key = jax.random.split(rng)
    new_ae, ae_info \
        = update_gae(rng, ae, history_observations, observations, future_observations, n_nbd=5)

    # NOTE: SAS train: Note that SAS is latent model. That is, input state is embedded in the latent space.
    rng, dropout_key_1, dropout_key_2, dropout_key_3 = jax.random.split(rng, 4)

    _, latent_observation = new_ae(observations, deterministic=False, rngs={"dropout": dropout_key_1})
    _, latent_next_observation = new_ae(next_observations, deterministic=False, rngs={"dropout": dropout_key_2})
    _, latent_future = new_ae(future_observations, deterministic=False, rngs={"dropout": dropout_key_3})

    new_sas, sas_info \
        = update_sas(rng, sas_predictor, latent_observation, actions, latent_next_observation)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    rng, dropout_key = jax.random.split(rng)

    def mean_state_change(_latent_observations: jnp.ndarray, _actions: jnp.ndarray):
        next_state_pred = new_sas(_latent_observations, _actions, rngs={"dropout": dropout_key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 1)
    state_grads = diff_ft(latent_future, future_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    latent_goal_observation = latent_future[jnp.arange(batch_size), max_indices, ...]

    # NOTE: Second AE train
    rng, key, dropout_key, noise_key = jax.random.split(rng, 4)
    new_second_ae, second_ae_info = update_wae(
        rng=key,
        ae=second_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        recon_target=latent_goal_observation
    )

    rng, key = jax.random.split(rng)
    new_ae, new_ae_info = update_gae_by_td3_policy_grad_flow(
        gamma=gamma,
        rng=rng,
        ae=new_ae,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        latent_next_observation=latent_next_observation,
        latent_goal_observation=latent_goal_observation,
        dones=dones,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip
    )
    ae_info.update(new_ae_info)

    rng, key = jax.random.split(rng)


    history = jnp.concatenate((history_observations, history_actions), axis=2)
    history_for_next_timesteup = history[:, 1:, ...]
    current = jnp.hstack((observations, actions))[:, jnp.newaxis, :]
    history_for_next_timestep = jnp.concatenate((history_for_next_timesteup, current), axis=1)

    current_conditioned_latent, _ = new_second_ae(
        history,
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )
    next_conditioned_latent, _ = new_second_ae(
        history_for_next_timestep,
        next_observations,
        deterministic=False,
        rngs={"dropout": dropout_key, "noise": noise_key}
    )

    # Distance to goal state is the rewards.
    rewards = - jnp.sum((latent_next_observation - latent_goal_observation) ** 2, axis=-1)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_td3_critic(
        rng=rng,
        gamma=gamma,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        current_conditioned_latent=current_conditioned_latent,
        next_conditioned_latent=next_conditioned_latent,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip,
    )

    new_actor, actor_info = update_td3_actor(
        rng=rng,
        update_flag=update_flag,
        actor=actor,
        sas_predictor=new_sas,
        critic=new_critic,
        observations=observations,
        actions=actions,
        latent_observation=latent_observation,
        latent_goal_observation=latent_goal_observation,
        current_conditioned_latent=current_conditioned_latent
    )

    new_critic_target = update_target(new_critic, critic_target, tau=tau)
    new_actor_target = update_target(new_actor, actor_target, tau=tau)

    infos = {**ae_info, **sas_info, **second_ae_info, **critic_info, **actor_info}
    new_models = {
        "actor": new_actor,
        "actor_target": new_actor_target,
        "critic": new_critic,
        "critic_target": new_critic_target,
        "ae": new_ae,
        "second_ae": new_second_ae,
        "sas_predictor": new_sas
    }

    return rng, infos, new_models
