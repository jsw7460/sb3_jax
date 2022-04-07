from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import functools

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.buffers import ReplayBuffer, TaskDictReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, InfoDict, ReplayBufferSamples, Params
from offline_baselines_jax.sac.policies import SACPolicy

def log_prob_correction(x: jnp.ndarray) -> jnp.ndarray:
    # Squash correction (from original SAC implementation)
    return jnp.sum(jnp.log(1.0 - jnp.tanh(x) ** 2 + 1e-6), axis=1)

class LogAlphaCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_alpha', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp

class LogEntropyCoef(nn.Module):
    init_value: float = 1.0
    num_tasks: int = 10

    @nn.compact
    def __call__(self, task_id: jnp.array) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((self.num_tasks, ), jnp.log(self.init_value)))
        return jnp.sum(log_temp * task_id, keepdims=True, axis=1)

def log_alpha_update(log_alpha_coef: Model, conservative_loss: float) -> Tuple[Model, InfoDict]:
    def alpha_loss_fn(alpha_params: Params):
        alpha_coef = jnp.exp(log_alpha_coef.apply_fn({'params': alpha_params}))
        alpha_coef_loss = -alpha_coef * conservative_loss

        return alpha_coef_loss, {'alpha_coef': alpha_coef, 'alpha_coef_loss': alpha_coef_loss}

    new_alpha_coef, info = log_alpha_coef.apply_gradient(alpha_loss_fn)
    new_alpha_coef = param_clip(new_alpha_coef, 1e+6)
    return new_alpha_coef, info

def log_ent_coef_update(key:Any, log_ent_coef: Model, actor:Model , target_entropy: float, replay_data:ReplayBufferSamples,
                        task_id: jnp.array) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(ent_params: Params):
        dist = actor(replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params}, task_id)
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(key: int, actor: Model, critic:Model, log_ent_coef: Model, replay_data:ReplayBufferSamples,
                     task_id: jnp.ndarray):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, replay_data.observations)
        actions_pi = dist.sample(seed=key)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef(task_id))

        q_values_pi = critic(replay_data.observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=0)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(key:Any, actor: Model, critic: Model, critic_target: Model, log_ent_coef: Model,
                                log_alpha_coef: Model, replay_data: ReplayBufferSamples, gamma:float, conservative_weight:float,
                                lagrange_thresh:float, task_id: jnp.ndarray,):
    next_dist = actor(replay_data.next_observations)
    next_actions = next_dist.sample(seed=key)
    next_log_prob = next_dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(replay_data.next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=0)

    ent_coef = jnp.exp(log_ent_coef(task_id))
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

    repeated_observations = dict()
    repeated_observations['obs'] = jnp.repeat(replay_data.observations['obs'], repeats=10, axis=0)
    repeated_observations['task'] = jnp.repeat(replay_data.observations['task'], repeats=10, axis=0)

    key, subkey = jax.random.split(key, 2)
    random_actions = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(batch_size * 10, action_dim))

    random_density = jnp.log(0.5 ** action_dim)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q = critic.apply_fn({'params': critic_params}, replay_data.observations, replay_data.actions)
        cql_q = critic.apply_fn({'params': critic_params}, replay_data.observations, cql_actions)
        random_q = critic.apply_fn({'params': critic_params}, repeated_observations, random_actions)
        conservative_loss = 0
        for idx in range(len(cql_q)):
            conservative_loss += jax.scipy.special.logsumexp(jnp.ndarray(
                [jnp.repeat(cql_q[idx], repeats=10, axis=0) - cql_log_prob, random_q[idx] - random_density])).mean() - \
                                 current_q[idx].mean()

        conservative_loss = (conservative_weight * ((conservative_loss) / len(cql_q)) - lagrange_thresh)
        # Compute critic loss
        critic_loss = 0
        for q in current_q:
            critic_loss = critic_loss + jnp.mean(jnp.square(q - target_q_values))
        critic_loss = critic_loss / len(current_q) + alpha_coef * conservative_loss

        return critic_loss, {'critic_loss': critic_loss, 'current_q': current_q.mean(), 'conservative_loss': conservative_loss}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def target_update(critic: Model, critic_target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic_target.params)
    return critic_target.replace(params=new_target_params)

def param_clip(log_alpha_coef: Model, a_max: float) -> Model:
    new_log_alpha_params = jax.tree_multimap(lambda p: jnp.clip(p, a_max=jnp.log(a_max)), log_alpha_coef.params)
    return log_alpha_coef.replace(params=new_log_alpha_params)

@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update',
                                             'alpha_update', 'conservative_weight', 'lagrange_thresh'))
def _update_jit(
    rng: int, actor: Model, critic: Model, critic_target: Model, log_ent_coef: Model, log_alpha_coef: Model, replay_data: ReplayBufferSamples,
        gamma: float, target_entropy: float, tau: float, target_update_cond: bool, entropy_update: bool, alpha_update: bool,
        conservative_weight:float, lagrange_thresh:float, task_id: jnp.array,
) -> Tuple[int, Model, Model, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = sac_critic_update(key, actor, critic, critic_target, log_ent_coef, log_alpha_coef, replay_data,
                                                gamma, conservative_weight, lagrange_thresh, task_id)
    if target_update_cond:
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, key = jax.random.split(rng)
    new_actor, actor_info = sac_actor_update(key, actor, new_critic, log_ent_coef, replay_data, task_id)
    rng, key = jax.random.split(rng)
    if entropy_update:
        new_temp, ent_info = log_ent_coef_update(key, log_ent_coef, new_actor, target_entropy, replay_data, task_id)
    else:
        new_temp, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    if alpha_update:
        new_alpha, alpha_info = log_alpha_update(log_alpha_coef, critic_info['conservative_loss'])
    else:
        new_alpha, alpha_info = log_alpha_coef, {'alpha_coef': jnp.exp(log_alpha_coef()), 'alpha_coef_loss': 0}

    return rng, new_actor, new_critic, new_critic_target, new_temp, new_alpha, {**critic_info, **actor_info, **ent_info, **alpha_info}


class MTCQL(OffPolicyAlgorithm):
    """
    Conservative Q Learning (CQL)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = TaskDictReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        num_tasks: int = 10,
        # Add for CQL
        alpha_coef: float = "auto",
        lagrange_thresh: int = 10.0,
        without_exploration: bool = True,
        conservative_weight: float = 10.0,
    ):
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = dict()
        replay_buffer_kwargs['num_tasks'] = num_tasks

        super(MTCQL, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
            without_exploration=without_exploration,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.log_alpha_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.entropy_update = True
        self.alpha_update = True
        self.lagrange_thresh = lagrange_thresh
        self.alpha_coef = alpha_coef
        self.conservative_weight = conservative_weight

        self.num_tasks = num_tasks

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(MTCQL, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            task_id = jax.nn.one_hot(jnp.repeat(jnp.arange(0, self.num_tasks), 1), self.num_tasks)
            log_ent_coef_def = LogEntropyCoef(init_value, self.num_tasks)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key, task_id],
                                             tx=optax.adam(learning_rate=self.lr_schedule(1)))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(self.ent_coef)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            self.entropy_update = False

        if isinstance(self.alpha_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of alpha_coef when learned
            init_value = 1.0
            log_alpha_coef_def = LogAlphaCoef(init_value)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_alpha_coef = Model.create(log_alpha_coef_def, inputs=[temp_key],
                                             tx=optax.adam(learning_rate=self.lr_schedule(1)))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            log_alpha_coef_def = LogAlphaCoef(self.alpha_coef)
            self.key, temp_key = jax.random.split(self.key, 2)
            self.log_alpha_coef = Model.create(log_alpha_coef_def, inputs=[temp_key])
            self.alpha_update = False

        self.task_id = jax.nn.one_hot(jnp.repeat(jnp.arange(0, self.num_tasks), self.batch_size//self.num_tasks), self.num_tasks)


    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        alpha_coef_losses, alpha_coefs, conservative_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.key, key = jax.random.split(self.key, 2)

            target_update_cond = gradient_step % self.target_update_interval == 0
            self.key, new_actor, new_critic, new_critic_target, new_log_ent_coef, new_log_alpha_coef, info \
                = _update_jit(key, self.actor, self.critic, self.critic_target, self.log_ent_coef, self.log_alpha_coef,
                              replay_data, self.gamma, self.target_entropy, self.tau, target_update_cond,
                              self.entropy_update, self.alpha_update, self.conservative_weight, self.lagrange_thresh, self.task_id)

            ent_coef_losses.append(info['ent_coef_loss'])
            ent_coefs.append(info['ent_coef'])
            critic_losses.append(info['critic_loss'])
            actor_losses.append(info['actor_loss'])
            alpha_coefs.append(info['alpha_coef'])
            alpha_coef_losses.append(info['alpha_coef_loss'])
            conservative_losses.append(info['conservative_loss'])

            self.policy.actor = new_actor
            self.policy.critic = new_critic
            self.policy.critic_target = new_critic_target
            self.log_ent_coef = new_log_ent_coef
            self.log_alpha_coef = new_log_alpha_coef

            self._create_aliases()

        # self.replay_buffer._reload_task_latents()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/alpha_coef", np.mean(alpha_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/conservative_loss", np.mean(conservative_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/alpha_coef_loss", np.mean(alpha_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1000,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(MTCQL, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(MTCQL, self)._excluded_save_params() + ["actor", "critic", "critic_target", "log_ent_coef", "log_alpha_coef"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['log_ent_coef'] = self.log_ent_coef.params
        params_dict['log_alpha_coef'] = self.log_alpha_coef.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'log_ent_coef', 'log_alpha_coef']