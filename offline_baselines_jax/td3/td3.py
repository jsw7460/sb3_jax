from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import functools

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, InfoDict, ReplayBufferSamples, Params
from offline_baselines_jax.td3.policies import TD3Policy

def td3_critic_update(key:Any, critic: Model, critic_target: Model, actor_target: Model,
                      replay_data:ReplayBufferSamples, gamma:float, target_policy_noise: float, target_noise_clip: float):

    # Select action according to policy and add clipped noise
    noise = jax.random.normal(key) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = jnp.clip((actor_target(replay_data.next_observations) + noise), -1, 1)

    # Compute the next Q-values: min over all critics targets
    next_q_values = critic_target(replay_data.next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=0)
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        current_q = critic.apply_fn({'params': critic_params}, replay_data.observations, replay_data.actions)

        critic_loss = 0
        # Compute critic loss
        for q in current_q:
            critic_loss = critic_loss + jnp.mean(jnp.square(q - target_q_values))
        critic_loss = critic_loss / len(current_q)
        return critic_loss, {'critic_loss': critic_loss, 'current_q': current_q.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def td3_actor_update(actor: Model, critic: Model, replay_data:ReplayBufferSamples, alpha: float, without_exploration: bool):
    if without_exploration:
        actions_pi = actor(replay_data.observations)
        q1 = critic(replay_data.observations, actions_pi)[0]
        coef_lambda = alpha / (jnp.mean(jnp.abs(q1)))
    else:
        coef_lambda = 1

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Compute actor loss
        actions_pi = actor.apply_fn({'params': actor_params}, replay_data.observations)
        q_value = critic(replay_data.observations, actions_pi)[0].mean()

        actor_loss = - q_value
        if without_exploration:
            bc_loss = jnp.mean(jnp.square(actions_pi - replay_data.actions))
            actor_loss = coef_lambda * actor_loss + bc_loss

        return actor_loss, {'actor_loss': actor_loss, 'q_value': q_value, 'coef_lambda': coef_lambda}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def target_update(model: Model, target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), model.params, target.params)
    return target.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=('gamma', 'tau', 'target_policy_noise', 'target_noise_clip', 'alpha',
                                             'without_exploration', 'actor_update_cond'))
def _update_jit(rng: int, actor: Model, critic: Model, actor_target: Model, critic_target: Model, replay_data: ReplayBufferSamples,
                actor_update_cond: bool, tau: float, target_policy_noise: float, target_noise_clip: float, gamma: float, alpha: float, without_exploration: bool,
                ) -> Tuple[int, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = td3_critic_update(key, critic, critic_target, actor_target, replay_data, gamma, target_policy_noise, target_noise_clip)

    if actor_update_cond:
        new_actor, actor_info = td3_actor_update(actor, new_critic, replay_data, alpha, without_exploration)
        new_actor_target = target_update(new_actor, actor_target, tau)
        new_critic_target = target_update(new_critic, critic_target, tau)
    else:
        new_actor, actor_info = actor, {'actor_loss': 0, 'q_value': 0, 'coef_lambda': 1}
        new_actor_target = actor_target
        new_critic_target = critic_target

    return rng, new_actor, new_critic, new_actor_target, new_critic_target, {**critic_info, **actor_info}

class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, 'episode'),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        alpha: int = 2.5,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
    ):

        super(TD3, self).__init__(
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
            action_noise=action_noise,
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
        if without_exploration and gradient_steps == -1:
            self.gradient_steps = policy_delay

        self.alpha = alpha
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        actor_losses, critic_losses, coef_lambda = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.key, key = jax.random.split(self.key, 2)

            actor_update_cond = self._n_updates % self.policy_delay == 0

            self.key, new_actor, new_critic, new_actor_target, new_critic_target, info = \
                _update_jit(key, self.actor, self.critic, self.actor_target, self.critic_target, replay_data,
                            actor_update_cond, self.tau, self.target_policy_noise, self.target_noise_clip, self.gamma, self.alpha, self.without_exploration)

            self.policy.actor = new_actor
            self.policy.critic = new_critic
            self.policy.actor_target = new_actor_target
            self.policy.critic_target = new_critic_target

            self._create_aliases()
            actor_losses.append(info['actor_loss'])
            critic_losses.append(info['critic_loss'])
            coef_lambda.append(info['coef_lambda'])


        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/coef", np.mean(coef_lambda))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(TD3, self).learn(
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
        return super(TD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['actor_target'] = self.actor_target.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'actor_target']

    def _load_policy(self) -> None:
        super(TD3, self)._load_policy()
        self.policy.actor_target = self.actor_target
