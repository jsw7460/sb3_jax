from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from offline_baselines_jax.common.buffers import ReplayBuffer

np.random.seed(0)


class History(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray


Future = History
STFuture = Future
LTFuture = Future


class STermSubtrajBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    history: History
    st_future: STFuture


class STermSubtrajRewardBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    higher_actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    history_observations: np.ndarray
    history_actions: np.ndarray
    st_future_observations: np.ndarray
    st_future_actions: np.ndarray


class StateActionBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray


class ReplayBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    higher_actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class TrajectoryBuffer(BaseBuffer):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        data_path: str = None,
        normalize: bool = True,
        limit: int = -1,
        use_jax: bool = False,
        buffer_size: int = -1,
    ):
        if data_path is not None:
            import pickle
            with open(data_path + ".pkl", "rb") as f:
                expert_dataset = pickle.load(f)

            buffer_size = len(expert_dataset)
            max_traj_len = max([len(traj["observations"]) for traj in expert_dataset])
            self.use_terminal = "terminals" in expert_dataset[0]

        else:
            expert_dataset = None
            max_traj_len = 1_000
            self.use_terminal = True

        super(TrajectoryBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_dim=observation_dim,
            action_dim=action_dim
        )

        self.data_path = data_path
        self.use_jax = use_jax

        self.expert_dataset = expert_dataset
        self.max_traj_len = max_traj_len
        self.normalize = normalize
        self.limit = limit
        self.normalizing_factor = None

        self.observation_traj = None
        self.next_observation_traj = None
        self.action_traj = None
        self.reward_traj = None
        self.terminal_traj = None
        self.traj_lengths = None

        self.pos = None
        self.full = False
        self.timestep_pos = None
        self.reset()

    def normalize_obs(self, observations: jnp.ndarray, normalizing_facotr: float = None):
        if normalizing_facotr is None:
            normalizing_facotr = self.normalizing_factor
        return deepcopy(observations) / normalizing_facotr

    def reset(self):
        self.observation_traj = np.zeros((self.buffer_size, self.max_traj_len, self.observation_dim))
        self.action_traj = np.zeros((self.buffer_size, self.max_traj_len, self.action_dim))
        self.reward_traj = np.zeros((self.buffer_size, self.max_traj_len))
        if self.data_path is not None:
            if self.use_terminal:
                self.terminal_traj = np.zeros((self.buffer_size, self.max_traj_len))
        else:
            self.terminal_traj = np.zeros((self.buffer_size, self.max_traj_len))
        self.traj_lengths = np.ones((self.buffer_size, 1))
        self.pos = 0
        self.timestep_pos = 0

        if self.data_path is None:
            self.next_observation_traj = np.zeros((self.buffer_size, self.max_traj_len, self.observation_dim))
            return

        for traj_idx in range(self.buffer_size):
            traj_data = self.expert_dataset[traj_idx]
            cur_traj_len = len(traj_data["rewards"])
            self.observation_traj[traj_idx, :cur_traj_len, :] = traj_data["observations"].copy()
            self.action_traj[traj_idx, :cur_traj_len, :] = traj_data["actions"].copy()
            self.reward_traj[traj_idx, :cur_traj_len] = traj_data["rewards"].copy()
            if self.use_terminal:
                self.terminal_traj[traj_idx, :cur_traj_len] = traj_data["terminals"].copy()
            self.traj_lengths[traj_idx, ...] = cur_traj_len

        if self.normalize:
            max_obs = np.max(self.observation_traj)
            min_obs = np.min(self.observation_traj)
            self.normalizing_factor = np.max([max_obs, -min_obs])
        else:
            self.normalizing_factor = 1.0

        if self.limit > 0:
            assert self.use_jax, "Landmark설정 할 때만 이걸 사용하라"
            self.observation_traj = self.observation_traj[:self.limit, ...]
            self.action_traj = self.action_traj[:self.limit, ...]

        if self.use_jax:
            self.observation_traj = jnp.array(self.observation_traj)
            self.action_traj = jnp.array(self.action_traj)

        self.observation_traj /= self.normalizing_factor

        self.pos = self.buffer_size
        self.full = True

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_only_final_state(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        # With Gaussian noise
        batch_key, timestep_key, normal_key = jax.random.split(key, 3)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=(self.traj_lengths - 1).squeeze(),
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        # With Gaussian noise
        batch_key, timestep_key, normal_key = jax.random.split(key, 3)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size, ), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size, ),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        # noise = jax.random.normal(normal_key, shape=current_observations.shape) * 3e-4
        # current_observations = current_observations + noise

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2))
    def noise_sample(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        batch_key, timestep_key = jax.random.split(key, 2)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        # batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        # timesteps = np.random.randint(0, self.traj_lengths - 1)[batch_inds].squeeze()

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def _history_sample(self, key: jnp.ndarray, batch_size: int):
        batch_key, timestep_key = jax.random.split(key)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        return batch_inds, timesteps

    def history_sample(self, key, batch_size: int, history_len: int, st_future_len: int = None, k_nn: int = 0):
        batch_inds, timesteps = self._history_sample(key, batch_size)

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        history_observations, history_actions = [], []
        st_future_observations, st_future_actions = [], []

        for timestep, batch in zip(timesteps, batch_inds):
            start_pts = np.max([timestep - history_len, 0])
            hist_obs = self.observation_traj[batch, start_pts: timestep, ...]
            hist_act = self.action_traj[batch, start_pts: timestep, ...]
            cur_hist_len = len(hist_obs)        # 앞쪽에서 timestep이 골라지면, 인자로 받은 history_len보다 짧아진다.

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            history_observations.append(hist_obs)
            history_actions.append(hist_act)

            st_fut_obs = self.observation_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            st_future_observations.append(st_fut_obs)
            st_future_actions.append(st_fut_act)

        history_observations = jnp.vstack(history_observations).reshape(batch_size, history_len, -1)
        history_actions = jnp.vstack(history_actions).reshape(batch_size, history_len, -1)

        st_future_observations = jnp.vstack(st_future_observations).reshape(batch_size, st_future_len, -1)
        st_future_actions = jnp.vstack(st_future_actions).reshape(batch_size, st_future_len, -1)

        return STermSubtrajBufferSample(
            observations=current_observations,
            actions=current_actions,
            history=History(history_observations, history_actions),
            st_future=STFuture(st_future_observations, st_future_actions)
        )

    def o_history_sample(self, batch_size: int, history_len: int, st_future_len: int = None):
        batch_inds = np.random.randint(0, self.pos, size=batch_size)
        timesteps = np.random.randint(0, self.traj_lengths - 1)[batch_inds].squeeze()

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        history_observations = np.zeros((batch_size, history_len, self.observation_dim))
        history_actions = np.zeros((batch_size, history_len, self.action_dim))

        st_future_observations = np.zeros((batch_size, st_future_len, self.observation_dim))
        st_future_actions = np.zeros((batch_size, st_future_len, self.action_dim))

        for idx, batch in enumerate(batch_inds):
            timestep = timesteps[idx]
            start_pts = np.max([timestep - history_len, 0])
            hist_obs = self.observation_traj[batch, start_pts: timestep, ...]
            hist_act = self.action_traj[batch, start_pts: timestep, ...]
            cur_hist_len = len(hist_obs)  # 앞쪽에서 timestep이 골라지면, 인자로 받은 history_len보다 짧아진다.

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            history_observations[idx] = hist_obs
            history_actions[idx] = hist_act

            st_fut_obs = self.observation_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            st_future_observations[idx] = st_fut_obs
            st_future_actions[idx] = st_fut_act

        return STermSubtrajBufferSample(
            observations=current_observations,
            actions=current_actions,
            history=History(history_observations, history_actions),
            st_future=STFuture(st_future_observations, st_future_actions)
        )

    def history_reward_sample(self, batch_size: int, history_len: int, st_future_len: int = None):
        batch_current_observations = []
        batch_current_actions = []
        batch_current_rewards = []
        batch_current_dones = []

        batch_history_observations = []
        batch_history_actions = []

        batch_st_future_observations = []
        batch_st_future_actions = []

        for batch in range(batch_size):
            batch_ind = np.random.randint(0, self.pos + 1)
            timestep = np.random.randint(0, self.traj_lengths[batch_ind]).squeeze()

            current_observation = self.observation_traj[batch_ind, timestep, :]
            current_action = self.action_traj[batch_ind, timestep, :]
            current_reward = self.reward_traj[batch_ind, timestep]
            current_done = self.terminal_traj[batch_ind, timestep]

            batch_current_observations.append(current_observation)
            batch_current_actions.append(current_action)
            batch_current_rewards.append(current_reward)
            batch_current_dones.append(current_done)

            start_pts = np.max([timestep - history_len, 0])
            hist_obs = self.observation_traj[batch_ind, start_pts: timestep, ...]
            hist_act = self.action_traj[batch_ind, start_pts: timestep, ...]
            cur_hist_len = len(hist_obs)

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            batch_history_observations.append(hist_obs)
            batch_history_actions.append(hist_act)

            st_fut_obs = self.observation_traj[batch_ind, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch_ind, timestep + 1: timestep + 1 + st_future_len, ...]

            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            batch_st_future_observations.append(st_fut_obs)
            batch_st_future_actions.append(st_fut_act)

        current_observations = np.vstack(batch_current_observations)
        current_actions = np.vstack(batch_current_actions)
        current_rewards = np.array(batch_current_rewards)
        current_dones = np.array(batch_current_dones)

        history_observations = np.array(batch_history_observations)
        history_actions = np.array(batch_history_actions)

        st_future_observations = np.array(batch_st_future_observations)
        st_future_actions = np.array(batch_st_future_actions)

        return STermSubtrajRewardBufferSample(
            observations=current_observations,
            actions=current_actions,
            next_observations=st_future_observations[:, 0, :],
            rewards=current_rewards,
            dones=current_dones,
            history=History(history_observations, history_actions),
            st_future=STFuture(st_future_observations, st_future_actions)
        )

    @staticmethod
    def timestep_marking(
        x: jnp.ndarray,
        backward: int,
    ) -> jnp.ndarray:
        """
        History: [batch_size, len_subtraj, obs_dim + action_dim]
        Future: [batch_size, len_subtraj, obs_dim + action_dim]
        Future may be none, especially when evaluation.

        NOTE: History, Future 표현 방식 바꾸려면 여기 바꿔야 함
        Here, we add additional information that the trajectory is whether "history" or "future"

        For history --> -1, -2, -3, ...
        For future --> +1, +2, +3, ...
        """
        # assert x.ndim == 3, "x should have a shape [batch, len_subtraj, dim]"
        batch_size, len_subtraj, _ = x.shape
        marker = jnp.arange(0, len_subtraj)[None, ...] / len_subtraj
        for _ in range(backward):
            marker = jnp.flip(marker, axis=1)
        marker = jnp.repeat(marker, repeats=batch_size, axis=0)[..., None]
        x = jnp.concatenate((x, marker), axis=2)

        return x

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
        infos: List[Dict[str, Any]]
    ) -> None:
        self.observation_traj[self.pos, self.timestep_pos] = deepcopy(obs)
        self.action_traj[self.pos, self.timestep_pos] = deepcopy(action)
        self.next_observation_traj[self.pos, self.timestep_pos] = deepcopy(next_obs)
        self.reward_traj[self.pos, self.timestep_pos] = deepcopy(reward)
        self.terminal_traj[self.pos, self.timestep_pos] = deepcopy(done)

        self.traj_lengths[self.pos] = (self.timestep_pos + 1)

        if done:
            self.pos += 1
            self.timestep_pos = 0
        else:
            self.timestep_pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


class FinetuneReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observations_space: spaces.Space,
        lower_action_space: spaces.Space,
        higher_action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(FinetuneReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observations_space,
            action_space=lower_action_space,
            n_envs=n_envs
        )
        self.higher_action_space = higher_action_space
        self.higher_action_dim = higher_action_space.shape[0]
        self.higher_actions = np.zeros((self.buffer_size, self.n_envs, self.higher_action_dim))

    def add_traj(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            higher_action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.higher_actions[self.pos] = np.array(higher_action).copy()

        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.higher_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(data))

    def metla_sample(self, batch_size: int, history_len: int, future_len: int) -> STermSubtrajRewardBufferSample:
        assert self.n_envs == 1
        # upper_bound = self.buffer_size if self.full else self.pos
        # batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        upper_bound = self.pos + (self.buffer_size - self.pos) * self.full
        # batch_inds = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=upper_bound)
        batch_inds = np.random.randint(0, upper_bound, size=batch_size, )

        batch_current_observations = self.observations[batch_inds, 0, :]
        batch_current_actions = self.actions[batch_inds, 0, :]
        batch_current_higher_actions = self.higher_actions[batch_inds, 0, :]
        batch_next_observations = self.next_observations[batch_inds, 0, :]
        batch_rewards = self.rewards[batch_inds, ...]
        batch_dones = (self.dones[batch_inds, 0] * (1 - self.timeouts[batch_inds, 0])).reshape(-1, 1)

        batch_history_observations = []
        batch_history_actions = []

        batch_st_future_observations = []
        batch_st_future_actions = []
        for i, batch_ind in enumerate(batch_inds):  # history, future를 처리하라
            dones = self.dones.copy()
            history_dones = dones.copy()
            history_dones[batch_ind:, ...] = 0
            history_done_timestep, _ = np.nonzero(history_dones)

            # 앞쪽으로 done이 일어난 가장 최근 부분
            latest_done_history = -1 if len(history_done_timestep) == 0 else history_done_timestep[-1]
            start_pts = np.max(np.array([latest_done_history + 1, batch_ind - history_len]))

            hist_obs = np.squeeze(self.observations[start_pts: batch_ind, ...], axis=1)
            hist_act = np.squeeze(self.actions[start_pts: batch_ind, ...], axis=1)
            cur_hist_len = len(hist_obs)

            # Chceck whether ther is further dones before pos comeback to the initial point
            if latest_done_history == -1 and self.full:
                dones_copy = self.dones.copy()
                dones_copy[: self.buffer_size - (history_len - cur_hist_len)] = 0
                latest_done_history, _ = np.nonzero(dones_copy)

                start_pts = self.buffer_size if len(latest_done_history) == 0 else np.max(latest_done_history)

                additional_hist_obs = np.squeeze(self.observations[start_pts: self.buffer_size, ...], axis=1)
                additional_hist_act = np.squeeze(self.actions[start_pts: self.buffer_size, ...], axis=1)
                hist_obs = np.vstack((additional_hist_obs, hist_obs))
                hist_act = np.vstack((additional_hist_act, hist_act))
                cur_hist_len = len(hist_obs)

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            batch_history_observations.append(hist_obs)
            batch_history_actions.append(hist_act)

            future_dones = dones.copy()
            future_dones[:batch_ind + 1, ...] = 0
            future_done_timestep, _ = np.nonzero(future_dones)

            if len(future_done_timestep) == 0:
                latest_done_future = self.pos if not self.full else self.buffer_size
            else:
                latest_done_future = future_done_timestep[0]
            end_pts = np.min([latest_done_future, batch_ind + future_len])
            st_fut_obs = np.squeeze(self.observations[batch_ind + 1: end_pts + 1, ...], axis=1)
            st_fut_act = np.squeeze(self.actions[batch_ind + 1: end_pts + 1, ...], axis=1)

            cur_future_len = len(st_fut_obs)

            # Check wheter there is further dones after pos comeback to the initial point
            if latest_done_future == self.buffer_size and self.full:
                dones_copy = self.dones.copy()
                dones_copy[future_len - cur_future_len: ] = 0
                latest_done_future, _ = np.nonzero(dones_copy)
                end_pts = 0 if len(latest_done_future) == 0 else np.min(latest_done_future)
                additional_fut_obs = np.squeeze(self.observations[0: end_pts, ...], axis=1)
                additional_fut_act = np.squeeze(self.actions[0: end_pts, ...], axis=1)
                st_fut_obs = np.vstack((st_fut_obs, additional_fut_obs))
                st_fut_act = np.vstack((st_fut_act, additional_fut_act))
                cur_future_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((future_len - cur_future_len, self.observation_dim))
            st_fut_padding_act = np.zeros((future_len - cur_future_len, self.action_dim))

            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            batch_st_future_observations.append(st_fut_obs)
            batch_st_future_actions.append(st_fut_act)

        history_observations = np.array(batch_history_observations)
        history_actions = np.array(batch_history_actions)

        st_future_observations = np.array(batch_st_future_observations)
        st_future_actions = np.array(batch_st_future_actions)

        return STermSubtrajRewardBufferSample(
            observations=batch_current_observations,
            actions=batch_current_actions,
            higher_actions=batch_current_higher_actions,
            next_observations=batch_next_observations,
            rewards=batch_rewards,
            dones=batch_dones,
            history_observations=history_observations,
            history_actions=history_actions,
            st_future_observations=st_future_observations,
            st_future_actions=st_future_actions
        )