import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, NamedTuple
from numba import njit, prange
import functools

import numpy as np
# import stable_baselines.acktr.kfac_utils
from gym import spaces

from offline_baselines_jax.common.preprocessing import get_action_dim, get_obs_shape
from offline_baselines_jax.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
)

import jax
import jax.numpy as jnp
from stable_baselines3.common.vec_env import VecNormalize
from copy import deepcopy

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class STermSubtrajRewardBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    history_observations: np.ndarray
    history_actions: np.ndarray
    st_future_observations: np.ndarray
    st_future_actions: np.ndarray


@jax.jit
def normal_sampling(key:Any, task_latents_mu: jnp.ndarray, task_latents_log_std:jnp.ndarray):
    return task_latents_mu + jax.random.normal(key, shape=(task_latents_log_std.shape[-1], )) * jnp.exp(0.5 * task_latents_log_std)


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.observation_dim = None
        try:
            self.observation_dim = self.observation_space.shape[0]
        except:
            pass

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
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
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(data))

    # def _metla_sample(self, batch_size, history_len, future_len):
    #     batch_history_observations = np.zeros((batch_size, history_len, self.observation_dim))
    #     batch_history_actions = np.zeros((batch_size, history_len, self.action_dim))
    #     batch_st_future_observations = np.zeros((batch_size, future_len, self.observation_dim))
    #     batch_st_future_actions = np.zeros((batch_size, future_len, self.action_dim))
    #     upper_bound = self.pos + (self.buffer_size - self.pos) * self.full
    #     batch_inds = np.random.randint(0, upper_bound, size=batch_size, )
    #
    #     batch_current_observations = self.observations[batch_inds, 0, :]
    #     batch_current_actions = self.actions[batch_inds, 0, :]
    #     batch_next_observations = self.next_observations[batch_inds, 0, :]
    #     batch_rewards = self.rewards[batch_inds, ...]
    #     batch_dones = (self.dones[batch_inds, 0] * (1 - self.timeouts[batch_inds, 0])).reshape(-1, 1)
    #
    #     return self.jit_metla_sample(
    #         batch_size=batch_size,
    #         history_len=history_len,
    #         future_len=future_len,
    #         pos=self.pos,
    #         buffer_size=self.buffer_size,
    #         full=self.full,
    #         observations=self.observations,
    #         actions=self.actions,
    #         next_observations=self.next_observations,
    #         rewards=self.rewards,
    #         dones=self.dones,
    #         timeouts=self.timeouts,
    #         observation_dim=self.observation_dim,
    #         action_dim=self.action_dim,
    #         batch_history_observations=batch_history_observations,
    #         batch_history_actions=batch_history_actions,
    #         batch_st_future_observations=batch_st_future_observations,
    #         batch_st_future_actions=batch_st_future_actions,
    #         batch_inds=batch_inds,
    #         upper_bound=upper_bound
    #     )
    #
    # @staticmethod
    # @njit(parallel=True)
    # def jit_metla_sample(batch_size, history_len, future_len, pos, buffer_size, full, observations, actions, next_observations, rewards, dones, timeouts, observation_dim, action_dim,
    #                      batch_history_observations, batch_history_actions, batch_st_future_observations, batch_st_future_actions, batch_inds, upper_bound):
    #     # upper_bound = self.buffer_size if self.full else self.pos
    #     # batch_inds = np.random.randint(0, upper_bound, size=batch_size)
    #     # upper_bound = pos + (buffer_size - pos) * full
    #     # batch_inds = np.random.randint(0, upper_bound, size=batch_size, )
    #     # batch_inds = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=upper_bound)
    #
    #     # batch_history_observations = []
    #     # batch_history_actions = []
    #     #
    #     # batch_st_future_observations = []
    #     # batch_st_future_actions = []
    #     for b in prange(len(batch_inds)):  # history, future를 처리하라
    #         batch_ind = batch_inds[b]
    #         dones = dones.copy()
    #         history_dones = dones.copy()
    #         history_dones[batch_ind:, ...] = 0
    #         history_done_timestep, _ = np.nonzero(history_dones)
    #
    #         # 앞쪽으로 done이 일어난 가장 최근 부분
    #         latest_done_history = -1 if len(history_done_timestep) == 0 else history_done_timestep[-1]
    #         start_pts = np.max(np.array([latest_done_history + 1, batch_ind - history_len]))
    #
    #         hist_obs = observations[start_pts: batch_ind, 0, ...]
    #         hist_act = actions[start_pts: batch_ind, 0, ...]
    #         cur_hist_len = len(hist_obs)
    #
    #         # Chceck whether ther is further dones before pos comeback to the initial point
    #         if latest_done_history == -1 and full:
    #             dones_copy = dones.copy()
    #             dones_copy[: buffer_size - (history_len - cur_hist_len)] = 0
    #             latest_done_history, _ = np.nonzero(dones_copy)
    #
    #             start_pts = buffer_size if len(latest_done_history) == 0 else np.max(latest_done_history)
    #
    #             # additional_hist_obs = np.squeeze(observations[start_pts: buffer_size, ...], axis=1)
    #             # additional_hist_act = np.squeeze(actions[start_pts: buffer_size, ...], axis=1)
    #             additional_hist_obs = observations[start_pts: buffer_size, 0, ...]
    #             additional_hist_act = actions[start_pts: buffer_size, 0, ...]
    #             hist_obs = np.vstack((additional_hist_obs, hist_obs))
    #             hist_act = np.vstack((additional_hist_act, hist_act))
    #             cur_hist_len = len(hist_obs)
    #
    #         hist_padding_obs = np.zeros((history_len - cur_hist_len, observation_dim))
    #         hist_padding_act = np.zeros((history_len - cur_hist_len, action_dim))
    #         hist_obs = np.vstack((hist_padding_obs, hist_obs))
    #         hist_act = np.vstack((hist_padding_act, hist_act))
    #
    #         # batch_history_observations.append(hist_obs)
    #         batch_history_observations[b] = hist_obs
    #         batch_history_actions[b] = hist_act
    #         # batch_history_actions.append(hist_act)
    #
    #         future_dones = dones.copy()
    #         future_dones[:batch_ind + 1, ...] = 0
    #         future_done_timestep, _ = np.nonzero(future_dones)
    #
    #         if len(future_done_timestep) == 0:
    #             latest_done_future = pos if not full else buffer_size
    #         else:
    #             latest_done_future = future_done_timestep[0]
    #         end_pts = np.min(np.array([latest_done_future, batch_ind + future_len]))
    #         st_fut_obs = observations[batch_ind + 1: end_pts + 1, 0, ...]
    #         st_fut_act = actions[batch_ind + 1: end_pts + 1, 0, ...]
    #
    #         cur_future_len = len(st_fut_obs)
    #
    #         # Check wheter there is further dones after pos comeback to the initial point
    #         if latest_done_future == buffer_size and full:
    #             dones_copy = dones.copy()
    #             dones_copy[future_len - cur_future_len: ] = 0
    #             latest_done_future, _ = np.nonzero(dones_copy)
    #             end_pts = 0 if len(latest_done_future) == 0 else np.min(latest_done_future)
    #             additional_fut_obs = observations[0: end_pts, 0, ...]
    #             additional_fut_act = actions[0: end_pts, 0, ...]
    #             st_fut_obs = np.vstack((st_fut_obs, additional_fut_obs))
    #             st_fut_act = np.vstack((st_fut_act, additional_fut_act))
    #             cur_future_len = len(st_fut_obs)
    #
    #         st_fut_padding_obs = np.zeros((future_len - cur_future_len, observation_dim))
    #         st_fut_padding_act = np.zeros((future_len - cur_future_len, action_dim))
    #
    #         st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
    #         st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))
    #
    #         # batch_st_future_observations.append(st_fut_obs)
    #         # batch_st_future_actions.append(st_fut_act)
    #         batch_st_future_observations[b] = st_fut_obs
    #         batch_st_future_actions[b] = st_fut_act
    #
    #     # history_observations = np.array(batch_history_observations)
    #     # history_actions = np.array(batch_history_actions)
    #     # st_future_observations = np.array(batch_st_future_observations)
    #     # st_future_actions = np.array(batch_st_future_actions)
    #     history_observations = batch_history_observations
    #     history_actions = batch_history_actions
    #     st_future_observations = batch_st_future_observations
    #     st_future_actions = batch_st_future_actions
    #
    #     return history_observations, history_actions, st_future_observations, st_future_actions
    #
    #     # return STermSubtrajRewardBufferSample(
    #     #     observations=batch_current_observations,
    #     #     actions=batch_current_actions,
    #     #     next_observations=batch_next_observations,
    #     #     rewards=batch_rewards,
    #     #     dones=batch_dones,
    #     #     history_observations=history_observations,
    #     #     history_actions=history_actions,
    #     #     st_future_observations=st_future_observations,
    #     #     st_future_actions=st_future_actions
    #     # )


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.key = jax.random.PRNGKey(0)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }


        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        task_embeddings: Dict[str, np.ndarray] = None,
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()})

        observations = {key: obs for key, obs in obs_.items()}
        next_observations = {key: obs for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.actions[batch_inds, env_indices],
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices]).reshape(
                -1, 1
            ),
            rewards=self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )


class TaskDictReplayBuffer(object):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            num_tasks: int = 10,
    ):
        self.replay_buffers = []
        self.num_tasks = num_tasks
        for _ in range(num_tasks):
            self.replay_buffers.append(DictReplayBuffer(buffer_size//num_tasks, observation_space, action_space,
                                                        n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
                                                        handle_timeout_termination=handle_timeout_termination,))

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            task_embeddings: Dict[str, np.ndarray] = None,
    ) -> None:
        self.replay_buffers[infos[0]['task']].add(obs, next_obs, action, reward, done, infos, task_embeddings)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        observations = {'task': [], 'obs': []}
        actions = []
        next_observations = {'task': [], 'obs': []}
        rewards = []
        dones = []
        for i in range(self.num_tasks):
            batch = self.replay_buffers[i].sample(batch_size//self.num_tasks, env)
            for key, data in batch.observations.items():
                observations[key].append(data)
            actions.append(batch.actions)
            for key, data in batch.next_observations.items():
                next_observations[key].append(data)
            rewards.append(batch.rewards)
            dones.append(batch.dones)

        for key, data in observations.items():
            observations[key] = np.concatenate(data, axis=0)
        for key, data in next_observations.items():
            next_observations[key] = np.concatenate(data, axis=0)

        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)

        return DictReplayBufferSamples(observations=observations, actions=actions, next_observations=next_observations,
                                       dones=dones, rewards=rewards)
