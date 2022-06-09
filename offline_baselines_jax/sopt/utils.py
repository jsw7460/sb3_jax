from collections import deque
from typing import List, Dict, Any, NamedTuple

import gym
import numpy as np

from d4rl_maze.pointmaze import maze_model, maze_layouts
from offline_baselines_jax.common.buffers import ReplayBuffer


def get_env(cfg):
    layout_str = maze_layouts.rand_layout(0, size=cfg.rand_maze_size, coverage_frac=cfg.coverage_frac)
    env = maze_model.MazeEnv(
        layout_str,
        agent_centric_view=cfg.agent_centric,
        img_state=cfg.img_based,
        reset_target=True,
        reward_type=cfg.reward_type
    )
    # env = maze_model.ImgStateDictMazeEnv(
    #     layout_str,
    #     agent_centric_view=cfg.agent_centric,
    #     img_state=cfg.img_based,
    #     img_height=cfg.img_shape,
    #     img_width=cfg.img_shape,
    #     reset_target=True,
    #     reward_type=cfg.reward_type
    # )
    return env


class GoalMDPSensorObservationStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int):
        super(GoalMDPSensorObservationStackWrapper, self).__init__(env)
        from offline_baselines_jax.common.preprocessing import is_image_space
        assert not is_image_space(env.observation_space)
        self.env = env
        self._n_frames = n_frames
        self.obs_frames = deque(maxlen=n_frames)
        self.goal_frames = deque(maxlen=n_frames)

        dummy_obs = self.reset()

        self.observation_space = gym.spaces.Dict({
            "observations": gym.spaces.Box(-np.inf, np.inf, shape=dummy_obs["observations"].shape),
            "goals": gym.spaces.Box(-np.inf, np.inf, shape=dummy_obs["goals"].shape)
        })

    def reset(self, **kwargs):
        obs = self.env.reset()
        goals = self.env.get_target()
        for _ in range(self._n_frames):
            self.obs_frames.append(obs.copy())
            self.goal_frames.append(goals.copy())
        _obs = np.array(self.obs_frames).copy()
        _goals = np.array(self.goal_frames).copy()

        observation = {
            "observations": _obs,
            "goals": _goals
        }
        return observation

    def step(self, action: np.ndarray):
        assert len(self.obs_frames) == self._n_frames
        obs, reward, done, info = self.env.step(action)
        goals = info["goal"]
        self.obs_frames.append(obs.copy())
        self.goal_frames.append(goals.copy())

        _obs = np.array(self.obs_frames).copy()
        _goals = np.array(self.goal_frames).copy()

        observation = {
            "observations": _obs,
            "goals": _goals
        }
        return observation, reward, done, info


class RewardMDPSensorObservationStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int):
        super(RewardMDPSensorObservationStackWrapper, self).__init__(env)
        self.env = env
        self._n_frames = n_frames
        self.obs_frames = deque(maxlen=n_frames)

        dummy_obs = self.reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=dummy_obs.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset()
        for _ in range(self._n_frames):
            self.obs_frames.append(obs.copy())
        observation = np.array(self.obs_frames).copy()
        return observation

    def step(self, action: np.ndarray):
        assert len(self.obs_frames) == self._n_frames
        obs, reward, done, info = self.env.step(action)
        self.obs_frames.append(obs.copy())

        observation = np.array(self.obs_frames).copy()

        return observation, reward, done, info



class PosGoalReplayBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    goals: np.ndarray


class PosGoalReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        optimize_memory_usage: bool = False,
    ):
        super(PosGoalReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=True
        )

        self.goals = np.zeros((self.buffer_size, self.n_envs, 2))        # goals: xy - pos


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

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.goals[self.pos] = np.array([info.get("goal", None) for info in infos]).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env = None) -> PosGoalReplayBufferSample:
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
            self.goals[batch_inds, env_indices]
        )
        return PosGoalReplayBufferSample(*tuple(data))