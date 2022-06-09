import pickle
from typing import NamedTuple

import numpy as np


class GoalGeneratorBufferSample(NamedTuple):
    observations: np.ndarray
    subgoal_observations: np.ndarray
    goal_observations: np.ndarray
    target_future_hop: np.ndarray


class CondVaeGoalGeneratorBuffer(object):
    # n_traj: int
    # buffer_size: int
    #
    # observations: np.ndarray
    # traj_lengths: np.ndarray
    # subgoal_cand: np.ndarray
    # observation_traj: np.ndarray
    # goal_pos: np.ndarray
    # max_traj_len: np.ndarray

    def __init__(self, data_path: str, n_subgoal: int = 20):
        with open(data_path + ".pkl", "rb") as f:
            self.dataset = pickle.load(f)

        self.n_traj = None
        self.buffer_size = None
        self.observations = None
        self.traj_lengths = None
        self.subgoal_cand = None
        self.observation_traj = None
        self.goal_pos = None
        self.max_traj_len = None
        self.mean_target_future_hop = None

        self.current_trunc = 0
        self.n_subgoal = n_subgoal

        self.truncation_reset(0)

    def truncation_reset(self, current_trunc: int):
        """
        :param current_trunc: 데이터가 커서, 전체 dataset을 truncation해서 사용해야 한다.
        그 truncation이 몇 번째인지 알려주는 수가 current_trunc이다.
        :return:
        """
        dataset = self.dataset[500 * current_trunc: 500 * (current_trunc + 1)]

        self.n_traj = len(dataset)
        observations = [data["observations"].copy() for data in dataset]
        traj_lengths = [len(observation) for observation in observations]
        subgoal_cand = [np.floor(np.linspace(0, traj_len - 1, self.n_subgoal)) for traj_len in traj_lengths]

        self.traj_lengths = np.array(traj_lengths)  # [5166, ]
        self.mean_target_future_hop = np.floor(np.mean(self.traj_lengths / self.n_subgoal))     # 평균 몇 step이후의 goal을 예측하는지
        self.subgoal_cand = np.array(subgoal_cand, dtype=np.int32)  # [5166, 20]
        self.buffer_size = len(self.traj_lengths)

        obs_sample = observations[0][0]
        max_traj_len = np.max(self.traj_lengths)
        self.observation_traj = np.zeros(([self.n_traj, max_traj_len] + [*obs_sample.shape]))  # [5166, 501, 32, 32, 3]
        self.goal_pos = np.zeros((self.n_traj, 2))  # [5166, 2]

        for traj_idx, traj_len in zip(range(self.n_traj), self.traj_lengths):
            self.observation_traj[traj_idx, :traj_len, ...] = dataset[traj_idx]["observations"].copy()
            # Goal is fixed during episode. So we just need 0th goal.
            self.goal_pos[traj_idx] = dataset[traj_idx]["goals"][0].copy()

        self.max_traj_len = max_traj_len

    def sample(self, batch_size: int = 256) -> GoalGeneratorBufferSample:
        batch_inds = np.random.randint(0, self.n_traj, size=batch_size)
        timesteps = np.random.randint(0, self.traj_lengths)[batch_inds]
        subgoal_inds = self.subgoal_cand[batch_inds]
        subgoal_inds[timesteps[:, np.newaxis] > subgoal_inds] = 99999
        subgoal_inds = np.min(subgoal_inds, axis=1)

        # If timestep == 0, then subgoal index is zero. So we insert 1 for such indices.
        subgoal_inds[subgoal_inds == 0] = 1

        observation = self.observation_traj[batch_inds, timesteps, ...].copy()
        subgoal_observation = self.observation_traj[batch_inds, subgoal_inds, ...].copy()
        goal_observation = self.goal_pos[batch_inds].copy()
        target_future_hop = (subgoal_inds - timesteps)[..., np.newaxis]

        data = (observation / 255, subgoal_observation / 255, goal_observation, target_future_hop)

        return GoalGeneratorBufferSample(*data)


class SensorBasedExpertBuffer(object):
    def __init__(
        self,
        data_path: str,
        n_frames: int = 5
    ):
        with open(data_path + ".pkl", "rb") as f:
            dataset = pickle.load(f)

        n_traj = len(dataset)
        observations = [data["observations"].copy() for data in dataset]
        actions = [data["actions"].copy() for data in dataset]
        traj_lengths = [len(obs) for obs in observations]

        obs_sample = observations[0][0]
        act_sample = actions[0][0]
        max_traj_len = np.max(traj_lengths) + n_frames

        # +1: Relabel action 할 때, 문제있는 부분이 생김
        self.observation_traj = np.zeros(([n_traj, max_traj_len + 1] + [*obs_sample.shape]))
        self.action_traj = np.zeros(([n_traj, max_traj_len + 1] + [*act_sample.shape]))
        self.relabled_action_traj = None  # It will be labled later

        self.observation_dim = obs_sample.shape[-1]
        self.n_frames = n_frames

        # Save goal observation if exist
        self.goal_traj = None
        if dataset[0].get("goals", None) is not None:
            goal_sample = dataset[0].get("goals")[0]
            self.goal_traj = np.zeros(([n_traj, max_traj_len + 1] + [*goal_sample.shape]))

        # Save all the data
        for traj_idx, traj_len in zip(range(n_traj), traj_lengths):
            # The environment observation is temporally stacked
            original_obs = dataset[traj_idx]["observations"]
            original_act = dataset[traj_idx]["actions"]

            # 첫 번째 observation은 history가 없기 때문에, n_frames-1 만큼 늘려준다. 첫번째는 이미 들어있기 때문에 -1 해준다.
            frame_aug_obs = np.repeat(dataset[traj_idx]["observations"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
            frame_aug_obs = np.vstack((frame_aug_obs, original_obs))

            # Action은 stacking 안해주지만, 그래도 dataset에서는 augment한다. 안그러면 sampling할 때 index가 안맞는다.
            frame_aug_act = np.repeat(dataset[traj_idx]["actions"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
            frame_aug_act = np.vstack((frame_aug_act, original_act))

            self.observation_traj[traj_idx, :traj_len + n_frames - 1] = frame_aug_obs.copy()
            self.action_traj[traj_idx, :traj_len + n_frames - 1] = frame_aug_act.copy()

            if self.goal_traj is not None:
                original_goal = dataset[traj_idx]["goals"]
                frame_aug_goals = np.repeat(dataset[traj_idx]["goals"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
                frame_aug_goals = np.vstack((frame_aug_goals, original_goal))
                self.goal_traj[traj_idx, :traj_len + n_frames - 1] = frame_aug_goals.copy()

        self.n_traj = n_traj
        self.traj_lengths = traj_lengths
        self.max_traj_len = max_traj_len

        self.lower_action_dim = self.observation_dim        # type: int

    def relabel_action_by_obs_difference(self) -> None:
        action_traj = np.abs(self.observation_traj[:, 1:, ...] - self.observation_traj[:, :-1, ...])

        # State가 끝나는 부분에서 state간의 차이를 action으로 정의하면, 하나의 수치가 비정상적으로 커진다. 그래서 강제로 0으로 만든다.
        action_traj[np.arange(self.n_traj), np.array(self.traj_lengths) + self.n_frames - 1, ...] = 0

        last_action = np.zeros((self.n_traj, 1, self.observation_dim))
        action_traj = np.concatenate((action_traj, last_action), axis=1)            # [n_traj, max_traj_len, 4]

        max_act = action_traj.max(axis=1, keepdims=True).max(axis=0, keepdims=True)
        min_act = action_traj.min(axis=1, keepdims=True).max(axis=0, keepdims=True)

        action_traj = (action_traj - min_act) / (max_act - min_act)

        self.lower_action_dim = action_traj.shape[-1]
        self.relabled_action_traj = action_traj

    def sample(self, batch_size: int = 256, relabled_action: bool = True):
        batch_inds = np.random.randint(0, self.n_traj, size=batch_size)
        timesteps = np.random.randint(self.n_frames - 1, np.array(self.traj_lengths)-1)     # [n_traj, ]

        # For next timestep
        # +2 까지 해야 실제 arange는 +1 까지 됨. 뒤에서 obs, next_obs로 나뉠 것이므로 일부러 하나 더 timestep을 만들어주는 중
        idxs = np.array([np.arange(timestep - self.n_frames + 1, timestep + 2) for timestep in timesteps])
        observations_chunk = self.observation_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds].copy()
        observations = observations_chunk[:, : -1, ...]
        next_observations = observations_chunk[:, 1:, ...]

        if self.goal_traj is not None:
            goals_chunk = self.goal_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds].copy()
            goals = goals_chunk[:, : -1, ...]
            next_goals = goals_chunk[:, 1:, ...]

            observations = {
                "observations": observations,
                "goals": goals
            }

            next_observations = {
                "observations": next_observations,
                "goals": next_goals
            }

        action_traj_buffer = self.relabled_action_traj if relabled_action else self.action_traj
        actions = action_traj_buffer[np.arange(self.n_traj), timesteps, ...][batch_inds].copy()

        assert np.mean(np.isnan(observations)) == 0
        assert np.mean(np.isnan(actions)) == 0

        return observations, actions, next_observations
