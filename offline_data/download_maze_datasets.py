import gym
import numpy as np

import collections
import pickle

from os import listdir
from os.path import isfile, join
from os import walk
import h5py

import d4rl_maze

if __name__ == "__main__":
	for b in range(0, 4):
		# mypath = f"/workspace/expertdata/maze/batch_{b}/"
		mypath = f"/workspace/spirl/d4rl/maze2d-0.0/batch_{b}/"
		name=f"Maze2d_img_{b}"
		onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

		rollouts = []
		for (dirpath, dirnames, filenames) in walk(mypath):
			rollouts.extend(filenames)

		observations = []
		next_observations = []
		goals = []
		actions = []
		rewards = []
		terminals = []

		u = 0
		for i, rollout in enumerate(rollouts):
			print("II", i)
			file = mypath + rollout
			with h5py.File(file, "r") as f:
				observation = f["traj0"]["images"].value
				action = f["traj0"]["actions"].value
				terminal = 1 - f["traj0"]["pad_mask"].value[:, np.newaxis]
				goal = f["traj0"]["goals"].value

				observations.append(observation)
				goals.append(goal)
				actions.append(action)
				terminals.append(terminal)

		observations = np.vstack(observations)
		goals = np.vstack(goals)
		actions = np.vstack(actions)
		terminals = np.vstack(terminals).squeeze()

		next_observations = np.zeros_like(observations)
		rewards = np.zeros_like(terminals)

		dataset = {
			"observations": observations,
			"next_observations": next_observations,
			"goals": goals,
			"actions": actions,
			"rewards": rewards,
			"terminals": terminals
		}

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000 - 1)
			for k in ['observations', 'next_observations', 'goals', 'actions', 'rewards', 'terminals']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(
			f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)
