import gym
import numpy as np

import collections
import pickle

import d4rl

if __name__ == "__main__":
	# with open("halfcheetah-expert-v2.pkl", "rb") as f:
	# 	z = pickle.load(f)
	#
	# for i, item in enumerate(z):
	# 	print(i, type(item))
	# exit()
	datasets = []

	for env_name in ["halfcheetah"]:
		for dataset_type in ["medium"]:
		# for dataset_type in ["human", "cloned", "expert"]:
			name = f'{env_name}-{dataset_type}-v2'
			env = gym.make(name)
			# dataset = env.get_dataset()
			dataset = d4rl.qlearning_dataset(env)
			print(dataset.keys())
			print(dataset["observations"].shape)			# [size, dim]
			print(dataset["next_observations"].shape)		# [size, dim]
			print(dataset["actions"].shape)					# [size, dim]
			print(dataset["rewards"].shape)					# [size, ]
			print(dataset["terminals"].shape)				# [size, ]
			exit()
			# print("data", dataset)
			# exit()
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
				for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
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
