import numpy as np
import gym
import navigation_2d

from offline_baselines_jax import MTSAC, SoftModularization
from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy
from offline_baselines_jax.soft_modularization.policies import SoftModulePolicy
from offline_baselines_jax.common.callbacks import EvalCallback

class dummy_task_env(gym.Env):
    def __init__(self, envs):
        self.envs = envs
        self.env = None
        self._task_idx = -1
        self.observation_space = gym.spaces.Dict({'obs': self.envs[0].observation_space,
                                                  'task': gym.spaces.Box(low=0, high=1, shape=(len(self.envs), ))})
        self.action_space = self.envs[0].action_space

    def step(self, action):
        task_onehot = np.zeros(len(self.envs))
        task_onehot[self._task_idx] = 1

        obs, reward, done, info = self.env.step(action)
        info['task'] = self._task_idx
        info['task_name'] = 'goal_{}'.format(self._task_idx)
        if 'is_success' in info.keys():
            info['success'] = info['is_success']
        return {'obs': obs, 'task': task_onehot}, reward, done, info

    def reset(self):
        self._task_idx = (self._task_idx + 1) % len(self.envs)
        self.env = self.envs[self._task_idx]

        obs = self.env.reset()
        task_onehot = np.zeros(len(self.envs))
        task_onehot[self._task_idx] = 1
        return {'obs': obs, 'task': task_onehot}


class dummy_task_routing(gym.Env):
    def __init__(self, env, num_resource, num_layer, model):
        self.env = env
        self.num_resource = num_resource
        self.num_layer = num_layer
        self.obs = None
        self.model = model
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(12, ))

    def step(self, action):
        action = action.reshape((3, 4))
        ind = np.argsort(action, axis=1)
        action = []
        for idx, a in enumerate(ind):
            act = np.where(a < self.num_resource[idx], 1, 0)
            action.append(act)
        kwargs = {'module_select': action}

        real_action = self.model.predict(self.obs, **kwargs)[0][0]
        self.obs, reward, done, info = self.env.step(real_action)
        return self.obs, reward, done, info

    def reset(self):
        self.obs =  self.env.reset()
        return self.obs

if __name__ == '__main__':
    envs = [gym.make('Navi-Acc-Lidar-Obs-Task{}_very_hard-v0'.format(i)) for i in range(8)]
    env = dummy_task_env(envs)

    test_env = dummy_task_env(envs)
    # policy_kwargs = {'net_arch':[64] }
    # model = SoftModularization(SoftModulePolicy, env, seed=777, verbose=1, batch_size=1024, num_tasks=8, buffer_size=50000, train_freq=1600, gradient_steps=200, policy_kwargs=policy_kwargs,
    #                            tensorboard_log='../SF/SF', learning_rate=3e-5)
    # model.learn(total_timesteps=1000000, log_interval=50, eval_freq=100000, n_eval_episodes=80, eval_log_path='../SF/SF', eval_env=test_env)
    #
    # model.save('./model.zip')
    model = SoftModularization.load('./model.zip')
    # state = test_env.reset()
    # kwargs = {'module_select': [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    # print(model.predict(state, **kwargs))
    #
    # kwargs = {'module_select': [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]}
    # print(model.predict(state, **kwargs))
    #
    # kwargs = {'module_select': [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
    # print(model.predict(state, **kwargs))

    policy_kwargs = {'net_arch': [64, 64]}
    routing_env = dummy_task_routing(env, [3, 3, 3], 3, model)
    routing_test_env = dummy_task_routing(test_env, [3, 3, 3], 3, model)

    routing_model = MTSAC(MultiInputPolicy, routing_env, seed=777, num_tasks=8, batch_size=1024, verbose=1,  policy_kwargs=policy_kwargs, tensorboard_log='../SF/Route')
    routing_model.learn(total_timesteps=500000, log_interval=50, eval_freq=50000, n_eval_episodes=80, eval_log_path='../SF/Route', eval_env=routing_test_env)
