from offline_baselines_jax import SAC
from offline_baselines_jax.sac.policies import SACPolicy

import gym

train_env = gym.make('HalfCheetah-v2')

model = SAC(SACPolicy, train_env, seed=777, verbose=1, batch_size=1024, buffer_size=50000, train_freq=1)

model.learn(total_timesteps=10000)
model.save('./model.zip')
model = SAC.load('./model.zip', train_env)

model.learn(total_timesteps=10000)
