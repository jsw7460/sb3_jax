import numpy as np
import gym
import d4rl_mujoco

from offline_baselines_jax import SAC


STATE_DIM = 17
ACTION_DIM = 4

env = gym.make("Walker2d-v3")

model = SAC(
    env,
    verbose=1,
    learning_starts=5000,
    batch_size=256,
    buffer_size=100_000,
    train_freq=1,
    without_exploration=False,
    seed=777
)
model.learn(10000000)
