# Offline Baselines (JAX)

Offline Baselines with JAX is a set of rimplementations of reinforcement learning algorithms in JAX.

This library is based on Stable Baselines (https://github.com/DLR-RM/stable-baselines3), and JAXRL (https://github.com/ikostrikov/jaxrl).

## Implemented Algorithms

| **Name**       | **Online_learning** | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
|----------------|---------------------| ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| SAC            | :heavy_check_mark:  | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| CQL            | :heavy_check_mark:  | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| Multi-task SAC | :heavy_check_mark:  | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| Multi-task CQL | :heavy_check_mark:  | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |

### Windows 10

Offline baselines does not support Window OS.

### Install

```
python setup.py install
```

### Install using pip (not implemented)
Install the Stable Baselines3 package:
```
pip install offline_baselines_jax
```

## Example
```python
from offline_baselines_jax import SAC
from offline_baselines_jax.sac.policies import SACPolicy

import gym

train_env = gym.make('HalfCheetah-v2')

model = SAC(SACPolicy, train_env, seed=777, verbose=1, batch_size=1024, buffer_size=50000, train_freq=1)

model.learn(total_timesteps=10000)
model.save('./model.zip')
model = SAC.load('./model.zip', train_env)

model.learn(total_timesteps=10000)

```