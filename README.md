# Offline Baselines (JAX)

Offline Baselines with JAX is a set of implementations of reinforcement learning algorithms in JAX.

This library is based on Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3), and JAXRL (https://github.com/ikostrikov/jaxrl).

### Windows 10

Offline baselines does not support Window OS.

### Install

```
git clone https://github.com/mjyoo2/offline_baselines_jax.git
python setup.py install
```

### Install using pip 
Install the offline baselines with jax package:
```
pip install git+https://github.com/mjyoo2/offline_baselines_jax
```

## Performance
We check speed SAC and TD3 algorithm. We use RTX 3090, Intel i9-10940. Learning environment is HalfCheetah-v2. 

| **Algorithm** | **Stable Baselines (Pytorch)** | **Offline Baselines (Jax)** |
|---------------|--------------------------------|-----------------------------|
| SAC           | 125 steps / 1 second           | 570 steps / 1 second        |
| TD3           | 240 steps / 1 second           | 800 steps / 1 second        |

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
