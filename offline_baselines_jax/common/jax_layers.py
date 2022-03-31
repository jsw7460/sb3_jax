from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any

import os
import gym
import jax
import flax
import optax

import jax.numpy as jnp
import flax.linen as nn

from offline_baselines_jax.common.preprocessing import is_image_space
from offline_baselines_jax.common.type_aliases import TensorDict
from dataclasses import dataclass, field

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

def default_init():
    return nn.initializers.he_normal()

"""
  actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
"""

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    _observation_space: gym.Space
    _feature_dim: int = 0

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def __call__(self, observations: jnp.array) -> jnp.array:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return x.reshape((x.shape[0], -1))


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    feature_dim: int = 512

    @nn.compact
    def __call__(self, observations: jnp.array) -> jnp.array:
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(observations)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # flatten

        x = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(x)
        return x

class ResourceRestrictedSoftModule(nn.Module):
    select_arch: List[int]
    num_modules: int

    @nn.compact
    def __call__(self, observation:TensorDict) -> jnp.ndarray:
        obs = observation['obs']
        task = observation['task']
        state_embeddings = MLP(net_arch = self.stata_base_arch, last_activation=True)(obs)
        task_embeddings = MLP(net_arch = self.task_base_arch, last_activation=True)(task)
        routing_inputs = state_embeddings * task_embeddings
        module_selection = MLP(net_arch=self.select_arch + [self.num_modules])(routing_inputs)
        return module_selection


class SoftModule(nn.Module):
    _observation_space: gym.spaces.Dict
    state_base_arch: List[int] = field(default_factory=lambda : [64, 64])
    task_base_arch: List[int] = field(default_factory=lambda : [64])
    net_arch: List[int] = field(default_factory=lambda : [4, 4, 4])
    module_arch: List[int] = field(default_factory=lambda : [32])

    @nn.compact
    def __call__(self, observation: TensorDict, module_select:List[List[int]] = None) -> jnp.ndarray:
        assert self.state_base_arch[-1] == self.task_base_arch[-1]
        obs = observation['obs']
        task = observation['task']
        state_embeddings = MLP(net_arch = self.state_base_arch, last_activation=True)(obs)
        task_embeddings = MLP(net_arch = self.task_base_arch, last_activation=True)(task)
        routing_inputs = state_embeddings * task_embeddings

        prev_model_outputs = [state_embeddings] * self.net_arch[0]
        first_routing_inputs = routing_inputs
        module_attention_list = []

        if module_select is None:
            module_select = []
            for num_module in self.net_arch:
                module_select.append(jnp.ones(num_module))

        for idx in range(len(self.net_arch) - 1):
            routing_output_dim = self.net_arch[idx] * self.net_arch[idx + 1]
            module_attention = MLP(net_arch=[routing_output_dim])(routing_inputs)
            module_attention = nn.softmax(jnp.reshape(module_attention, (-1, self.net_arch[idx+1], self.net_arch[idx])))
            # module_attention = module_attention[..., tuple(module_select[idx + 1]), tuple(module_select[idx])]

            # Not in Paper, but implemented in github (https://github.com/RchalYang/Soft-Module)
            module_attention_list.append(jnp.reshape(module_attention, (-1, self.net_arch[idx + 1] * self.net_arch[idx])))

            module_outputs = []
            for i in range(self.net_arch[idx]):
                module_outputs.append(MLP(net_arch=self.module_arch, last_activation=True)(prev_model_outputs[i]) * module_select[idx][i])


            prev_model_outputs = []
            for i in range(self.net_arch[idx + 1]):
                model_inputs = jnp.zeros_like(module_outputs[0])
                for j in range(self.net_arch[idx]):
                    if module_outputs[j] is not None:
                        model_inputs += jnp.expand_dims(module_attention[..., i, j], axis=-1) * module_outputs[j]
                prev_model_outputs.append(model_inputs)

            cond = MLP(net_arch=[routing_inputs.shape[-1]])(jnp.concatenate(module_attention_list, axis=-1))
            cond = cond * first_routing_inputs
            routing_inputs = nn.relu(cond)

        module_outputs = []
        for i in range(self.net_arch[-1]):
            module_outputs.append(MLP(net_arch=self.module_arch, last_activation=True)(prev_model_outputs[i]) * module_select[-1][i])

        model_inputs = jnp.zeros_like(module_outputs[0])
        for j in range(len(module_outputs)):
            model_inputs += module_outputs[j]
        output = nn.relu(model_inputs)
        return output


class ModuleLayer(nn.Module):
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    n_modules: int = 2
    last_activation: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        VmapCritic = nn.vmap(MLP,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.n_modules)
        qs = VmapCritic(self.net_arch, self.activation_fn, self.last_activation)(inputs)
        return qs

class MLP(nn.Module):
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    squash_output: bool = False
    last_activation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.net_arch):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.net_arch) or self.last_activation:
                x = self.activation_fn(x)
        if self.squash_output:
            x = nn.tanh(x)
        return x

def create_mlp(output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.relu,
               squash_output: bool = False) -> nn.Module:
    if output_dim > 0:
        net_arch = list(net_arch)
        net_arch.append(output_dim)
    return MLP(net_arch, activation_fn, squash_output)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """
    _observation_space: gym.spaces.Dict
    cnn_output_dim: int = 256

    @nn.compact
    def __call__(self, observation: TensorDict):
        encoded_tensor_list = []
        for key, subspace in self._observation_space.spaces.items():
            if is_image_space(subspace):
                encoded_tensor_list.append(NatureCNN(self.cnn_output_dim)(observation[key]))
            else:
                # The observation key is a vector, flatten it if needed
                encoded_tensor_list.append(observation[key].reshape((observation[key].shape[0], -1)))
        return jnp.concatenate(encoded_tensor_list, axis=1)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


if __name__ == '__main__':
    import numpy as np
    jax_array = {'aa': jnp.array(np.array([[1, 2], [3, 4]])), 'bb': jnp.array(np.array([[5, 6], [7, 8]]))}

    model = CombinedExtractor(_observation_space=gym.spaces.Dict({'aa': gym.spaces.Box(low=np.zeros(2), high=np.ones(2)),
                                                                 'bb': gym.spaces.Box(low=np.zeros(2), high=np.ones(2))}))
    variable = model.init(jax.random.PRNGKey(0), jax_array)
    print(model.apply(variable, jax_array))