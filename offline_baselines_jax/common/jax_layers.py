import math
from typing import Dict, List, Tuple, Type, Union, Sequence, Any, Callable

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
from flax.linen.initializers import zeros
from jax import lax

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import is_image_space
from offline_baselines_jax.common.type_aliases import TensorDict

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, float]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]

default_kernel_init = nn.initializers.xavier_normal()
default_bias_init = zeros


def polyak_update(source: Model, target: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), source.params, target.params)
    return target.replace(params=new_target_params)


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def create_mlp(
    output_dim: int,
    net_arch: List[int],
    activation_fn: Callable = nn.relu,
    dropout: float = 0.0,
    squash_output: bool = False,
    layernorm: bool = False,
    batchnorm: bool = False,
    use_bias: bool = True,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
) -> nn.Module:

    if output_dim > 0:
        net_arch = list(net_arch)
        net_arch.append(output_dim)
    return MLP(net_arch, activation_fn, dropout, squash_output, layernorm, batchnorm, use_bias, kernel_init, bias_init)


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
    try:
        net_arch = list(net_arch)
    except:
        pass

    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


class MLP(nn.Module):
    net_arch: List
    activation_fn: nn.Module
    dropout: float = 0.0
    squashed_out: bool = False

    layernorm: bool = False
    batchnorm: bool = False
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, x, deterministic: bool = False, training: bool = True):

        for feature in self.net_arch[:-1]:
            x = nn.Dense(feature, kernel_init=self.kernel_init, use_bias=self.use_bias, bias_init=self.bias_init)(x)
            if self.batchnorm: x = nn.BatchNorm(use_running_average=not training, momentum=0.1)(x)
            if self.layernorm: x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        if len(self.net_arch) > 0:
            x = nn.Dense(
                self.net_arch[-1],
                kernel_init=self.kernel_init,
                use_bias=self.use_bias,
                bias_init=self.bias_init
            )(x)

        if self.squashed_out: return nn.tanh(x)
        else: return x


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, *args, **kwargs):
      for layer in self.layers:
        x = layer(x, *args, **kwargs)
      return x


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
        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(x)
        return x


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
