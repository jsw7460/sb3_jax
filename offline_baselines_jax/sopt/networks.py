from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from offline_baselines_jax.common.jax_layers import (
    create_mlp,
    Sequential,
    FlattenExtractor,
    BaseFeaturesExtractor
)

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -10
CLIPPING_CONST = 10.0


class ImgExtractor(nn.Module):
    img_shape: int = 32
    n_channel: int = 3
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return x.reshape((x.shape[0], self.img_shape, self.img_shape, self.n_channel))


class CondVaeGoalGenerator(nn.Module):
    recon_dim: int          # One side length of image shape
    latent_dim: int
    dropout: float
    kernel_size: int
    strides: int
    features: list

    enc_conv = None
    dec_conv = None
    encoder = None
    decoder = None
    mu = None
    log_std = None

    def setup(self):
        modules = []
        for i in range(len(self.features)):
            modules.append(
                nn.Conv(
                    features=self.features[i],
                    kernel_size=[self.kernel_size, self.kernel_size],
                    strides=self.strides
                )
            )
            modules.append(nn.relu)
        modules.append(FlattenExtractor())      # Note Fix here

        self.enc_conv = Sequential(modules)
        self.encoder = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[256, 128],
            dropout=self.dropout
        )
        self.mu = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[64, 32],
            dropout=self.dropout
        )
        self.log_std = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[64, 32],
            dropout=self.dropout
        )
        self.decoder = Sequential([
            create_mlp(
                output_dim=3 * self.recon_dim * self.recon_dim,
                net_arch=[128, 256, 512, 1024],
                dropout=self.dropout
            ),
            ImgExtractor(img_shape=self.recon_dim, n_channel=3)
        ])


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,              # This is not an image but (x, y) pos.
        target_future_hop: jnp.ndarray,
        deterministic: bool = False,
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, ...]]:

        mu, log_std = self.encode(state, goal, target_future_hop, deterministic)  # Use deterministic encoder. No sampling
        latent = self.get_latent_vector(mu, log_std)
        recon = self.decode(state, goal, target_future_hop, latent=latent, deterministic=deterministic)

        return recon, latent, (mu, log_std)

    def encode(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool = False
    ):
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        state: image based state.
        """
        proj = self.enc_conv(state)

        encoder_input = jnp.concatenate((proj, goal, target_future_hop), axis=1)
        emb = self.encoder(encoder_input, deterministic)
        mu = self.mu(emb, deterministic=deterministic)
        log_std = self.log_std(emb, deterministic=deterministic)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def decode(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool,
        latent: np.ndarray = None
    ) -> jnp.ndarray:
        """
        This is conditional VAE. So we conditionally input a true goal state.
        Here, true goal state means the final observation of given trajectory.
        """
        if latent is None:
            mu, log_std = self.encode(state, goal, target_future_hop, deterministic)
            latent = self.get_latent_vector(mu, log_std)

        decoder_input = jnp.concatenate((latent, goal, target_future_hop), axis=1)
        recon = self.decoder(decoder_input)
        return recon

    def deterministic_sampling(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool = True
    ):
        mu, log_std = self.encode(state, goal, target_future_hop, deterministic=True)
        recon = self.decode(state, goal, target_future_hop, deterministic=True, latent=mu)
        return recon, mu, (mu, log_std)

    def get_latent_vector(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        rng = self.make_rng("sampling")
        std = jnp.exp(log_std)
        latent = mu + std * jax.random.normal(rng, shape=mu.shape)
        return latent


class SensorBasedDoubleStateDiscriminator(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float

    latent_pi = None
    def setup(self):
        # Hyperparams from the paper demo-guided RL with learned skills (Pertsch et al.)
        self.latent_pi = create_mlp(
            output_dim=1,
            net_arch=[32, 32, 32],
            activation_fn=nn.leaky_relu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observation: jnp.ndarray, next_observation: jnp.ndarray, deterministic: bool = False):
        x1 = self.features_extractor(observation)
        x2 = self.features_extractor(next_observation)
        x = jnp.concatenate((x1, x2), axis=1)
        y = self.latent_pi(x, deterministic=deterministic)
        y = jnp.clip(y, -10.0, 10.0)
        y = nn.sigmoid(y)
        return y


class SensorBasedSingleStateDiscriminator(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float

    latent_pi = None
    def setup(self):

        self.latent_pi = create_mlp(
            output_dim=1,
            net_arch=[1],
            activation_fn=nn.leaky_relu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observation: jnp.ndarray, deterministic: bool = False):
        x = self.features_extractor(observation)
        y = self.latent_pi(x, deterministic=deterministic)
        y = jnp.clip(y, -10.0, 10.0)
        y = nn.sigmoid(y)
        return y


class SensorBasedSingleStateActionMatcherFromHighToLow(nn.Module):      # g: S x A_h x S --> A_l (A_h: Env action, A_l: Relabeled action)
    features_extractor: BaseFeaturesExtractor
    dropout: float
    highaction_dim: int
    squash_output: bool

    latent_pi = None

    def setup(self):
        self.latent_pi = create_mlp(
            output_dim=self.highaction_dim,
            net_arch=[64, 64],
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=self.squash_output
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        high_obs: jnp.ndarray,
        low_act: jnp.ndarray,
        deterministic: bool = False,
    ):
        high_obs = self.features_extractor(high_obs)

        x = jnp.concatenate((high_obs, low_act), axis=-1)
        return self.latent_pi(x, deterministic=deterministic)


class NaiveSensorBasedBehaviorCloner(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    lowaction_dim: int

    latent_pi = None
    def setup(self):
        self.latent_pi = create_mlp(
            output_dim=self.lowaction_dim,
            net_arch=[256, 256, 256],
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=True
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False
    ):
        features = self.features_extractor(observations)
        return self.latent_pi(features, deterministic=deterministic)


class SensorBasedInverseDynamics(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    highaction_dim: int
    squash_output: bool

    latent_pi = None

    def setup(self):
        self.latent_pi = create_mlp(
            output_dim=self.highaction_dim,
            net_arch=[128, 128, 128],
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=self.squash_output
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, next_observations: jnp.ndarray, deterministic: bool = False):
        observations = self.features_extractor(observations)
        next_observations = self.features_extractor(next_observations)

        x = jnp.concatenate((observations, next_observations), axis=1)
        return self.latent_pi(x, deterministic=deterministic)
