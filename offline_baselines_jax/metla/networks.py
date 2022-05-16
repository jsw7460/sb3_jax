from typing import Tuple, Any, List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from offline_baselines_jax.common.jax_layers import create_mlp
from .buffer import TrajectoryBuffer

tfd = tfp.distributions
tfb = tfp.bijectors


Params = flax.core.FrozenDict[str, Any]
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -10
CLIPPING_CONST = 10.0


class FlattenExtractor(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return x.reshape((x.shape[0], -1))


class SASPredictor(nn.Module):      # Input: s, a // Output: predicted next state (S x A --> S: SAS)
    state_dim: int
    net_arch: List = None
    dropout: float = 0.0
    squash_output: bool = False

    predictor = None

    def setup(self):
        net_arch = self.net_arch
        if net_arch is None:
            net_arch = [256, 256]

        self.predictor = create_mlp(
            output_dim=self.state_dim,
            net_arch=net_arch,
            dropout=self.dropout,
            squash_output=self.squash_output,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, actions: jnp.ndarray, deterministic=False):
        x = jnp.concatenate((observations, actions), axis=-1)
        return self.predictor(x, deterministic=deterministic)


class VariationalAutoEncoder(nn.Module):
    recon_dim: int
    dropout: float

    flatten_extractor = None
    encoder = None
    decoder = None
    mu = None
    log_std = None

    def setup(self):
        self.flatten_extractor = FlattenExtractor()

        self.encoder = create_mlp(
            output_dim=20,  # 이건 상관이 없다.
            net_arch=[64, 64],
            dropout=self.dropout,
            squash_output=False
        )

        self.mu = create_mlp(
            output_dim=20,
            net_arch=[64, 64],
            dropout=self.dropout,
            squash_output=False
        )

        self.log_std = create_mlp(
            output_dim=20,
            net_arch=[64, 64],
            dropout=self.dropout,
            squash_output=False
        )

        self.decoder = create_mlp(
            output_dim=self.recon_dim,
            net_arch=[64, 64],
            dropout=self.dropout,
            squash_output=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        history: jnp.ndarray,
        cur_state: jnp.ndarray,
        deterministic: bool = False,
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, ...]]:

        if history.ndim == 1:
            history = history[np.newaxis, np.newaxis, ...]
        elif history.ndim == 2:
            history = history[np.newaxis, ...]

        history = TrajectoryBuffer.timestep_marking(history)

        mu, log_std = self.encode(history, cur_state, deterministic)        # Use deterministic encoder. No sampling
        latent = self.get_latent_vector(mu, log_std)
        recon = self.decode(history, latent=latent, deterministic=deterministic)

        return recon, latent, (mu, log_std)

    def encode(self, history: np.ndarray, cur_state: jnp.ndarray, deterministic: bool):
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        """
        history = self.flatten_extractor(history)
        encoder_input = jnp.hstack((history, cur_state))
        emb = self.encoder(encoder_input, deterministic)
        mu = self.mu(emb, deterministic=deterministic)
        log_std = self.log_std(emb, deterministic=deterministic)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def decode(self, history: np.ndarray, deterministic: bool, latent: np.ndarray = None) -> jnp.ndarray:
        if latent is None:
            history, *_ = TrajectoryBuffer.timestep_marking(history)
            mu, log_std = self.encode(history, deterministic)
            latent = self.get_latent_vector(mu, log_std)

        recon = self.decoder(latent, deterministic)
        return recon

    @staticmethod
    def get_latent_vector(mu: np.ndarray, log_std: np.ndarray, key: jnp.ndarray) -> np.ndarray:
        std = jnp.exp(log_std)
        latent = mu + std * jax.random.normal(key, shape=mu.shape)
        return latent


class WassersteinAutoEncoder(nn.Module):
    recon_dim: int      # Reconstruction 하는 대상의 dimension
    dropout: float
    rbf_var: float = 5.0
    reg_weight: float = 100.0

    flatten_extractor = None
    encoder = None
    decoder = None
    mu = None
    log_std = None

    def setup(self):
        self.flatten_extractor = FlattenExtractor()

        self.encoder = create_mlp(
            output_dim=8,          # 이건 상관이 없다.
            net_arch=[64, 128, 256],
            dropout=self.dropout,
            squash_output=False
        )

        self.decoder = create_mlp(
            output_dim=self.recon_dim,
            net_arch=[256, 128, 64],
            dropout=self.dropout,
            squash_output=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        history: jnp.ndarray,
        cur_state: jnp.ndarray,
        deterministic: bool = False,
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        rng = self.make_rng("noise")

        if history.ndim == 1:
            history = history[np.newaxis, np.newaxis, ...]
        elif history.ndim == 2:
            history = history[np.newaxis, ...]

        history = TrajectoryBuffer.timestep_marking(history)

        latent = self.encode(history, cur_state, deterministic)        # Use deterministic encoder. No sampling
        latent = (latent + jax.random.normal(rng, latent.shape) * 0.05)
        recon = self.decode(history, latent=latent, deterministic=deterministic)

        return recon, latent

    def encode(self, history: np.ndarray, cur_state: jnp.ndarray, deterministic: bool):
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        """
        history = self.flatten_extractor(history)
        encoder_input = jnp.hstack((history, cur_state))
        latent = self.encoder(encoder_input, deterministic)
        return latent

    def decode(self, history: np.ndarray, deterministic: bool, latent: np.ndarray = None) -> jnp.ndarray:
        if latent is None:
            history, *_ = TrajectoryBuffer.timestep_marking(history)
            latent = self.encode(history, deterministic)
        recon = self.decoder(latent, deterministic)
        recon = CLIPPING_CONST * jnp.tanh(recon)
        return recon

    def rbf_mmd_loss(self, z: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        # Compute the mmd loss with rbf kernel
        prior_shape = z.shape
        batch_size = prior_shape[0]

        reg_weight = self.reg_weight / batch_size
        prior_z = jax.random.normal(key, prior_shape)       # batch_size, latent_dim

        zz = jnp.mean(self.compute_kernel(prior_z, prior_z))
        zhat_zhat = jnp.mean(self.compute_kernel(z, z))
        z_zhat = jnp.mean(self.compute_kernel(z, prior_z))

        return reg_weight * (zz + zhat_zhat - 2 * z_zhat)

    def compute_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        z_dim = x1.shape[-1]
        x1 = jnp.expand_dims(x1, axis=0)
        x2 = jnp.expand_dims(x2, axis=1)
        kernel = jnp.exp(- (x1 - x2) ** 2 / (2.0 * z_dim * self.rbf_var))
        return kernel


class GeneralizedAutoEncoder(nn.Module):
    recon_dim: int
    latent_dim: int
    squashed_out: bool
    dropout: float = 0.1
    n_nbd: int = 5      # How many states we reconstruct?

    flatten_extractor = None
    encoder = None
    decoder = None

    def setup(self):
        self.flatten_extractor = FlattenExtractor()

        self.encoder = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[64, 32],
            dropout=self.dropout,
            squash_output=False
        )

        # Reconstruct nearest "2 * (n_nbd + 1)" states.
        # This is due to the "n_nbd history + current + n_nbd future" recon.
        self.decoder = create_mlp(
            output_dim=self.recon_dim * (2 * self.n_nbd + 1),
            net_arch=[32, 64, 128],
            dropout=self.dropout,
            squash_output=self.squashed_out
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, ...]]:

        latent = self.encode(observations, deterministic)  # Use deterministic encoder. No sampling
        recon = self.decode(observations, latent=latent, deterministic=deterministic)
        return recon, latent

    def encode(self, observations: jnp.ndarray, deterministic: bool = False):
        # observations = self.flatten_extractor(observations)
        latent = self.encoder(observations, deterministic=deterministic)
        latent = CLIPPING_CONST * jnp.tanh(latent)
        return latent

    def decode(self, observations: jnp.ndarray, deterministic: bool = False, latent: jnp.ndarray = None) -> jnp.ndarray:
        batch_size = observations.shape[0]

        if latent is None:
            observations, *_ = TrajectoryBuffer.timestep_marking(observations)
            latent = self.encode(observations, deterministic)

        recon = self.decoder(latent, deterministic)
        recon = recon.reshape(batch_size, -1, 2 * self.n_nbd + 1, self.recon_dim)
        return jnp.squeeze(recon)
