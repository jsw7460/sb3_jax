import os

from offline_baselines_jax.sac import SAC
from offline_baselines_jax.cql import CQL
from offline_baselines_jax.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()