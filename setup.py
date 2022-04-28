import os

from setuptools import find_packages, setup

with open(os.path.join("offline_baselines_jax", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="offline_baselines_jax",
    packages=[package for package in find_packages() if package.startswith("offline_baselines_jax")],
    package_data={"offline_baselines_jax": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3==1.4.0",
        "jax=0.3.4",
        "jax=3.2 + cuda11.cudnn82",
        "flax==0.4.0",
        "tensorflow_probability",
        'optax==0.1.1'
    ],
    description="Jax version of implementations of offline reinforcement learning algorithms.",
    author="Minjong Yoo",
    url="https://github.com/mjyoo2/offline_baselines_jax",
    author_email="mjyoo222@gmail.com",
    license="MIT",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
