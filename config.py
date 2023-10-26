import os.path
import sys

from jax import jit
import jax.numpy as jnp

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def j(rel_path):
    return os.path.join(PROJECT_DIR, rel_path)


def latent_shape():
    return 256


def tensor_shape():
    return 4200, 244, 3
