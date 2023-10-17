import jax
from jax import jit, vmap, pmap, grad
import jax.numpy as jnp
import pandas as pd
import tensorflow as tf
from config import j, dim

latent_dim = dim()

encoder = tf.keras.models.load_model(j(f'results/{latent_dim}/encoder.h5'))
decoder = tf.keras.models.load_model(j(f'results/{latent_dim}/decoder.h5'))

@jit
def encode(X):
    return encoder.predict(X)

@jit
def decode(X):
    return decoder.predict(X)


