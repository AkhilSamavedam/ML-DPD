import jax
from jax import jit, vmap, pmap, grad
import jax.numpy as jnp
import pandas as pd
import tensorflow as tf
from jax.experimental import jax2tf
from config import j, dim

latent_dim = dim()

encoder = tf.keras.models.load_model(j(f'results/{latent_dim}/encoder.h5'))
decoder = tf.keras.models.load_model(j(f'results/{latent_dim}/decoder.h5'))

encoder = jax2tf.convert(encoder)
decoder = jax2tf.convert(decoder)

def encode(X):
    return encoder(X)

def decode(X):
    return decoder(X)


