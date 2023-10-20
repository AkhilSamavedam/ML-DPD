import jax
import numpy as np
from jax import jit, vmap, pmap, grad
import jax.numpy as jnp
import pandas as pd
import tensorflow as tf
import tf2jax
from config import j, latent_shape, tensor_shape

tensor_dim = tensor_shape()
latent_dim = latent_shape()

encoder = tf.keras.models.load_model(j(f'results/{latent_dim}/encoder.h5'))
decoder = tf.keras.models.load_model(j(f'results/{latent_dim}/decoder.h5'))

def e(x):
    return encoder(x)


def d(x):
    return decoder(x)


xla_e = tf.function(e, jit_compile=True)
xla_d = tf.function(d, jit_compile=True)

jax_e, e_params = tf2jax.convert(xla_e, jnp.expand_dims(jnp.zeros(shape=tensor_dim), axis=0))
jax_d, d_params = tf2jax.convert(xla_d, jnp.expand_dims(jnp.zeros(shape=latent_dim), axis=0))


@jit
def encode(x):
    return jax_e(e_params, x)[0]

@jit
def decode(x):
    return jax_d(d_params, x)[0]
