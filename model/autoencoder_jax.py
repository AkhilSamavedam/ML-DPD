import jax
from jax import pmap, vmap, grad, jit
import jax.numpy as jnp
import h5py
import haiku as hk
import equinox as eqx
import optax
import orbax
from config import j, latent_shape, input_shape
import os
from encoder_decoder import decode

key = jax.random.PRNGKey(5678)

latent_dim = latent_shape()
input_shape = input_shape()


class UpSampling2D(hk.Module):
    def __init__(self, shape=(2, 2), method='bilinear', name: str = None, ):
        super().__init__(name=name)
        self.shape = shape
        self.method = method

    def __call__(self, x, *args, **kwargs):
        resized = jax.image.resize(
            x,
            shape=(
                x.shape[0],
                x.shape[1] * self.shape[0],
                x.shape[2] * self.shape[1],
                x.shape[3]
            ),
            method=self.method
        )
        return resized


def Encoder(x):
    x = hk.Conv2D(32, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Conv2D(64, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Conv2D(128, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Conv2D(64, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Conv2D(64, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Conv2D(64, kernel_shape=3, padding='SAME',
                  w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)

    x = hk.Flatten()(x)
    return hk.Linear(latent_dim)(x)


def Decoder(x):
    x = hk.Linear(16896)(x)
    x = jax.nn.relu(x)
    x = hk.Reshape((66, 4, 64))(x)

    x = hk.Conv2DTranspose(64, kernel_shape=3, stride=1, padding='SAME',
                           w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = UpSampling2D(shape=(2, 2), method='nearest')(x)

    x = hk.Conv2DTranspose(64, kernel_shape=3, stride=1, padding='SAME',
                           w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jax.nn.relu(x)
    x = UpSampling2D(shape=(2, 2), method='nearest')(x)

    # Rest of model

    x = hk.Conv2DTranspose(3, kernel_shape=3, stride=1, padding='SAME',
                           w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(x)
    x = jnp.tanh(x)  # Use tanh instead of selu

    return x

def Autoencoder(x):
    z = Encoder(x)
    x_prime = Decoder(z)
    return x_prime


Encoder = hk.transform(Encoder)
Decoder = hk.transform(Decoder)
Autoencoder = hk.transform(Autoencoder)


