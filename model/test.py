import optax
import jax
import haiku as hk
from jax import jit
import jax.numpy as jnp
from utils import UpSampling2D, Cropping2D
from config import latent_shape, tensor_shape, j
from time import time
import orbax


class Encoder(hk.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.layers = [
            hk.Conv2D(32, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Conv2D(64, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Conv2D(128, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Conv2D(64, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Conv2D(64, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Conv2D(64, kernel_shape=(3, 3), padding='SAME'),
            jax.nn.relu,
            hk.MaxPool((2, 2), padding='SAME', strides=2),
            hk.Flatten(),
            hk.Linear(self.latent_dim)
        ]

    def __call__(self, x, *args, **kwargs) -> jnp.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(hk.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_dim
        self.layers = [
            hk.Linear(16896),
            jax.nn.relu,
            hk.Reshape((66, 4, 64)),
            hk.Conv2DTranspose(64, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(64, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(64, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(128, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(64, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(32, (3, 3), padding='SAME'),
            jax.nn.relu,
            UpSampling2D((2, 2)),
            hk.Conv2DTranspose(3, (3, 3), padding='SAME'),
            jax.nn.selu,
            Cropping2D(((0, 24), (0, 12)))
        ]

    def __call__(self, x, *args, **kwargs) -> jnp.array:
        for layer in self.layers:
            x = layer(x)
        return x


def encoder(x):
    e = Encoder(latent_shape(), tensor_shape())
    return e(x)


def decoder(x):
    d = Decoder(latent_shape(), tensor_shape())
    return d(x)


encoder = hk.without_apply_rng(hk.transform(encoder))
decoder = hk.without_apply_rng(hk.transform(decoder))

encoder_apply = jit(encoder.apply)
decoder_apply = jit(decoder.apply)


# Initialize parameters.
rng = jax.random.PRNGKey(42)

params = hk.data_structures.merge(
    encoder.init(rng, jnp.expand_dims(jax.random.normal(rng, tensor_shape()), axis=0)),
    decoder.init(rng, jnp.expand_dims(jax.random.normal(rng, [latent_shape()]), axis=0))
)


# Define optimizer.
x = jax.random.normal(rng, (5, 4200, 244, 3))
latent = jax.random.normal(rng, (5, 1024))

#print(hk.experimental.tabulate(encoder)(x))
#print(hk.experimental.tabulate(decoder)(latent))

"""
for name, param in params.items():
  if isinstance(param, dict):
    print(f"{name}: {param['w'].shape}")
"""


# Batch input data
batch_size = 32
train_data = x  # batch dataset
num_epochs = 10

# Training loop

opt = optax.adam(learning_rate=1e-3)
opt_state = opt.init(params)


@jit
def loss_fn(params, batch):
    x = batch
    latent_code = encoder_apply(params, x)
    reconstruction = decoder_apply(params, latent_code)
    mse = jnp.mean(jnp.square(x - reconstruction))
    return mse


@jit
def step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


for i, batch in enumerate(train_data):
    t0 = time()
    params, opt_state, loss_value = step(params, opt_state, jnp.expand_dims(batch, axis=0))
    t = time()
    print(f'step {i}, loss: {loss_value}, step_time: {t - t0}')

encoder = jit(lambda x: encoder_apply(params, x))
decoder = jit(lambda x: decoder_apply(params, x))

t0 = time()
print(encoder_apply(params, jax.random.normal(rng, (1, 4200, 244, 3))).shape)
t = time()
print(t - t0)
