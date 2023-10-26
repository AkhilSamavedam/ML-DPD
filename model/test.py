import os
import graphviz
import optax
import jax
import haiku as hk
from jax import jit
import jax.numpy as jnp
from utils import UpSampling2D, Cropping2D
from config import latent_shape, tensor_shape, j
from time import time
import cloudpickle as cpkl
import orbax
from glob import glob


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


def get_dataset(path, training_split):
    ls = glob(f'{path}/*.npy')
    data = jnp.expand_dims(jnp.zeros(tensor_shape()), axis=0)
    for fn in ls:
        data = jnp.concatenate((data, jnp.expand_dims(jnp.load(fn), axis=0)))
    train_len = int(data.shape[0] * training_split)
    #data = jax.random.permutation(rng, data, axis=0, independent=True)
    return data[:train_len], data[train_len:]


# Batch input data
batch_size = 2
train_data, val_data = get_dataset(j(f'Numpy'), training_split=0.8)  # batch dataset
print(train_data.shape)

num_epochs = 10

# Training loop

opt = optax.adam(learning_rate=1e-3)
opt_state = opt.init(params)

'''@jit
def batch(data, batch_size):
    return jnp.split(data, batch_size, axis=0)
'''

@jit
def loss_fn(params, batch):
    latent_code = encoder_apply(params, batch)
    reconstruction = decoder_apply(params, latent_code)
    mse = jnp.mean(jnp.square(batch - reconstruction))
    return mse


@jit
def step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


#batches = batch(train_data, batch_size)

for epoch in range(num_epochs):
    batch = jnp.zeros((5, 4200, 244, 3))
    t0 = time()
    params, opt_state, loss_value = step(params, opt_state, batch)
    t = time()
    print(f'step {epoch}, loss: {loss_value}, step_time: {t - t0}')
    print(f'Val Loss {loss_fn(params, val_data)}')


@jit
def encoder(x):
    return encoder_apply(params, x)


@jit
def decoder(x):
    return decoder_apply(params, x)


with open(j(f'results/{latent_shape()}/encoder.hk'), 'wb') as f:
    cpkl.dump(encoder, f)

with open(j(f'results/{latent_shape()}/decoder.hk'), 'wb') as f:
    cpkl.dump(decoder, f)


def plot_model(name, model, x):
    dot = hk.experimental.to_dot(model)(x)
    graph = graphviz.Source(dot)
    graph.render(j(f'results/{latent_shape()}/{name}'), view=False)
    os.remove(j(f'results/{latent_shape()}/{name}'))


plot_model('encoder', encoder, jnp.zeros((1, 4200, 244, 3)))
plot_model('decoder', decoder, jnp.zeros((1, latent_shape())))
