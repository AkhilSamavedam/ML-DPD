import os
import graphviz
import optax
import jax
import haiku as hk
from jax import jit
import jax.numpy as jnp
from config import latent_shape, tensor_shape, j
from time import time
import cloudpickle as cpkl
import orbax
from glob import glob
from utils import create_dataset_from_npy_folder

class UpSampling2D(hk.Module):
    def __init__(self, shape, method='nearest'):
        super().__init__()
        self.shape = shape
        self.method = method

    def __call__(self, x, *args, **kwargs):
        if x.ndim == 3:  # Detect channel dimension
            height, width, channels = x.shape
            new_height = height * self.shape[0]
            new_width = width * self.shape[1]
            resized_image = jax.image.resize(x, (new_height, new_width, channels), method=self.method)
            return resized_image
        elif x.ndim == 4:  # Detect batch dimension
            batch_size, height, width, channels = x.shape
            new_height = height * self.shape[0]
            new_width = width * self.shape[1]
            resized_image = jax.image.resize(x, (batch_size, new_height, new_width, channels), method=self.method)
            return resized_image
        else:
            raise ValueError('Invalid input dimension')


class Cropping2D(hk.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def __call__(self, x, *args, **kwargs):
        (top, bottom), (left, right) = self.cropping
        if x.ndim == 3:
            if bottom == 0:
                bottom = -x.shape[0]
            if right == 0:
                right = -x.shape[1]
            return x[top:-bottom, left:-right, :]
        elif x.ndim == 4:
            if bottom == 0:
                bottom = -x.shape[1]
            if right == 0:
                right = -x.shape[2]
            return x[:, top:-bottom, left:-right, :]
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
train_data, val_data, _ = create_dataset_from_npy_folder()


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
    for batch in train_data:
        batch = jnp.array(batch.numpy())
        t0 = time()
        params, opt_state, loss_value = step(params, opt_state, batch)
        t = time()
        print(f'step {epoch}, loss: {loss_value}, step_time: {t - t0}')
    val_loss = jnp.mean(jnp.array([loss_fn(params, jnp.array(val_batch.numpy())) for val_batch in val_data]))
    print(f'Validation Loss after epoch {epoch}: {val_loss}')


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