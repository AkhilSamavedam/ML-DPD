import tensorflow as tf
import jax.numpy as jnp
import jax
from jax import jit, vmap, pmap, grad
import os
import re
import pandas as pd
from config import j, dim
from time import time
from utils import decode

latent_dim = dim()

K = jnp.load(j(f'results/{latent_dim}/koopman.npy'))

pathname = j(f'latent/{latent_dim}')
state_pathname = j('Numpy')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


ls = os.listdir(pathname)
state_ls = os.listdir(state_pathname)

ls = [i for i in ls if i.endswith('.npy')]
state_ls = [i for i in state_ls if i.endswith('.npy')]

ls.sort(key=natural_sort_key)
state_ls.sort(key=natural_sort_key)

latent_vectors = jnp.array([jnp.load(os.path.join(pathname, i)) for i in ls])
states = jnp.array([jnp.load(os.path.join(state_pathname, i)) for i in state_ls])

entries = []

matrix_power = jit(jnp.linalg.matrix_power)

@jit
def measure(n):
    start_time = time()
    predicted_latent = latent @ matrix_power(K, n)
    predicted_state = decode(predicted_latent)
    end_time = time()

    prediction_time = end_time - start_time

    actual_latent = latent_vectors[i + n]
    actual_state = states[i + n]

    latent_mean = jnp.mean(actual_latent)
    state_mean = jnp.mean(actual_state)

    latent_rmse = jnp.sqrt(jnp.mean(jnp.square(actual_latent - predicted_latent)))
    rmse = jnp.sqrt(jnp.mean(jnp.square(actual_state - predicted_state)))

    latent_mae = jnp.mean(jnp.abs(actual_latent - latent))
    mae = jnp.mean(jnp.abs(actual_state - predicted_state))

    epsilon = 1e-13
    latent_percent = 100 * jnp.sum(jnp.abs(predicted_latent - actual_latent)) / (
            jnp.sum(jnp.abs(actual_latent)) + epsilon)
    percent = 100 * jnp.sum(jnp.abs(predicted_state - actual_state)) / (jnp.sum(jnp.abs(actual_latent)) + epsilon)

    return latent_rmse, rmse, latent_mae, mae, latent_percent, percent, prediction_time


for i in range(len(ls) - 10):

    dictionary = {'timestep': int(ls[i][:-4])}
    latent = latent_vectors[i]
    decode(jnp.zeros((1, 256)))  # To Make sure there is no lag on tensorflow startup
    for n in range(1, 10):
        latent_rmse, rmse, latent_mae, mae, latent_percent, percent, prediction_time = measure(n)

        dictionary[f'latent-rmse_{n}'] = float(latent_rmse)
        dictionary[f'rmse_{n}'] = float(rmse)
        dictionary[f'latent-mae_{n}'] = float(latent_mae)
        dictionary[f'mae_{n}'] = float(mae)
        dictionary[f'latent%_{n}'] = float(latent_percent)
        dictionary[f'%_{n}'] = float(percent)

        dictionary[f'time_{n}'] = float(prediction_time * 1000)  # converting time to ms

        print(f'{ls[i][:-4]} --> {ls[i + n][:-4]}')
    entries.append(dictionary)
df = pd.DataFrame.from_records(entries, index=['timestep'])
df.to_csv(j(f'results/{latent_dim}/performance.csv'))
