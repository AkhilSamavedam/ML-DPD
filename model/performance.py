import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd
from config import j, latent_shape
from timeit import default_timer as time
from encoder_decoder import decode
from utils import koopman_power
from timeit import timeit

latent_dim = latent_shape()

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

latent_vectors = tf.convert_to_tensor([np.load(os.path.join(pathname, i)) for i in ls])
states = tf.convert_to_tensor([np.load(os.path.join(state_pathname, i)) for i in state_ls])

entries = []


def measure(n):
    predicted_latent = latent @ koopman_power(n)
    predicted_state = decode(predicted_latent)


    prediction_time = timeit('latent @ koopman_power(n)', 'from __main__ import koopman_power, n, latent')

    actual_latent = latent_vectors[i + n]
    actual_state = states[i + n]

    latent_mean = tf.reduce_mean(actual_latent)
    state_mean = tf.reduce_mean(actual_state)

    latent_rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_latent - predicted_latent)))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_state - predicted_state)))

    latent_mae = tf.reduce_mean(tf.abs(actual_latent - latent))
    mae = tf.reduce_mean(tf.abs(actual_state - predicted_state))

    epsilon = 1e-13
    latent_percent = 100 * tf.reduce_sum(tf.abs(predicted_latent - actual_latent)) / (
            tf.reduce_sum(tf.abs(actual_latent)) + epsilon)
    percent = 100 * tf.reduce_sum(tf.abs(predicted_state - actual_state)) / (tf.reduce_sum(tf.abs(actual_latent)) + epsilon)

    return latent_rmse, rmse, latent_mae, mae, latent_percent, percent, prediction_time


for i in range(len(ls) - 10):

    dictionary = {'timestep': int(ls[i][:-4])}
    latent = latent_vectors[i]
    decode(tf.zeros((1, latent_dim)))  # To Make sure there is no lag on tensorflow startup
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
