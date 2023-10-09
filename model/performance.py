import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd
from config import j, dim
from time import time

latent_dim = dim()

K = np.load(j(f'results/{latent_dim}/koopman.npy'))
decoder = tf.keras.models.load_model(j(f'results/{latent_dim}/decoder.h5'))

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

latent_vectors = np.array([np.load(os.path.join(pathname, i)) for i in ls])
states = np.array([np.load(os.path.join(state_pathname, i)) for i in state_ls])


entries = []


for i in range(len(ls) - 10):

    dictionary = {'timestep': int(ls[i][:-4])}
    latent = latent_vectors[i]
    decoder.predict(np.zeros((1, 256))) # To Make sure there is no lag on tensorflow startup
    for n in range(1, 10):

        start_time = time()
        predicted_latent = latent @ np.linalg.matrix_power(K, n)
        predicted_state = decoder.predict(predicted_latent)
        end_time = time()

        prediction_time = end_time - start_time

        actual_latent = latent_vectors[i + n]
        actual_state = states[i + n]

        latent_mean = tf.reduce_mean(actual_latent)
        state_mean = tf.reduce_mean(actual_state)

        latent_rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_latent - predicted_latent)))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_state - predicted_state)))

        latent_mae = tf.reduce_mean(tf.abs(actual_latent - latent))
        mae = tf.reduce_mean(tf.abs(actual_state - predicted_state))

        epsilon = 1e-13
        latent_percent = 100 * np.sum(np.abs(predicted_latent - actual_latent)) / (np.sum(np.abs(actual_latent)) + epsilon)
        percent = 100 * np.sum(np.abs(predicted_state - actual_state)) / (np.sum(np.abs(actual_latent)) + epsilon)


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
