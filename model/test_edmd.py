import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd
from config import j


K = np.load(j('results/256/koopman.npy'))
decoder = tf.keras.models.load_model(j('results/256/decoder.h5'))


pathname = j('latent/256')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

ls = os.listdir(pathname)
ls.sort(key=natural_sort_key)

history_file = j('results/256/edmd.csv')

entries = []

for i in range(len(ls) - 10):

    dictionary = {'timestep': int(ls[i][:-4]), 'timestep_predicted': int(ls[i + 1][:-4])}

    for n in range(1, 10):
        latent = np.load(os.path.join(pathname, ls[i]))
        predicted_latent = (np.linalg.matrix_power(K, n) @ latent.T).T
        predicted_state = decoder.predict(predicted_latent)
        actual_latent = np.load(os.path.join(pathname, ls[i + n]))
        actual_state = np.load(os.path.join(pathname, j('Numpy'), f'slice_{ls[i + n]}'))

        latent_mean = tf.reduce_mean(actual_latent)
        state_mean = tf.reduce_mean(actual_state)

        latent_rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_latent - predicted_latent)))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_state - predicted_state)))
        epsilon = 1e-10
        latent_percent = 100 * np.sum(np.abs(predicted_latent - actual_latent)) / (np.sum(np.abs(actual_latent)) + epsilon)
        percent = 100 * np.sum(np.abs(predicted_state - actual_state)) / (np.sum(np.abs(actual_latent)) + epsilon)

        dictionary[f'latent_rmse${n}'] = float(latent_rmse)
        dictionary[f'rmse${n}'] = float(rmse)
        dictionary[f'latent%${n}'] = float(latent_percent)
        dictionary[f'%${n}'] = float(percent)
    entries.append(dictionary)
    print(f'{ls[i][:-4]} --> {ls[i + 1][:-4]}')
df = pd.DataFrame.from_records(entries, index=['timestep'])
df.to_csv(j('results/256/edmd.csv'))
