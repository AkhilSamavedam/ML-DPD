import tensorflow as tf
import os
from config import j, latent_shape, tensor_shape
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def load_npy_files(file_paths):
    for file_path in file_paths:
        data = np.load(file_path)
        yield tf.convert_to_tensor(data, dtype=tf.float32)


def create_dataset_from_npy_folder(folder_path=j('Numpy'), batch_size=32, train_val_split=0.8):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_list.sort()

    def generator():
        for file in file_list:
            data = np.load(os.path.join(folder_path, file))
            yield data.astype(tf.float32)  # Make sure data is of dtype float32

    dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(4200, 244, 3), dtype=tf.float32))

    # Duplicate the dataset to get both input (x) and target (y) as the same data (for autoencoder)
    # dataset = dataset.map(lambda x: (x, x))

    # Split the dataset into training and validation sets
    num_files = len(file_list)
    num_train_files = int(train_val_split * num_files)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]

    train_dataset = dataset.take(num_train_files)
    val_dataset = dataset.skip(num_train_files)

    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, file_list


K = tf.convert_to_tensor(np.load(j(f'results/{latent_shape()}/koopman.npy')), dtype=tf.float32)
j = tf.constant(0)


@tf.function
def koopman_power(n):
    def body(i, result):
        return i + 1, result @ K

    i = j
    result = K

    _, result = tf.while_loop(lambda i, _: i < n - 1, body, loop_vars=(i, result))

    return result


koopman_power(1) # Forces compilation of koopman_power