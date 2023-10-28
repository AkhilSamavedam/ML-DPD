import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk
import os
from config import j


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


def load_npy_files(file_paths):
    for file_path in file_paths:
        data = jnp.load(file_path)
        yield tf.convert_to_tensor(data, dtype=jnp.float32)


def create_dataset_from_npy_folder(folder_path=j('Numpy'), batch_size=32, train_val_split=0.8):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_list.sort()

    def generator():
        for file in file_list:
            data = jnp.load(os.path.join(folder_path, file))
            yield data.astype(jnp.float32)  # Make sure data is of dtype float32

    dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(4200, 244, 3), dtype=jnp.float32))

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
