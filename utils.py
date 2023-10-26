import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk


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