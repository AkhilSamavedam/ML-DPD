import tensorflow as tf
from glob import glob
import os
from config import j, latent_shape
from encoder_decoder import encode
import numpy as np


latent_dim = latent_shape()


folder_path = j('Numpy')
ls = glob(j('Numpy/*.npy'))


def create_latent(fn):
    X = np.load(fn)
    tensor = tf.expand_dims(X, axis=0)
    encoded_tensor = encode(tensor)
    np.save(j(f'latent/{latent_dim}/{os.path.basename(fn)[6:]}'), encoded_tensor.numpy())


[create_latent(fn) for fn in ls]
