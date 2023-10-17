import tensorflow as tf
import jax.numpy as jnp
from glob import glob
import os
from config import j, dim
from utils import encode
from jax import jit

latent_dim = dim()

folder_path = j('Numpy')
ls = glob(j('Numpy/*.npy'))
@jit
def create_latent(fn):
    tensor = jnp.load(fn)
    tensor = jnp.expand_dims(tensor, axis=0)
    encoded_tensor = encode(tensor)
    jnp.save(j(f'latent/{latent_dim}/{os.path.basename(fn)[6:]}'), encoded_tensor)
[encode(fn) for fn in ls]
