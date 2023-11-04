from jax import jit
import tensorflow as tf
from timeit import timeit
from plot_model import plot_model
import jax.numpy as jnp
import tf2jax
import numpy as np

A = tf.ones((256, 256), dtype=tf.float32)
j = tf.constant(0)
B = jnp.ones((256, 256), dtype=np.float32)

func = lambda x: tf.linalg.pinv((tf.linalg.pinv(x) + 1) @ x)

xla_func = tf.function(func)

xla_func(A)

converted_func, params = tf2jax.convert(xla_func, B)

jax_func = lambda x: converted_func(params, x)

n = 20

print(1000 * timeit('xla_func(A)', 'from __main__ import xla_func, func, A', number=n) / n)

plot_model('test', jax_func, B)