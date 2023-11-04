import haiku as hk
import graphviz
from encoder_decoder import encode, decode
from config import latent_shape, j
import jax.numpy as jnp
import os

def plot_model(name, model, x):
    dot = hk.experimental.to_dot(model)(x)
    graph = graphviz.Source(dot)
    graph.render(j(f'results/{latent_shape()}/{name}'), view=False)
    os.remove(j(f'results/{latent_shape()}/{name}'))


plot_model('encoder', encode, jnp.zeros((1, 4200, 244, 3)))
plot_model('decoder', decode, jnp.zeros((1, latent_shape())))