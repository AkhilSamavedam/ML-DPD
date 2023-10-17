import jax.numpy as jnp
from jax import jit
import pandas as pd
import os
import multiprocessing as mp
import glob
from config import j

ls = glob.glob(j('CSV/*.csv'))

@jit
def create_tensor(filename):
    df = pd.read_csv(filename)
    x_set = set(df['x'])
    y_set = set(df['y'])

    x_list = jnp.array(sorted(list(x_set)))
    y_list = jnp.array(sorted(list(y_set)))

    tensor = jnp.zeros((len(x_list), len(y_list), 3))

    # Convert the 'x' and 'y' columns to arrays for faster indexing
    x_arr = jnp.array(df['x'].values)
    y_arr = jnp.array(df['y'].values)

    # Convert the 'vx', 'vy', and 'vz' columns to arrays for faster indexing
    vx_arr = jnp.array(df['vx'].values)
    vy_arr = jnp.array(df['vy'].values)
    vz_arr = jnp.array(df['vz'].values)

    # Calculate the indices for each point
    x_indices = jnp.searchsorted(x_list, x_arr)
    y_indices = jnp.searchsorted(y_list, y_arr)

    # Populate the tensor using vectorized indexing
    tensor[x_indices, y_indices, 0] = vx_arr
    tensor[x_indices, y_indices, 1] = vy_arr
    tensor[x_indices, y_indices, 2] = vz_arr
    
    name = os.path.basename(filename)
    

    jnp.save(j(f'Numpy/{name[:-4]}.npy'), tensor)

    jnp.save(j('legend/x_list.npy'), x_list)
    jnp.save(j('legend/y_list.npy'), y_list)
    return tensor, x_list, y_list

with mp.Pool() as pool:
    pool.map(create_tensor, ls)

