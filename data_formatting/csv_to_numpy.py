import numpy as np
from jax import jit
import pandas as pd
import os
import multiprocessing as mp
import glob
from config import j

ls = glob.glob(j('CSV/*.csv'))

def create_tensor(filename):
    df = pd.read_csv(filename)
    x_set = set(df['x'])
    y_set = set(df['y'])

    x_list = np.array(sorted(list(x_set)))
    y_list = np.array(sorted(list(y_set)))

    tensor = np.zeros((len(x_list), len(y_list), 3))

    # Convert the 'x' and 'y' columns to arrays for faster indexing
    x_arr = np.array(df['x'].values)
    y_arr = np.array(df['y'].values)

    # Convert the 'vx', 'vy', and 'vz' columns to arrays for faster indexing
    vx_arr = np.array(df['vx'].values)
    vy_arr = np.array(df['vy'].values)
    vz_arr = np.array(df['vz'].values)

    # Calculate the indices for each point
    x_indices = np.searchsorted(x_list, x_arr)
    y_indices = np.searchsorted(y_list, y_arr)

    # Populate the tensor using vectorized indexing
    tensor[x_indices, y_indices, 0] = vx_arr
    tensor[x_indices, y_indices, 1] = vy_arr
    tensor[x_indices, y_indices, 2] = vz_arr
    
    name = os.path.basename(filename)
    

    np.save(j(f'Numpy/{name[:-4]}.npy'), tensor)

    np.save(j('legend/x_list.npy'), x_list)
    np.save(j('legend/y_list.npy'), y_list)
    return tensor, x_list, y_list

#with mp.Pool() as pool:
#    pool.map(create_tensor, ls)
[create_tensor(i) for i in ls]
