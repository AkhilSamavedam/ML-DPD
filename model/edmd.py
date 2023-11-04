import os
import re
import tensorflow as tf
from config import j, latent_shape
from utils import natural_sort_key
import numpy as np

latent_dim = latent_shape()


def read_latent_vectors(folder_path):
    latent_vectors = []
    ls = os.listdir(folder_path)
    ls.sort(key=natural_sort_key)
    print(ls)
    for filename in ls:
        if filename.endswith(".npy"):
            filepath = os.path.join(folder_path, filename)
            latent_vector = np.load(filepath)
            latent_vectors.append(latent_vector[0]) # Remove Batch Dim
    return np.column_stack(latent_vectors)


def perform_edmd(observable_functions):
    observable_functions = observable_functions[:, :int(0.8 * observable_functions.shape[1])]
    X = tf.convert_to_tensor(observable_functions[:, :-1].T)  # Snapshot matrix
    Y = tf.convert_to_tensor(observable_functions[:, 1:].T)   # Shifted snapshot matrix

    K_approximation = tf.linalg.pinv(X) @ Y

    return K_approximation


# Example usage (replace state_matrix with your actual input matrix):
state_matrix = read_latent_vectors(j(f'latent/{latent_dim}'))  # latent_dim features (variables), n snapshots

koopman_operator = perform_edmd(state_matrix)

# Print the Koopman operator K and its eigenvalues
print("Koopman Operator Approximation:")
print(koopman_operator)

np.save(j(f'results/{latent_dim}/koopman.npy'), koopman_operator.numpy())
