import os
import re
import numpy as np
from config import j

def natural_sort_key(s):
    # Extract the numeric part from the filename using regex
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def read_latent_vectors(folder_path):
    latent_vectors = []
    ls = os.listdir(folder_path)
    ls.sort(key=natural_sort_key)
    print(ls)
    for filename in ls:
        if filename.endswith(".npy"):
            filepath = os.path.join(folder_path, filename)
            latent_vector = np.load(filepath)
            latent_vectors.append(latent_vector[0])
    return np.column_stack(latent_vectors)


def perform_edmd(observable_functions, dt=0.01):
    observable_functions = observable_functions[:, :int(0.8 * observable_functions.shape[1])]
    X = observable_functions[:, :-1]  # Snapshot matrix
    Y = observable_functions[:, 1:]   # Shifted snapshot matrix

    K_approximation = Y @ np.linalg.pinv(X)
    eigenvalues, eigenvectors = np.linalg.eig(K_approximation)

    frequencies = np.log(eigenvalues) / dt

    initial_condition = np.linalg.pinv(eigenvectors) @ observable_functions[:, 0]

    # Reconstruct the dynamic modes over time
    num_snapshots = observable_functions.shape[1]
    time_indices = np.arange(num_snapshots) * dt
    dynamic_modes = np.zeros((eigenvectors.shape[0], num_snapshots), dtype=np.complex64)
    for i in range(num_snapshots):
        dynamic_modes[:, i] = (eigenvectors * np.exp(frequencies * time_indices[i])) @ initial_condition

    return K_approximation, frequencies, eigenvectors, initial_condition, dynamic_modes


# Example usage (replace state_matrix with your actual input matrix):
state_matrix = read_latent_vectors(j('latent/256'))  # 128 features (variables), 22 snapshots

koopman_operator, frequencies, dynamic_modes, initial_condition, reconstructed_modes = perform_edmd(state_matrix)

# Print the Koopman operator K and its eigenvalues
print("Koopman Operator Approximation:")
print(koopman_operator)

np.save('results/256/koopman_alt.npy', koopman_operator)

print("Eigenvalues (Frequency):")
print(frequencies)

print("Eigenvectors (Dynamic Modes):")
print(dynamic_modes)

print("Initial Condition (Coefficients):")
print(initial_condition)

print("Reconstructed Dynamic Modes over Time:")
print(reconstructed_modes)
