import os.path
import sys
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(enabled=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def j(rel_path):
    return os.path.join(PROJECT_DIR, rel_path)


def latent_shape():
    return 256


def tensor_shape():
    return 4200, 244, 3
