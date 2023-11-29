import numpy as np
import matplotlib.pyplot as plt
from config import j
from glob import glob
import seaborn as sns
import cmasher as cmr
import tensorflow as tf
from timeit import timeit

ls = glob(j('Numpy/*.npy'))

data = tf.convert_to_tensor([np.load(i) for i in ls])

# data has shape of (32, 4200, 244, 3)
