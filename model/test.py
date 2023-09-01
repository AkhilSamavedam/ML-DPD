import numpy as np
import glob
import tensorflow as tf

ls = [np.load(fn) for fn in glob.glob('')]

combined_tensor = np.array(ls)

print(tf.config.list_physical_devices('GPU'))

