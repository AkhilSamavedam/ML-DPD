import numpy as np
import glob

ls = [np.load(fn) for fn in glob.glob('')]

combined_tensor = np.array(ls)

print(combined_tensor.shape)
