import numpy as np
import matplotlib.pyplot as plt
from config import j
from glob import glob
import seaborn as sns
import cmasher as cmr

ls = glob(j('Numpy/*.npy'))

data = np.load(ls[0])

sns.heatmap(data=np.mean(data, axis=2), square=True, xticklabels=100, yticklabels=500, vmin=0, vmax=1.5, cmap=cmr.neon)

plt.savefig(fname=j('test.pdf'), dpi=600)

plt.show()

