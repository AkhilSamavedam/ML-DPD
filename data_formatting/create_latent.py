import tensorflow as tf
import numpy as np
import glob
import os
from config import j

folder_path = j('Numpy')
encoder_model_path = j('results/256/encoder.h5')
encoder_model = tf.keras.models.load_model(encoder_model_path)
ls = glob.glob(j('Numpy/*.npy'))
def encode(fn): 
    tensor = np.load(fn)
    tensor = tf.expand_dims(tensor, axis=0)
    encoded_tensor = encoder_model.predict(tensor)
    np.save(j(f'latent/256/{os.path.basename(fn)[6:]}'), encoded_tensor)
[encode(fn) for fn in ls]
