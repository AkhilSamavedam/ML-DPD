from config import j, latent_shape, tensor_shape
import tensorflow as tf


tf.config.experimental.enable_tensor_float_32_execution(enabled=True)


tensor_dim = tensor_shape()
latent_dim = latent_shape()

encoder = tf.keras.models.load_model(j(f'results/{latent_dim}/encoder.h5'))
decoder = tf.keras.models.load_model(j(f'results/{latent_dim}/decoder.h5'))

encode = tf.function(lambda x: encoder(x))
decode = tf.function(lambda x: decoder(x))

encode(tf.expand_dims(tf.ones(tensor_dim), axis=0))
decode(tf.expand_dims(tf.ones(latent_dim), axis=0))
