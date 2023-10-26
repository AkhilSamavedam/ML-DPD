from config import j, latent_shape, tensor_shape
import cloudpickle as cpkl
import pickle as pkl

tensor_dim = tensor_shape()
latent_dim = latent_shape()

with open(j(f'results/{latent_dim}/encoder.hk'), 'rb') as ef:
    encode = cpkl.load(ef)

with open(j(f'results/{latent_dim}/decoder.hk'), 'rb') as df:
    decode = cpkl.load(df)