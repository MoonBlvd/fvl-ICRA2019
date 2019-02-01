import numpy as np

from keras import backend as K
from keras.layers import Lambda
from keras.activations import softmax

from keras.engine.topology import Layer

def zeros_like():
    return Lambda(lambda x: K.zeros_like(x))

def ones_like():
    return Lambda(lambda x: K.ones_like(x))

def expand_dim():
    return Lambda(lambda x: K.expand_dims(x, axis=1),
                  output_shape=lambda s: (s[0], 1, s[1]))

def mean():
    return Lambda(lambda x: K.mean(x, axis=-1))

def softmax_layer():
    return Lambda(lambda x: softmax(x, axis=1))

def prod_sum():
    return Lambda(lambda x: K.sum(x[0] * x[1], axis=1))

def tile_input(x, shape):
    return K.tile(x, shape)

def squeeze_tensor(x, axis):
    return K.squeeze(x, axis=axis)

def shifted_tanh(thresh):
    return Lambda(lambda x: (K.tanh(x) + 1) / 2 - thresh)

def query_tensor(x,i,dim):
    if dim == 3:
        return x[:,i,:]
    elif dim == 4:
        return x[:,i,:,:]
    elif dim == 5:
        return x[:,i,:,:,:]

def changeaxis(x, pattern):
    return K.permute_dimensions(x, pattern=pattern)

def sample_gaussian(args):
    mu, log_sigma, batch_size, hidden_size_z  = args
    eps = K.random_normal(shape=(batch_size, hidden_size_z), mean=0., std=1.)
    return mu + K.exp(log_sigma / 2) * eps

def sample_F_Bernoulli():
    # batch_size, hidden_size_z  = args
    # eps = K.random_binomial(shape=(batch_size, hidden_size_z), p = 0.5, seed=1)
    eps = K.random_binomial(shape=(64, 10), p = 0.5)

    return eps

class WeightedSum(Layer):
    ''' create a weighted sum layer with trainable parameters'''

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.batch_size = input_shape[0][0]
        self.param_1 = self.add_weight(name='kernel',
                                       shape=(input_shape[0][-1], self.output_dim),
                                       initializer='uniform',
                                       trainable=True)  # 512 * 100

        self.param_2 = self.add_weight(name='kernel',
                                       shape=(input_shape[1][-1], self.output_dim),
                                       initializer='uniform',
                                       trainable=True)  # 512 * 100

        super(WeightedSum, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        # batch_size  = K.shape(input[0])[0]
        outputs = []

        x1 = input[0]
        x2 = input[1]
        outputs = K.dot(x1, self.param_1) + K.dot(x2, self.param_2)

        return outputs
        # return np.dot(x1, self.param_1) + np.dot(x2, self.param_2) # 800*800

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

