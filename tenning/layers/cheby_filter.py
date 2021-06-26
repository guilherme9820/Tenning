from tenning.data_utils import rescale_laplacian
from tenning.generic_utils import eye_like
from tensorflow.keras.layers import Layer
from scipy import sparse
import tensorflow as tf
import healpy as hp
import numpy as np


class ChebyConv(Layer):

    def __init__(self, laplacian, out_channels, polynomial_degree, add_bias=True, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.out_channels = out_channels
        self.polynomial_degree = polynomial_degree
        self.add_bias = add_bias

        laplacian = tf.identity(laplacian)  # Copies laplacian matrix
        eigenvals = tf.linalg.eigvalsh(laplacian)
        lmax = 1.02 * eigenvals[-1]  # Keeps only the largest eigenvalue

        laplacian = rescale_laplacian(laplacian, lmax=lmax, scale=0.75)
        self.laplacian = tf.sparse.from_dense(laplacian)

    def build(self, input_shape):

        shape = [input_shape[-1] * self.polynomial_degree, self.out_channels]
        stddev = 1 / tf.sqrt(input_shape[-1] * (self.polynomial_degree + 0.5) / 2)
        initial = tf.keras.initializers.TruncatedNormal(mean=0., stddev=stddev)
        self.cheby_weights = self.add_weight(shape=shape, initializer=initial, trainable=self.trainable, name=self.name + '/cheby_weights')

        if self.add_bias:
            initial = tf.keras.initializers.Zeros()
            self.cheby_bias = self.add_weight(shape=[1, 1, self.out_channels], initializer=initial, trainable=self.trainable, name=self.name + '/cheby_bias')

    def call(self, input_tensor):
        features = tf.shape(input_tensor)[1]
        in_channels = tf.shape(input_tensor)[-1]
        # Transform to Chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # features x in_channels x batch_size
        x0 = tf.reshape(x0, [features, -1])  # features x in_channels*batch_size
        x = tf.expand_dims(x0, axis=0)  # 1 x features x in_channels*batch_size

        x1 = tf.sparse.sparse_dense_matmul(self.laplacian, x0)
        x = self._concat(x, x1)

        degree = tf.constant(2)
        while tf.less(degree, self.polynomial_degree):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.laplacian, x1) - x0  # features x in_channels*batch_size
            x = self._concat(x, x2)
            x0, x1 = x1, x2
            degree += 1

        x = tf.reshape(x, [self.polynomial_degree, features, in_channels, -1])  # K x features x in_channels x batch_size
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # batch_size x features x in_channels x K
        x = tf.reshape(x, [-1, in_channels * self.polynomial_degree])  # batch_size*features x in_channels*K
        # Filter: in_channels*Fout out_channels of order K, i.e. one filterbank per output feature.
        x = tf.matmul(x, self.cheby_weights)  # batch_size*features x Fout
        result = tf.reshape(x, [-1, features, self.out_channels])  # batch_size x features x Fout

        if self.add_bias:
            result += self.cheby_bias

        return result

    def _concat(self, x, x_):
        x_ = tf.expand_dims(x_, axis=0)  # 1 x features x in_channels*batch_size
        return tf.concat([x, x_], axis=0)  # K x features x in_channels*batch_size

    def get_config(self):

        config = super().get_config()

        config.update({'out_channels': self.out_channels,
                       'polynomial_degree': self.polynomial_degree,
                       'add_bias': self.add_bias})

        return config
