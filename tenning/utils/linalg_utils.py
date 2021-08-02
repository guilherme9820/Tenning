import matplotlib.pyplot as plt
from typing import Union
import tensorflow as tf
import numpy as np
import sys


def adjugate(tensor):

    det = tf.linalg.det

    sub1 = submatrix(tensor, row=0, column=0)
    sub2 = submatrix(tensor, row=0, column=1)
    sub3 = submatrix(tensor, row=0, column=2)
    sub4 = submatrix(tensor, row=1, column=0)
    sub5 = submatrix(tensor, row=1, column=1)
    sub6 = submatrix(tensor, row=1, column=2)
    sub7 = submatrix(tensor, row=2, column=0)
    sub8 = submatrix(tensor, row=2, column=1)
    sub9 = submatrix(tensor, row=2, column=2)

    cofactor = tf.convert_to_tensor([[det(sub1), -det(sub2), det(sub3)],
                                     [-det(sub4), det(sub5), -det(sub6)],
                                     [det(sub7), -det(sub8), det(sub9)]])

    return tf.transpose(cofactor)


def submatrix(tensor: Union[np.ndarray, tf.Tensor], row: int, column: int) -> tf.Tensor:
    """ A submatrix of a matrix is obtained by deleting any collection of rows and/or columns.
        For example, from the following 3-by-3 matrix, we can construct a 2-by-2 submatrix by
        removing row 3 and column 2:

            |1   2   3|
        A = |4   5   6| -> |1   2|
            |7   8   9|    |4   5|
    Args:
        tensor: A tensor whose inner-most two dimensions form a square matrix.
        row: The row index to be removed from input matrix.
        column: The column index to be removed from input matrix.

    Returns:
        Returns the submatrix of a square matrix after excluding the ith row and the kth column.

    References:
        - [Submatrix] (https://en.wikipedia.org/wiki/Matrix_(mathematics)#Submatrix)
    """

    if len(tensor.shape) < 3:
        tensor = tensor[tf.newaxis, ...]

    batch_size = tensor.shape[0]
    ndim = tensor.shape[-1]

    batch_indices = tf.range(batch_size)[:, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, (ndim-1)**2])
    batch_indices = tf.reshape(batch_indices, [-1, 1])

    x_indices = tf.where(tf.math.not_equal(tf.range(ndim), row))
    x_indices = tf.cast(x_indices, tf.int32)

    y_indices = tf.where(tf.math.not_equal(tf.range(ndim), column))
    y_indices = tf.cast(y_indices, tf.int32)

    x_indices = tf.reshape(tf.tile(x_indices, [1, ndim-1]), [-1, 1])
    y_indices = tf.tile(y_indices, [ndim-1, 1])

    indices = tf.concat([x_indices, y_indices], axis=1)
    indices = tf.tile(indices, [batch_size, 1])

    indices = tf.concat([batch_indices, indices], axis=1)

    minor = tf.gather_nd(tensor, indices)

    return tf.reshape(minor, [batch_size, ndim-1, ndim-1])


class PCA2DSquared:
    """ Implementation of 2D²PCA from 'Two-directional two-dimensional PCA for efficient face representation and recognition' paper"""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def get_k_top_eigen(self, matrix):

        eigenvals, eigenvecs = tf.linalg.eigh(matrix)

        # Flip values because tf.linalg.eigh returns the values in ascending order
        # and we want the values in descending order of importance
        eigenvals = tf.reverse(eigenvals, axis=[0])
        eigenvecs = tf.reverse(eigenvecs, axis=[1])

        cumsum = tf.cumsum(eigenvals) / tf.reduce_sum(eigenvals)

        # Gets the number of singular values that have a cumulative
        # energy at least equal or above the specified threshold
        quantity = tf.where(cumsum >= self.threshold)[0][0]

        # Return the reconstructed image in a compressed form
        return eigenvals[:quantity], eigenvecs[:, :quantity]

    def plot_eigen_energy(self, eigenvalues):

        x_axis = np.arange(len(eigenvalues))

        cumsum = np.cumsum(eigenvalues)

        plt.figure(0)
        plt.subplot(121)
        plt.semilogy(x_axis, eigenvalues)
        plt.subplot(122)
        plt.plot(x_axis, cumsum / np.sum(eigenvalues))
        plt.show()

    def unfit(self, images):
        unfit_cols = tf.matmul(images, self.g_eigenvecs, transpose_b=True)

        return tf.matmul(self.h_eigenvecs, unfit_cols)

    def fit(self, images):

        # Takes the mean image Î ∈ ℝ^{mxn} from I s.t. I_{i} ∈ ℝ^{mxn}
        mean_image = np.mean(images, axis=0, keepdims=True)

        # Subtracts the mean image from all images (samples, m, n)
        K = (images - mean_image).astype(np.float32)

        # Column covariance (n, n)
        G_t = tf.reduce_sum(tf.matmul(K, K, transpose_a=True), axis=0)

        # Row covariance (m, m)
        H_t = tf.reduce_sum(tf.matmul(K, K, transpose_b=True), axis=0)

        # Top p eigenvectors
        g_eigenvals, g_eigenvecs = self.get_k_top_eigen(G_t)

        self.g_eigenvecs = g_eigenvecs

        # Top q eigenvectors
        h_eigenvals, h_eigenvecs = self.get_k_top_eigen(H_t)

        self.h_eigenvecs = h_eigenvecs

        # (samples, m, p)
        col_transform = tf.matmul(images.astype(np.float32), g_eigenvecs)

        # (samples, q, p)
        transform = tf.matmul(h_eigenvecs, col_transform, transpose_a=True)

        return transform.numpy()


def cross(a, b):
    a_x = tf.slice(a, [0, 0], [-1, 1])
    a_y = tf.slice(a, [0, 1], [-1, 1])
    a_z = tf.slice(a, [0, 2], [-1, 1])

    b_x = tf.slice(b, [0, 0], [-1, 1])
    b_y = tf.slice(b, [0, 1], [-1, 1])
    b_z = tf.slice(b, [0, 2], [-1, 1])

    cross_x = a_y*b_z - a_z*b_y
    cross_y = a_z*b_x - a_x*b_z
    cross_z = a_x*b_y - a_y*b_x

    res = tf.concat([cross_x, cross_y, cross_z], axis=1)

    # res = tf.convert_to_tensor([a_y*b_z - a_z*b_y,
    #                             a_z*b_x - a_x*b_z,
    #                             a_x*b_y - a_y*b_x])

    # res = tf.convert_to_tensor([a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
    #                             a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
    #                             a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]])

    # return tf.transpose(res)
    return res


def sym_matrix_from_array(array):

    # Gets the number of elements in the inner most dimension
    original_shape = tf.shape(array)
    num_values = original_shape[-1]

    array = tf.reshape(array, [-1, num_values])

    batch_size = tf.shape(array)[0]

    batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, num_values, 1])

    # evaluates the matrix dimension based on the length of the input array
    matrix_dim = 0.5*(tf.math.sqrt(tf.cast(8*num_values + 1, 'float32')) - 1.)
    matrix_dim = tf.cast(matrix_dim, 'int32')

    symm_matrix = tf.zeros([batch_size, matrix_dim, matrix_dim])

    row_idx, col_idx = tf.meshgrid(tf.range(matrix_dim), tf.range(matrix_dim))

    upper_idx = tf.where(row_idx >= col_idx)
    lower_idx = tf.roll(upper_idx, shift=1, axis=-1)[tf.newaxis, ...]

    upper_idx = upper_idx[tf.newaxis, ...]
    upper_idx = tf.cast(tf.tile(upper_idx, [batch_size, 1, 1]), 'int32')
    upper_idx = tf.concat([batch_indices, upper_idx], axis=-1)

    lower_idx = tf.cast(tf.tile(lower_idx, [batch_size, 1, 1]), 'int32')
    lower_idx = tf.concat([batch_indices, lower_idx], axis=-1)

    symm_matrix = tf.tensor_scatter_nd_update(symm_matrix, upper_idx, array)
    symm_matrix = tf.tensor_scatter_nd_update(symm_matrix, lower_idx, array)

    new_shape = tf.concat([original_shape[:-1], [matrix_dim, matrix_dim]], axis=0)

    return tf.reshape(symm_matrix, new_shape)


if __name__ == "__main__":

    tensor = np.random.rand(2, 3, 3)
    row = int(sys.argv[1])
    col = int(sys.argv[2])

    minor = submatrix(tensor, row, col)

    print(f"tensor: {tensor}")
    print(f"minor from [{row}, {col}]: {minor}")
