from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tenning.generic_utils import get_object_config
import tensorflow as tf


class SVDO(Layer):
    """ Performs symmetric orthogonalization as detailed in the paper
        'An Analysis of SVD for Deep Rotation Estimation'
        (https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf)

        This implementation was taken from its original implementation at
        (https://github.com/google-research/google-research/tree/master/special_orthogonalization)
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def call(self, input_tensor):

        # Reshapes a (batch, 9) tensor to a (batch, 3, 3) tensor.
        input_tensor = tf.reshape(input_tensor, (-1, 3, 3))

        _, u, v = tf.linalg.svd(input_tensor)

        det = tf.linalg.det(tf.matmul(u, v, transpose_b=True))

        output = tf.matmul(
            tf.concat([u[:, :, :-1], u[:, :, -1:] * tf.reshape(det, [-1, 1, 1])], 2),
            v, transpose_b=True)

        return output

    def get_config(self):

        config = super().get_config()

        config.update({'trainable': self.trainable,
                       'name': self.name})

        return config
