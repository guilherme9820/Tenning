from tensorflow.keras.layers import Layer
import tensorflow as tf


class NTMMemory_v2(Layer):
    """Memory bank for NTM."""

    def __init__(self, mem_rows, word_length, trainable=False, **kwargs):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super().__init__(trainable=trainable, **kwargs)

        self.mem_rows = mem_rows
        self.word_length = word_length

        self.content = self.add_weight(shape=[self.mem_rows, self.word_length], initializer=tf.constant_initializer(1e-6),
                                       trainable=self.trainable, name=self.name + '/content')

    def reset(self):
        """Resets to default value"""

        self.content.assign(tf.constant_initializer(1e-6)([self.mem_rows, self.word_length]))

    def call(self, address):

        read_content = self.read(address)

        return read_content

    def read(self, address):

        # (batch, mem_rows) -> (batch, 1, mem_rows)
        address = tf.expand_dims(address, axis=1)

        # (batch, 1, word_length)
        results = tf.matmul(address, self.content)

        # Squeezes the one dimensional component of the 'results' tensor
        return tf.squeeze(results, axis=[1])

    def write(self, address, value):

        # Return (mem_rows, word_length)
        value = tf.matmul(address, value, transpose_a=True)

        self.content.assign(value)

    def similarity(self, k_t):

        # from (batch, word_length) to (batch, 1, word_length)
        k_t = tf.expand_dims(k_t, axis=1)

        # The cosine simimilarity from tensorflow is a loss function, so dissimilatity is
        # returned as a positive value while sililarity as negative value. Hence, we must
        # multiply by -1 to invert the orientations
        return -tf.keras.losses.cosine_similarity(k_t + 1e-16, self.content + 1e-16)

    def scaled_dot(self, k_t):

        # from (batch, word_length) to (batch, 1, word_length)
        k_t = tf.expand_dims(k_t, axis=1)

        scaled_dot = tf.matmul(self.content, k_t, transpose_b=True) / tf.math.sqrt(float(self.word_length))

        return tf.squeeze(scaled_dot, axis=[2])

    def kl_div(self, k_t):

        # from (batch, word_length) to (batch, 1, word_length)
        k_t = tf.expand_dims(k_t, axis=1)

        kl_div = tf.keras.losses.kullback_leibler_divergence(k_t, self.content)

        return 1. / (kl_div + 1e-7)

    def get_config(self):

        config = super().get_config()

        config.update({'mem_rows': self.mem_rows,
                       'word_length': self.word_length,
                       'name': self.name,
                       'trainable': self.trainable})

        return config
