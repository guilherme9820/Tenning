from tensorflow.keras.layers import Layer
import tensorflow as tf


# @tf.custom_gradient
# def write_op(*args):

#     print('WRITE OP')

#     address, content, erase, add = args

#     # (batch, mem_rows) -> (mem_rows, batch)
#     address = tf.transpose(address)

#     # Return (mem_rows, word_length)
#     erase_v = tf.matmul(address, erase)
#     add_v = tf.matmul(address, add)

#     new_content = content * (1 - erase_v) + add_v

#     def grad(*grad_ys):
#         print(f"GRADS: {grad_ys}")

#         grads = tf.gradients(new_content, [erase, add])

#         return grads

#     return new_content, grad


# @tf.custom_gradient
# def read_op(*args):

#     print('READ OP')

#     address, content = args

#     # (batch, mem_rows) -> (batch, 1, mem_rows)
#     exp_address = tf.expand_dims(address, axis=1)

#     # (batch, 1, word_length)
#     results = tf.matmul(exp_address, content)

#     def grad(*grad_ys):

#         grads = tf.gradients(results, [address], stop_gradients=[address])

#         return grads

#     # Squeezes the one dimensional component of the 'results' tensor
#     return tf.squeeze(results, axis=[1]), grad


class NTMMemory(Layer):
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

    def build(self, input_shape):

        self.reset()

    def reset(self):
        """Initializes memory content."""

        self.content = self.add_weight(shape=[self.mem_rows, self.word_length], initializer=tf.constant_initializer(1e-6),
                                       trainable=self.trainable, name=self.name + '/content')

    def call(self, address, erase, add, write_mode=False):

        if write_mode:
            self.write(address, erase, add)

        # read_content = read_op(address, self.content)
        read_content = self.read(address)

        # content_error = tf.stop_gradients(new_content - read_content)

        return read_content

    def read(self, address):

        # (batch, mem_rows) -> (batch, 1, mem_rows)
        address = tf.expand_dims(address, axis=1)

        # (batch, 1, word_length)
        results = tf.matmul(address, self.content)

        # Squeezes the one dimensional component of the 'results' tensor
        return tf.squeeze(results, axis=[1])

    def write(self, address, erase, add):

        # # (batch, mem_rows) -> (mem_rows, batch)
        # address = tf.transpose(address)

        # Return (mem_rows, word_length)
        erase_v = tf.matmul(address, erase, transpose_a=True)
        add_v = tf.matmul(address, add, transpose_a=True)

        new_content = self.content * (1 - erase_v) + add_v

        self.content.assign(new_content)

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


if __name__ == "__main__":

    memory = NTMMemory(mem_rows=10, word_length=8, batch_size=2)

    test_writing(memory, memory.mem_rows, memory.word_length, memory.batch_size)

    test_reading(memory, memory.mem_rows, memory.word_length, memory.batch_size)
