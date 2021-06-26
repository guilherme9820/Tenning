from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda
from tenning.layers import Controller
from tenning.layers import NTMMemory
from tenning.generic_utils import get_object_config
import tensorflow as tf


# @tf.custom_gradient
# def custom_op(*args):

#     written_content, read_content, k_t, beta, g, shifts, gamma, erase, add = args

#     def grad(*grad_ys):
#         # print(f"GRADS: {grad_ys}")

#         grads = tf.gradients(read_content, [k_t, beta, g, shifts, gamma])

#         # grads = grads + tf.gradients(written_content, [erase, add])

#         print(grads)

#         return grads

#     return read_content, grad


class NTMWrapper(Layer):
    """An NTM Controller."""

    def __init__(self,
                 mem_rows,
                 word_length,
                 batch_size=1,
                 allowed_shifts=3,
                 trainable=True,
                 name='ntm_wrapper',
                 **kwargs):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super().__init__(name=name, trainable=trainable)

        self.allowed_shifts = allowed_shifts
        self.word_length = word_length
        self.mem_rows = mem_rows
        self.batch_size = batch_size

        self.controller = Controller(word_length, allowed_shifts, trainable=trainable)
        self.memory = NTMMemory(mem_rows, word_length)
        self.conv1d = tf.keras.layers.Conv1D(self.mem_rows, self.allowed_shifts, activation='softmax', padding='same')

        # self.update_address = Lambda(lambda x: x, output_shape=(self.mem_rows, ))

    def build(self, input_shape):

        # self.inital_value = tf.zeros([self.batch_size, self.mem_rows])
        # self.prev_address = self.update_address(tf.zeros([self.batch_size, self.mem_rows]))

        # self.initialization = self.add_weight(shape=[1], initializer=tf.constant_initializer(True), trainable=False)

        # self.prev_address = self.add_weight(shape=[self.batch_size, self.mem_rows], initializer=tf.zeros_initializer(), trainable=False)

        self.memory.reset()

    def call(self, input_tensor, training=True):

        # if tf.equal(self.initialization, True):
        #     self.prev_address = self.update_address(self.inital_value)
        #     self.initialization.assign([False])

        k_t, beta1, erase, add = self.controller(input_tensor)

        address = self.get_address(k_t, beta1)

        read_content = self.memory(address, erase, add, write_mode=training)

        # with tf.control_dependencies([read_content]):
        #     # Ensures that read task is executed
        #     # before updating prev_address
        #     self.prev_address = self.update_address(address)
        #     self.prev_address = temp_var.assign(address)

        # args = [written_content, read_content, k_t, beta, g, shifts, gamma, erase, add]

        return read_content
        #     tf.gradients(output, [k_t, beta, g, shifts, gamma, erase, add], stop_gradients=[k_t, beta, g, shifts, gamma, erase, add])
        # return tf.stop_gradient(input_tensor) * (1 - a * (1-eps)) +  x * (eps + a * (1 - eps))

    def get_address(self, k, β1, γ=None):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param prev_address: The weighting produced in the previous time step.
        """
        # Content focus
        similarity = self.memory.similarity(k)
        wc = tf.keras.activations.softmax(β1 * similarity)

        # Location focus
        # wl = self.memory.kl_div(k)
        # wl = tf.keras.activations.softmax(β2 * wl)

        # wg = self._interpolate(wc, wl, g)
        # ŵ = self._shift(wc)
        # w = self._sharpen(ŵ, γ)

        return wc

    # @tf.function
    # def _interpolate(self, wc, wl, g):
    #     return g * wc + (1. - g) * wl

    @tf.function
    def _shift(self, wg):

        # Creates a copy from wg tensor
        # new_w = tf.identity(wg)

        # for i in range(self.batch_size):
        #     # (1, num_rows)
        #     w = tf.slice(wg, [i, 0], [1, self.mem_rows])

        #     result = self._convolve(w, shifts)

        #     new_w = tf.tensor_scatter_nd_update(new_w, [[i]], result)

        # new_w = tf.map_fn(lambda x: self._convolve(x[0], x[1]), (wg, shifts), dtype=tf.float32)

        wg = tf.reshape(wg, [-1, 1, self.mem_rows])
        new_w = self.conv1d(wg)
        new_w = tf.squeeze(new_w, axis=[1])

        return new_w

    # @tf.function
    # def _convolve(self, tensor, kernel):
    #     # s.shape == [num_shifts]
    #     num_shifts = tf.size(kernel, out_type=tf.float32)

    #     # tensor.shape == [1, num_cols]
    #     tensor = tf.expand_dims(tensor, axis=0)

    #     allowed_neighbours = tf.cast(tf.floor(num_shifts * 0.5), tf.int32)

    #     # Converts from negative index to positive index of a circular buffer.
    #     # Example: if number of shifts is 5 then we must get the last two elements
    #     #          of the array, a pythonic way would be array[-2:]. So, performing
    #     #          mod division (considering num_cols == 8) we would get the positive
    #     #          index array[6:]
    #     starting_index = -allowed_neighbours % self.word_length

    #     # Equivalent to tensor[-allowed_neighbours:] if tensor were a python array
    #     left_filler = tf.slice(tensor, [0, starting_index], [1, allowed_neighbours])

    #     # Equivalent to tensor[:allowed_neighbours] if tensor were a python array
    #     right_filler = tf.slice(tensor, [0, 0], [1, allowed_neighbours])

    #     full_tensor = tf.concat([left_filler, tensor, right_filler], axis=-1)
    #     full_tensor = tf.reshape(full_tensor, [1, tf.size(full_tensor), 1])

    #     kernel = tf.reshape(kernel, [tf.size(kernel), 1, 1])

    #     # (1, num_cols, 1)
    #     result = tf.nn.conv1d(full_tensor, kernel, 1, 'VALID')

    #     # return tf.squeeze(result, axis=[-1])
    #     return tf.squeeze(result)

    @tf.function
    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = w / tf.reduce_sum(w + 1e-16, axis=1, keepdims=True)
        return w

    def writing_test(self):

        print("\tMEMORY WRITING TEST")

        idx = tf.range(self.mem_rows)
        address = tf.one_hot(idx, self.mem_rows)
        add_v = tf.one_hot(idx, self.word_length)

        for i in range(self.mem_rows):

            addr = tf.slice(address, [i, 0], [1, self.mem_rows])
            value = tf.slice(add_v, [i, 0], [1, self.word_length])

            erase = tf.ones([1, self.word_length])

            self.memory.write(addr, erase, value)

        print(self.memory.content)

    def reading_test(self, num_rows):

        print("\tMEMORY READING TEST")

        self.prev_address = tf.zeros([num_rows, self.mem_rows])

        idx = tf.range(num_rows)

        k = tf.one_hot(idx % self.mem_rows, self.word_length, dtype=tf.float32)

        β = 100.
        g = 1.
        s = tf.tile(tf.constant([[0., 1., 0.]]), [num_rows, 1])
        γ = 100.

        address = self.get_address(k, β, g, s, γ)

        value = self.memory.read(address)

        print(value)

    def get_config(self):

        config = super().get_config()

        config.update({'allowed_shifts': self.allowed_shifts,
                       'word_length': self.word_length,
                       'mem_rows': self.mem_rows,
                       'trainable': self.trainable,
                       'name': self.name,
                       'controller': get_object_config(self.controller),
                       'memory': get_object_config(self.memory)})

        return config


if __name__ == "__main__":

    mem_rows = 10
    word_length = 8
    input_shape = [1, 5]

    ntm_wrapper = NTMWrapper(mem_rows, word_length)
    ntm_wrapper.build(input_shape)

    ntm_wrapper.writing_test()

    ntm_wrapper.reading_test(1)
