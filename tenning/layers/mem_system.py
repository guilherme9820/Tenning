from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tenning.layers import NTMMemory_v2
from tenning.generic_utils import get_object_config
import tensorflow as tf


class MemSystem(Layer):
    """An NTM Controller."""

    def __init__(self,
                 mem_rows,
                 word_length,
                 initializer='he_normal',
                 allowed_shifts=3,
                 trainable=True,
                 **kwargs):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super().__init__(trainable=trainable, **kwargs)

        self.allowed_shifts = allowed_shifts
        self.word_length = word_length
        self.mem_rows = mem_rows

        self.writing_value = Dense(word_length, kernel_intializer=initializer, name=self.name + "/writing_value")
        self.beta = Dense(1, kernel_intializer=initializer, activation='softplus', name=self.name + "/beta")
        self.gamma = Dense(1, kernel_intializer=initializer, activation='softplus', name=self.name + "/gamma")
        self.pred_address = Dense(mem_rows, kernel_intializer=initializer, name=self.name + "/pred_address")
        self.trainable_memory = NTMMemory_v2(mem_rows, word_length, trainable=True, name=self.name + "/trainable_memory")
        self.nontrainable_memory = NTMMemory_v2(mem_rows, word_length, trainable=False, name=self.name + "/nontrainable_memory")
        self.conv1d = Conv1D(1, self.allowed_shifts, padding='same')

    def build(self, input_shape):

        self.trainable_memory.reset()
        self.nontrainable_memory.reset()

    def call(self, input_tensor, training=True):

        writing_value = self.writing_value(input_tensor)
        beta = self.beta(input_tensor)
        gamma = 1. + self.gamma(input_tensor)

        address = self.get_address(writing_value, beta, gamma)

        self.nontrainable_memory.write(address, writing_value)
        self.trainable_memory.write(address, writing_value)

        predicted_address = self.pred_address(input_tensor)

        read_values = self.trainable_memory(predicted_address)

        return read_values

    def get_address(self, writing_value, beta, gamma):

        similarity = self.nontrainable_memory.similarity(writing_value)

        address = tf.keras.activations.softmax(beta * similarity)

        address = tf.reshape(address, [-1, self.mem_rows, 1])

        address = self.conv1d(address)

        address = tf.squeeze(address, axis=[-1])

        address = address ** gamma

        address = address / tf.reduce_sum(address + 1e-16, axis=1, keepdims=True)

        return address

    def get_config(self):

        config = super().get_config()

        config.update({'allowed_shifts': self.allowed_shifts,
                       'word_length': self.word_length,
                       'mem_rows': self.mem_rows,
                       'trainable': self.trainable,
                       'name': self.name,
                       'writing_value': get_object_config(self.writing_value),
                       'pred_address': get_object_config(self.pred_address),
                       'beta': get_object_config(self.beta),
                       'gamma': get_object_config(self.gamma),
                       'nontrainable_memory': get_object_config(self.nontrainable_memory),
                       'trainable_memory': get_object_config(self.trainable_memory),
                       'conv1d': get_object_config(self.conv1d)})

        return config
