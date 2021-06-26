from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tenning.generic_utils import get_object_config
from tenning.layers import FireModule
import tensorflow as tf


class Controller(Layer):
    """An NTM Controller."""

    def __init__(self,
                 word_length,
                 allowed_shifts=3,
                 trainable=True,
                 initializer='he_normal',
                 name='controller',
                 **kwargs):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super().__init__(name=name, trainable=trainable)

        assert allowed_shifts % 2 != 0, f"'allowed_shifts' must be an odd number!"

        self.allowed_shifts = allowed_shifts
        self.word_length = word_length

        # Corresponds to k_t, beta1, g, gamma
        internal_state_size = tf.cast(tf.reduce_sum([word_length, 1]), tf.int32)

        self.internal_state = Dense(internal_state_size, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/k_t')
        self.erase = Dense(word_length, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/erase')
        self.add = Dense(word_length, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/add')

        self.gram_conv1 = Conv2D(word_length, kernel_size=1, strides=1, name=self.name + "/gram_conv1", padding='valid',
                                 trainable=self.trainable, kernel_initializer=initializer)

        self.gram_conv2 = Conv2D(word_length, kernel_size=1, strides=1, name=self.name + "/gram_conv2", padding='valid',
                                 trainable=self.trainable, kernel_initializer=initializer)

        # self.k_t = Dense(word_length, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/k_t')
        # self.beta = Dense(1, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/beta')
        # self.g = Dense(1, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/g')
        # self.shifts = Dense(allowed_shifts, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/shifts')
        # self.gamma = Dense(1, kernel_initializer=initializer, trainable=self.trainable, name=self.name + '/gamma')

    def build(self, input_shape):

        self.fire_module1 = FireModule(input_shape[-1] // 4, input_shape[-1] // 2, input_shape[-1] // 2, name=self.name + "/fire_module1")
        self.fire_module2 = FireModule(input_shape[-1] // 2, input_shape[-1], input_shape[-1], name=self.name + "/fire_module2")

    def call(self, input_tensor):

        fire_module1 = self.fire_module1(input_tensor)

        gram1 = self.gram_module(self.gram_conv1, fire_module1)

        fire_module2 = self.fire_module2(fire_module1)

        gram2 = self.gram_module(self.gram_conv2, fire_module2)

        grams = tf.concat([gram1, gram2], axis=-1)

        flatten_grams = Flatten()(grams)

        erase = self.erase(flatten_grams)
        add = self.add(flatten_grams)

        flatten_fire = Flatten()(fire_module2)

        control = self.internal_state(tf.concat([flatten_fire, erase, add], axis=-1))

        # k_t = self.k_t(control)
        k_t = tf.slice(control, [0, 0], [-1, self.word_length])
        beta = tf.slice(control, [0, self.word_length], [-1, 1])
        # beta = self.beta(control)
        # g = self.g(control)
        # gamma = self.gamma(control)
        # shifts = tf.slice(control, [0, self.word_length + 4], [-1, self.allowed_shifts])

        # k_t = self.k_t(input_tensor)
        beta = tf.math.softplus(beta)
        # g = tf.math.sigmoid(g)
        # shifts = tf.keras.activations.softmax(shifts)
        # gamma = 1. + tf.math.softplus(gamma)

        # return k_t, beta, g, shifts, gamma, erase, add
        return k_t, beta, erase, add

    def gram_module(self, op, tensor):

        gram_conv = op(tensor)

        gram_act = tf.math.tanh(gram_conv)

        flatten = self._flatten_tensor(gram_act)

        return tf.matmul(flatten, flatten, transpose_b=True)

    def _flatten_tensor(self, x):
        """ Returns a tensor with shape [batch, channels, N], where N is the number of features"""

        batch = tf.shape(x)[0]
        channels = tf.shape(x)[-1]

        return tf.reshape(x, shape=[batch, channels, -1])

    def get_config(self):

        config = super().get_config()

        config.update({'allowed_shifts': self.allowed_shifts,
                       'word_length': self.word_length,
                       'trainable': self.trainable,
                       'name': self.name,
                       'internal_state': get_object_config(self.internal_state),
                       # 'k_t': get_object_config(self.k_t),
                       # 'beta': get_object_config(self.beta),
                       # 'g': get_object_config(self.g),
                       # 'shifts': get_object_config(self.shifts),
                       # 'gamma': get_object_config(self.gamma),
                       'erase': get_object_config(self.erase),
                       'add': get_object_config(self.add)})

        return config
