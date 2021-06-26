from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tenning.generic_utils import get_object_config
import tensorflow as tf


class FireModule(Layer):
    """An NTM Controller."""

    def __init__(self,
                 s1,  # Number of filters (1x1) of squeeze layer
                 e1,  # Number of filters (1x1) of expand layer
                 e3,  # Number of filters (3x3) of expand layer
                 trainable=True,
                 initializer='he_normal',
                 name='fire_module',
                 **kwargs):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super().__init__(name=name, trainable=trainable)

        assert s1 < (e1 + e3), f"'s1' parameter should me less than 'e1' + 'e3'"

        self.s1 = s1
        self.e1 = e1
        self.e3 = e3

        self.squeeze1 = Conv2D(s1, kernel_size=1, strides=1, name=self.name + "/s1", padding='valid',
                               trainable=self.trainable, kernel_initializer=initializer)
        self.expand1 = Conv2D(e1, kernel_size=1, strides=1, name=self.name + "/e1", padding='valid',
                              trainable=self.trainable, kernel_initializer=initializer)
        self.expand3 = Conv2D(e3, kernel_size=3, strides=1, name=self.name + "/e3", padding='same',
                              trainable=self.trainable, kernel_initializer=initializer)

    def call(self, input_tensor):

        squeeze1 = self.squeeze1(input_tensor)

        expand1 = self.expand1(squeeze1)

        expand3 = self.expand3(squeeze1)

        return tf.concat([expand1, expand3], axis=-1)

    def get_config(self):

        config = super().get_config()

        config.update({'s1': self.s1,
                       'e1': self.e1,
                       'e3': self.e3,
                       'trainable': self.trainable,
                       'name': self.name,
                       'squeeze1': get_object_config(self.squeeze1),
                       'expand1': get_object_config(self.expand1),
                       'expand3': get_object_config(self.expand3)})

        return config
