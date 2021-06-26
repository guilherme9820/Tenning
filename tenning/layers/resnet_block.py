import tensorflow.keras.constraints as constraints
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.layers import GroupNormalization
from tenning.generic_utils import get_object_config
from tenning.activations import Swish
import tensorflow as tf


class ResnetBlock(Layer):

    def __init__(self,
                 out_channels,
                 strides=1,
                 kernel_size=3,
                 trainable=True,
                 mode='identity',
                 initializer='he_normal',
                 normalization='instance_norm',
                 activation='leaky_relu',
                 groups=None,
                 squeeze_excitation=False,
                 squeeze_ratio=16,
                 **kwargs):

        super().__init__(trainable=trainable, **kwargs)

        allowed_normalizations = ['batch_norm', 'instance_norm', 'group_norm']

        allowed_modes = ['identity', 'downsample', 'upsample']

        assert mode in allowed_modes, f"Invalid mode!"
        assert normalization in allowed_normalizations, f"Invalid normalization!"

        conv_constraint = kwargs.get('conv_constraint', None)
        conv_constraint_arguments = kwargs.get('conv_constraint_arguments', [])

        dense_constraint = kwargs.get('dense_constraint', None)
        dense_constraint_arguments = kwargs.get('dense_constraint_arguments', [])

        if conv_constraint_arguments:
            if not isinstance(conv_constraint_arguments, list):
                raise TypeError(f"'conv_constraint_arguments' must be a list")

        if dense_constraint_arguments:
            if not isinstance(dense_constraint_arguments, list):
                raise TypeError(f"'dense_constraint_arguments' must be a list")

        if conv_constraint:
            conv_constraint = getattr(constraints, conv_constraint, None)(*conv_constraint_arguments)

        if dense_constraint:
            dense_constraint = getattr(constraints, dense_constraint, None)(*dense_constraint_arguments)

        self.out_channels = out_channels
        self.initializer = initializer
        self.mode = mode
        self.kernel_size = kernel_size
        self.strides = strides
        self.normalization = normalization
        self.groups = groups
        self.squeeze_excitation = squeeze_excitation
        self.squeeze_ratio = squeeze_ratio
        self.conv_constraint = conv_constraint
        self.dense_constraint = dense_constraint

        if normalization == 'group_norm':
            self.norm1 = GroupNormalization(groups=self.groups, name=self.name + '/norm1', trainable=self.trainable)
            self.norm2 = GroupNormalization(groups=self.groups, name=self.name + '/norm2', trainable=self.trainable)
            self.norm3 = GroupNormalization(groups=self.groups, name=self.name + '/norm3', trainable=self.trainable)

        elif normalization == 'instance_norm':
            self.norm1 = InstanceNormalization(name=self.name + '/norm1', trainable=self.trainable)
            self.norm2 = InstanceNormalization(name=self.name + '/norm2', trainable=self.trainable)
            self.norm3 = InstanceNormalization(name=self.name + '/norm3', trainable=self.trainable)

        else:
            self.norm1 = BatchNormalization(name=self.name + '/norm1', trainable=self.trainable)
            self.norm2 = BatchNormalization(name=self.name + '/norm2', trainable=self.trainable)
            self.norm3 = BatchNormalization(name=self.name + '/norm3', trainable=self.trainable)

        if activation == 'swish':
            self.relu1 = Swish(name=self.name + '/activation1')
            self.relu2 = Swish(name=self.name + '/activation2')
            self.relu3 = Swish(name=self.name + '/activation3')
        elif activation == 'leaky_relu':
            self.relu1 = LeakyReLU(name=self.name + '/activation1')
            self.relu2 = LeakyReLU(name=self.name + '/activation2')
            self.relu3 = LeakyReLU(name=self.name + '/activation3')
        else:
            self.relu1 = ReLU(name=self.name + '/activation1')
            self.relu2 = ReLU(name=self.name + '/activation2')
            self.relu3 = ReLU(name=self.name + '/activation3')

        self.in_conv = Conv2D(self.out_channels // 2, kernel_size=1, name=self.name + '/in_conv', strides=1, kernel_constraint=conv_constraint,
                              trainable=self.trainable, padding='same', kernel_initializer=self.initializer)

        if mode == 'identity':
            # Keeps image dimensions (height and width) intact
            self.mid_conv = Conv2D(self.out_channels // 2, kernel_size=1, name=self.name + '/mid_conv', strides=1,
                                   trainable=self.trainable, padding='same', kernel_constraint=conv_constraint, kernel_initializer=self.initializer)

        elif mode == 'downsample':
            # Causes a reduction over image dimensions. The new dimensions are calculated as follows:
            #                   new_dim = floor((old_dim - kernel_size)/stride + 1)
            # where new_dim and old_dim are either image height or width
            self.mid_conv = Conv2D(self.out_channels // 2, kernel_size=self.kernel_size, name=self.name + '/mid_conv', strides=self.strides,
                                   trainable=self.trainable, padding='valid', kernel_constraint=conv_constraint, kernel_initializer=self.initializer)

        else:
            # Causes an increase over image dimensions. The new dimensions are calculated as follows:
            #                   new_dim = old_dim * stride + max(kernel_size - stride, 0)
            # where new_dim and old_dim are either image height or width
            self.mid_conv = Conv2DTranspose(self.out_channels // 2, kernel_size=self.kernel_size, name=self.name + '/mid_conv', strides=self.strides,
                                            trainable=self.trainable, padding='valid', kernel_constraint=conv_constraint, kernel_initializer=self.initializer)

        self.global_pool = None
        self.squeeze_dense1 = None
        self.squeeze_dense2 = None

        if self.squeeze_excitation:
            self.global_pool = GlobalAveragePooling2D(name=self.name + "/global_pool")
            self.squeeze_dense1 = Dense(self.out_channels // self.squeeze_ratio,
                                        activation='relu',
                                        kernel_initializer=self.initializer,
                                        kernel_constraint=dense_constraint,
                                        trainable=self.trainable,
                                        name=self.name + "/squeeze_dense1")
            self.squeeze_dense2 = Dense(self.out_channels,
                                        activation='sigmoid',
                                        kernel_constraint=dense_constraint,
                                        kernel_initializer=self.initializer,
                                        trainable=self.trainable,
                                        name=self.name + "/squeeze_dense2")

        self.out_conv = Conv2D(self.out_channels, kernel_size=1, name=self.name + '/out_conv', strides=1,
                               trainable=self.trainable, padding='same', kernel_constraint=conv_constraint, kernel_initializer=self.initializer)

    def build(self, input_shape):

        if self.mode == 'identity':

            if input_shape[-1] != self.out_channels:
                # This mode is used when the image dimensions (height and width) don't change, but only its channel dimension
                self.shortcut = Conv2D(self.out_channels, kernel_size=1, name=self.name + '/shortcut', strides=1,
                                       trainable=self.trainable, padding='same', kernel_constraint=self.conv_constraint, kernel_initializer=self.initializer)
            else:
                # If the shapes are equal then returns the input data itself
                self.shortcut = Lambda(lambda x: x, output_shape=input_shape, name=self.name + '/shortcut')

        elif self.mode == 'downsample':
            self.shortcut = Conv2D(self.out_channels, kernel_size=self.kernel_size, name=self.name + '/shortcut', strides=self.strides,
                                   trainable=self.trainable, padding='valid', kernel_constraint=self.conv_constraint, kernel_initializer=self.initializer)
        else:
            self.shortcut = Conv2DTranspose(self.out_channels, kernel_size=self.kernel_size, name=self.name + '/shortcut', strides=self.strides,
                                            trainable=self.trainable, padding='valid', kernel_constraint=self.conv_constraint, kernel_initializer=self.initializer)

    def call(self, input_tensor, training=True):

        norm1 = self.norm1(input_tensor, training=training)
        relu1 = self.relu1(norm1)
        in_conv = self.in_conv(relu1)

        norm2 = self.norm2(in_conv, training=training)
        relu2 = self.relu2(norm2)
        mid_conv = self.mid_conv(relu2)

        norm3 = self.norm3(mid_conv, training=training)
        relu3 = self.relu3(norm3)
        out_conv = self.out_conv(relu3)

        if self.squeeze_excitation:
            global_pool = self.global_pool(out_conv)
            squeeze_dense1 = self.squeeze_dense1(global_pool)
            squeeze_dense2 = self.squeeze_dense2(squeeze_dense1)

            out_conv = tf.keras.layers.Multiply()([out_conv, squeeze_dense2])

        shortcut = self.shortcut(input_tensor)

        add = Add(name=self.name + '/add')([out_conv, shortcut])

        return add

    def get_config(self):

        config = super().get_config()

        config.update({'out_channels': self.out_channels,
                       'initializer': self.initializer,
                       'mode': self.mode,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'trainable': self.trainable,
                       'normalization': self.normalization,
                       'groups': self.groups,
                       'squeeze_excitation': self.squeeze_excitation,
                       'squeeze_ratio': self.squeeze_ratio,
                       # 'conv_constraint': self.conv_constraint,
                       # 'dense_constraint': self.dense_constraint,
                       'name': self.name,
                       'norm1': get_object_config(self.norm1),
                       'norm2': get_object_config(self.norm2),
                       'norm3': get_object_config(self.norm3),
                       'relu1': get_object_config(self.relu1),
                       'relu2': get_object_config(self.relu2),
                       'relu3': get_object_config(self.relu3),
                       'global_pool': get_object_config(self.global_pool),
                       'squeeze_dense1': get_object_config(self.squeeze_dense1),
                       'squeeze_dense2': get_object_config(self.squeeze_dense2),
                       'in_conv': get_object_config(self.in_conv),
                       'mid_conv': get_object_config(self.mid_conv),
                       'out_conv': get_object_config(self.out_conv)})

        return config
