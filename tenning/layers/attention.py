from tenning.generic_utils import get_object_config
import tensorflow.keras.constraints as constraints
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
import tensorflow as tf


class AttentionMechanism(Layer):

    def __init__(self,
                 out_channels,
                 strides=1,
                 kernel_size=1,
                 initializer='he_normal',
                 queries_ratio=8,
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        conv_constraint = kwargs.get('conv_constraint', None)
        constraint_arguments = kwargs.get('conv_constraint_arguments', [])

        assert out_channels >= queries_ratio, f"The number of output channels must be greater or equal to 'queries_ratio'"

        if constraint_arguments:
            if not isinstance(constraint_arguments, list):
                raise TypeError(f"'constraint_arguments' must be a list")

        if conv_constraint:
            conv_constraint = getattr(constraints, conv_constraint, None)(*constraint_arguments)

        self.out_channels = out_channels
        self.initializer = initializer
        self.kernel_size = kernel_size
        self.strides = strides
        self.queries_ratio = queries_ratio
        self.conv_constraint = conv_constraint
        self.kernel_constraint = conv_constraint

        self.keys = Conv2D(out_channels // queries_ratio, kernel_size=1, strides=1, name=self.name + '/keys',
                           trainable=trainable, kernel_constraint=conv_constraint, kernel_initializer=initializer)

        self.queries = Conv2D(out_channels // queries_ratio, kernel_size=1, strides=1, name=self.name + '/queries',
                              trainable=trainable, kernel_constraint=conv_constraint, kernel_initializer=initializer)

        self.values = Conv2D(out_channels // queries_ratio, kernel_size=1, strides=1, name=self.name + '/values',
                             trainable=trainable, kernel_constraint=conv_constraint, kernel_initializer=initializer)

        self.output_conv = Conv2D(out_channels, kernel_size=kernel_size, strides=strides, name=self.name + '/output_conv',
                                  trainable=trainable, kernel_constraint=conv_constraint, kernel_initializer=initializer)

        self.softmax = Softmax(name=self.name + '/softmax')

    def build(self, input_shape):

        self.gamma = self.add_weight(shape=[1], initializer=tf.zeros_initializer(), trainable=self.trainable, name=self.name + '/gamma')

        input_channels = input_shape[-1]
        old_dim = tf.constant(input_shape[1:3])
        new_dim = tf.cast(tf.floor((old_dim - self.kernel_size) / self.strides + 1), tf.int32)
        output_shape = [input_shape[0], new_dim[0], new_dim[1], self.out_channels]

        if input_channels != self.out_channels:
            # This mode is used when the image dimensions (height and width) don't change, but only its channel dimension
            self.shortcut = Conv2D(self.out_channels, kernel_size=self.kernel_size, name=self.name + '/shortcut', kernel_constraint=self.kernel_constraint,
                                   strides=self.strides, trainable=self.trainable, kernel_initializer=self.initializer)
        else:
            # If the shapes are equal then returns the input data itself
            self.shortcut = Lambda(lambda x: x, output_shape=output_shape, name=self.name + '/shortcut')

    def call(self, input_tensor, training=True):

        batch = tf.shape(input_tensor)[0]
        height = tf.shape(input_tensor)[1]
        width = tf.shape(input_tensor)[2]

        keys = self.keys(input_tensor)

        queries = self.queries(input_tensor)

        values = self.values(input_tensor)

        flattened_keys = self._flatten_tensor(keys)
        flattened_queries = self._flatten_tensor(queries)
        flattened_values = self._flatten_tensor(values)

        energy = tf.matmul(flattened_queries, flattened_keys, transpose_b=True)
        # flattened_queries_t = tf.transpose(flattened_queries, perm=[0, 2, 1])
        # energy = tf.keras.backend.batch_dot(flattened_queries_t, flattened_keys, axes=(1, 2))

        attention_maps = self.softmax(energy)

        feat_maps = tf.matmul(attention_maps, flattened_values)
        # feat_maps = tf.keras.backend.batch_dot(flattened_values, attention_maps, axes=(1, 2))

        feat_maps = tf.reshape(feat_maps, shape=[batch, height, width, self.out_channels // self.queries_ratio])

        output = self.output_conv(feat_maps)

        output = self.gamma * output

        # Residual connection
        shortcut = self.shortcut(input_tensor)

        return Add(name=self.name + '/add')([output, shortcut])

    def _flatten_tensor(self, x):
        """ Returns a tensor with shape [batch, N, channels], where N is the number of features"""

        batch = tf.shape(x)[0]
        channels = tf.shape(x)[-1]

        return tf.reshape(x, shape=[batch, -1, channels])

    def get_config(self):

        config = super().get_config()

        config.update({'out_channels': self.out_channels,
                       'initializer': self.initializer,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'queries_ratio': self.queries_ratio,
                       'trainable': self.trainable,
                       'name': self.name,
                       'keys': get_object_config(self.keys),
                       'queries': get_object_config(self.queries),
                       'values': get_object_config(self.values),
                       'softmax': get_object_config(self.softmax),
                       'output_conv': get_object_config(self.output_conv)})

        return config


class MultiHeadAttention(Layer):

    def __init__(self,
                 out_channels,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 drop=0.1,
                 initializer='he_normal',
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.temperature = d_k ** 0.5
        self.drop = drop
        self.d_k = d_k
        self.d_v = d_v
        self.initializer = initializer

        self.keys = Dense(num_heads*d_k, name=self.name + '/keys', trainable=trainable, use_bias=False, kernel_initializer=initializer)

        self.queries = Dense(num_heads*d_k, name=self.name + '/queries', trainable=trainable, use_bias=False, kernel_initializer=initializer)

        self.values = Dense(num_heads*d_v, name=self.name + '/values', trainable=trainable, use_bias=False, kernel_initializer=initializer)

        self.out = Dense(out_channels, name=self.name + '/out', trainable=trainable, use_bias=False, kernel_initializer=initializer)

        self.softmax = Softmax(name=self.name + '/softmax')

        self.dropout1 = tf.keras.layers.Dropout(self.drop)
        self.dropout2 = tf.keras.layers.Dropout(self.drop)

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input_tensor):

        batch = input_tensor.shape[0]

        keys = self.keys(input_tensor)

        queries = self.queries(input_tensor)

        values = self.values(input_tensor)

        queries = tf.reshape(queries, [batch, -1, self.num_heads, self.d_k])
        keys = tf.reshape(keys, [batch, -1, self.num_heads, self.d_k])
        values = tf.reshape(values, [batch, -1, self.num_heads, self.d_v])

        queries = tf.transpose(queries, [0, 2, 1, 3])
        keys = tf.transpose(keys, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])

        attn = tf.matmul(queries / self.temperature, tf.transpose(keys, [0, 1, 3, 2]))

        attention_maps = self.softmax(attn)

        attn = self.dropout1(attention_maps)

        output = tf.matmul(attn, values)

        output = tf.transpose(output, [0, 2, 1, 3])

        q = tf.reshape(output, [batch, -1, self.d_v*self.num_heads])

        q = self.out(q)

        q = self.dropout2(q) + input_tensor

        return self.norm(q)

    def get_config(self):

        config = super().get_config()

        config.update({'out_channels': self.out_channels,
                       'initializer': self.initializer,
                       'num_heads': self.num_heads,
                       'temperature': self.temperature,
                       'drop': self.drop,
                       'd_k': self.d_k,
                       'd_v': self.d_v,
                       'trainable': self.trainable,
                       'name': self.name,
                       'keys': get_object_config(self.keys),
                       'queries': get_object_config(self.queries),
                       'values': get_object_config(self.values),
                       'softmax': get_object_config(self.softmax),
                       'dropout1': get_object_config(self.dropout1),
                       'dropout2': get_object_config(self.dropout2),
                       'norm': get_object_config(self.norm)})

        return config
