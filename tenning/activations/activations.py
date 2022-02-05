from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
import tensorflow as tf


class Swish(Layer):
    """ Swish activation function (https://arxiv.org/abs/1710.05941).
        It allows a small gradient when the unit is not active:
        'f(x) = x * sigmoid(beta * x)'
        Input shape:
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.
        Output shape:
            Same shape as the input.
        Args:
            beta: Float > 0. slope coefficient.
    """

    def __init__(self, beta=1, **kwargs):
        super().__init__(**kwargs)

        if beta <= 0:
            print("'beta' must be a real value greater than zero, setting to its default value (beta = 1).")
            beta = 1

        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):

        features = tf.convert_to_tensor(inputs, name="features")

        return features * tf.math.sigmoid(self.beta * features)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class PReLU(Layer):
    """Parametric Rectified Linear Unit.

    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):

        self.alpha = self.add_weight(shape=[1], name='alpha')

        self.built = True

    def call(self, inputs):
        pos = K.relu(inputs)
        neg = -self.alpha * K.relu(-inputs)
        return pos + neg

    def get_config(self):
        return super().get_config()

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class Biternion(Layer):
    """ Biternion output layer (http://www.spencer.eu/papers/beyerBiternions15.pdf).
        It outputs a a natural alternative representation of an angle by the
        two-dimensional vector consisting of its sine and cosine out = (cos phi, sin phi).

        Input shape:
            A (batch, features) shaped tensor.
        Output shape:
            A (batch, 2) shaped tensor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dense = Dense(2)

    def call(self, inputs):

        features = self.dense(inputs)

        return tf.linalg.normalize(features, axis=1)[0]

    def get_config(self):
        return super().get_config()

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class HintonSquash(Layer):
    """ Squash activation function (https://arxiv.org/abs/1710.09829).
        Derived from Dynamic Routing Between Capsules paper, it ensures that short
        vectors get shrunk to almost zero length and long vectors get shrunk to a
        length slightly below 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eps = 1e-7

    def call(self, inputs, axis=-1):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + self.eps)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = squash_factor / safe_norm
        return inputs * unit_vector

    def get_config(self):
        return super().get_config()

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class EfficientSquash(Layer):
    """ Squash activation function (https://arxiv.org/abs/2101.12491).
        Derived from Efficient-CapsNet: Capsule Network With Self-Attention Routing paper, it ensures that short
        vectors get shrunk to almost zero length and long vectors get shrunk to a
        length slightly below 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eps = 1e-7

    def call(self, inputs, axis=-1):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        unit_vector = inputs / (norm + self.eps)

        return (1 - 1/(tf.math.exp(norm)+self.eps)) * unit_vector

    def get_config(self):
        return super().get_config()

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

# class SOTransform(Layer):
#     """Swish activation function (https://arxiv.org/abs/1710.05941).
#     It allows a small gradient when the unit is not active:
#     `f(x) = x * sigmoid(beta * x)`
#     Input shape:
#       Arbitrary. Use the keyword argument `input_shape`
#       (tuple of integers, does not include the samples axis)
#       when using this layer as the first layer in a model.
#     Output shape:
#       Same shape as the input.
#     Arguments:
#       beta: Float > 0. slope coefficient.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inputs):

#         features = tf.convert_to_tensor(inputs, name="features")

#         dim = features.shape[1]

#         # Assuming features is a matrix of shape [B, N, N, C]
#         features = tf.transpose(features, [0, 3, 1, 2])

#         exp = tf.math.exp(-tf.linalg.trace(features)/dim)[..., tf.newaxis, tf.newaxis]

#         transformed = exp * tf.linalg.expm(features)

#         return tf.transpose(transformed, [0, 2, 3, 1])

#     def get_config(self):
#         return super().get_config()

#     @tf_utils.shape_type_conversion
#     def compute_output_shape(self, input_shape):
#         return input_shape
