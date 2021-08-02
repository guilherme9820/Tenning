from tensorflow.keras.layers import Layer
import tensorflow as tf


def squash(tensor, axis=-1, epsilon=1e-7):
    squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = squash_factor / safe_norm
    return tensor * unit_vector


class CapsuleLayer1D(Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 trainable=True, kernel_initializer='glorot_uniform',
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        message = "The input Tensor should have shape=[batch_size, input_num_capsule, input_dim_capsule]"
        tf.debugging.assert_greater_equal(len(input_shape), 3, message)

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.weights = self.add_weight(shape=(1, self.num_capsule, self.input_num_capsule,
                                              self.dim_capsule, self.input_dim_capsule),
                                       initializer=self.kernel_initializer,
                                       name='weights')

        self.built = True

    def inference(self, inputs, logits):

        route = tf.nn.softmax(logits, axis=1)
        routing = tf.squeeze(tf.matmul(route, inputs), axis=-2)
        return squash(routing)

    def routing(self, inputs_hat):

        inputs_hat_stopped = tf.stop_gradient(inputs_hat)

        logits = tf.zeros([tf.shape(inputs_hat)[0], self.num_capsule, 1, self.input_num_capsule])
        outputs = tf.zeros([tf.shape(inputs_hat)[0], self.num_capsule, self.dim_capsule])

        counter = tf.constant(0, dtype=tf.int32)
        while tf.less(counter, self.num_routing):
            # route = tf.nn.softmax(logits, axis=1)

            # At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            if counter == self.num_routing - 1:
                # [None, num_capsule, dim_capsule]
                outputs = self.inference(inputs_hat, logits)
            else:
                # Otherwise, use 'inputs_hat_stopped' to update 'logits'. No gradients flow on this path.
                # [None, num_capsule, dim_capsule]
                outputs = self.inference(inputs_hat_stopped, logits)

                # [None, num_capsule, 1, input_num_capsule]
                distances = tf.matmul(tf.expand_dims(outputs, axis=-2), inputs_hat_stopped, transpose_b=True)
                logits += distances

            counter += 1

        return outputs

    def call(self, inputs, training=True):

        # [batch_size, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = tf.expand_dims(inputs, 1)

        # [batch_size, 1, input_num_capsule, 1, input_dim_capsule]
        inputs_expand = tf.expand_dims(inputs_expand, -2)

        # [batch_size, num_capsule, input_num_capsule, 1, input_dim_capsule]
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])

        # [batch_size, num_capsule, input_num_capsule, 1, dim_capsule]
        inputs_hat = tf.matmul(inputs_tiled, self.weights, transpose_b=True)

        # [batch_size, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.squeeze(inputs_hat, axis=-2)

        if training:
            outputs = self.routing(inputs_hat)
        else:
            logits = tf.zeros([tf.shape(inputs_hat)[0], self.num_capsule, 1, self.input_num_capsule])
            outputs = self.inference(inputs_hat, logits)

        return outputs

    def get_config(self):

        config = super().get_config()

        config.update({'num_capsule': self.num_capsule,
                       'dim_capsule': self.dim_capsule,
                       'num_routing': self.num_routing,
                       'trainable': self.trainable,
                       'name': self.name})

        return config


        self.primary_capsules = primary_capsules

        self.primary_dims = primary_dims

        self.secondary_capsules = secondary_capsules

        self.secondary_dims = secondary_dims

        self.squash_at_end = squash_at_end

        w_shape = [primary_capsules, secondary_capsules, secondary_dims, primary_dims]

        init_sigma = 0.1
        self.capsule_weights = self.add_weight(initializer=tf.keras.initializers.RandomNormal(stddev=init_sigma),
                                               shape=w_shape,
                                               dtype=tf.float32,
                                               trainable=False,
                                               name=self.name + '/capsule_weights')

        routing_weights_shape = [1, primary_capsules, secondary_capsules, 1, 1]
        self.routing_weights = self.add_weight(initializer=tf.zeros_initializer(),
                                               shape=routing_weights_shape,
                                               dtype=tf.float32,
                                               trainable=False,
                                               name=self.name + '/routing_weights')

    def call(self, caps1_raw, training=True):

        caps1_raw = tf.reshape(caps1_raw, [-1, self.primary_capsules, self.primary_dims])

        caps1_output = squash(caps1_raw)

        caps1_output_expanded = tf.expand_dims(caps1_output, -1)

        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)

        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.secondary_capsules, 1, 1])

        caps2_predicted = tf.matmul(self.capsule_weights, caps1_output_tiled)

        weighted_sum = inference(caps2_predicted, self.routing_weights)

        if self.squash_at_end:
            return squash(weighted_sum, axis=-2)

        return weighted_sum

    def get_config(self):

        config = super().get_config()

        config.update(primary_capsules=self.primary_capsules,
                      primary_dims=self.primary_dims,
                      secondary_capsules=self.secondary_capsules,
                      secondary_dims=self.secondary_dims,
                      squash_at_end=self.squash_at_end)

        return config

def RoutingByAgreement(*args, **kwargs):

    mode = kwargs.get('trainable', True)

    if mode:
        return TrainRouting(*args, **kwargs)

    return TestRouting(*args, **kwargs)
