from tensorflow.keras.layers import Layer
import tensorflow as tf


def squash(tensor, axis=-1, epsilon=1e-7):
    squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = tensor / safe_norm
    return squash_factor * unit_vector

def inference(capsule_preds, routing_weights):

    routing_weights = tf.keras.activations.softmax(routing_weights, axis=2)

    weighted_predictions = tf.multiply(routing_weights, capsule_preds)

    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)

    return weighted_sum


class TrainRouting(Layer):

    def __init__(self,
                 primary_capsules,
                 primary_dims,
                 secondary_capsules,
                 secondary_dims,
                 squash_at_end=True,
                 loop_iterations=2,
                 name='routing_by_agreement',
                 **kwargs):
        super().__init__(name=name, trainable=True)

        self.primary_capsules = primary_capsules

        self.primary_dims = primary_dims

        self.secondary_capsules = secondary_capsules

        self.secondary_dims = secondary_dims

        self.loop_iterations = loop_iterations

        self.squash_at_end = squash_at_end

        w_shape = [primary_capsules, secondary_capsules, secondary_dims, primary_dims]

        init_sigma = 0.1
        self.capsule_weights = self.add_weight(initializer=tf.keras.initializers.RandomNormal(stddev=init_sigma),
                                               shape=w_shape,
                                               dtype=tf.float32,
                                               trainable=True,
                                               name=self.name + '/capsule_weights')

        routing_weights_shape = [1, primary_capsules, secondary_capsules, 1, 1]
        self.routing_weights = self.add_weight(initializer=tf.zeros_initializer(),
                                               shape=routing_weights_shape,
                                               dtype=tf.float32,
                                               trainable=False,
                                               name=self.name + '/routing_weights')

    def routing(self, capsule_preds):

        capsule_preds_stopped = tf.stop_gradient(capsule_preds)

        counter = tf.constant(0, dtype=tf.int32)
        while tf.less(counter, self.loop_iterations - 1):

            weighted_sum = inference(capsule_preds_stopped, self.routing_weights)

            output_round = squash(weighted_sum, axis=-2)

            output_round_tiled = tf.tile(output_round, [1, self.primary_capsules, 1, 1, 1])

            agreement = tf.matmul(capsule_preds_stopped, output_round_tiled, transpose_a=True)

            mean_agreement = tf.reduce_mean(agreement, axis=0, keepdims=True)

            self.routing_weights.assign_add(mean_agreement)

            counter += 1

    def call(self, caps1_raw, training=True):

        caps1_raw = tf.reshape(caps1_raw, [-1, self.primary_capsules, self.primary_dims])

        caps1_output = squash(caps1_raw)

        caps1_output_expanded = tf.expand_dims(caps1_output, -1)

        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)

        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.secondary_capsules, 1, 1])

        caps2_predicted = tf.matmul(self.capsule_weights, caps1_output_tiled)

        self.routing(caps2_predicted)

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
                      loop_iterations=self.loop_iterations,
                      squash_at_end=self.squash_at_end)

        return config

class TestRouting(Layer):

    def __init__(self,
                 primary_capsules,
                 primary_dims,
                 secondary_capsules,
                 secondary_dims,
                 squash_at_end=True,
                 name='routing_by_agreement',
                 **kwargs):
        super().__init__(name=name, trainable=False)

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
